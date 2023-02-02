# Manages the stocks, calculate beta, alpha and their cov and all
import pandas as pd
import numpy as np
from tabulate import tabulate
import copy
import tqdm
from portfolio_optimizer.style import cprint


class Portfolio:
    def __init__(self, freq,  benchmark: dict, risk_free_rate=None) -> None:
        """
        Initializes the Portfolio class

        Parameters
        ----------
        benchmark : dict, compulsory
            Dictionary of the benchmark stock

        freq : str, compulsory
            Frequency of the data which will be further used for the calculations

        risk_free_rate : pd.DataFrame, optional
            Risk free rate of the market, by default None then taking it as 0.225

        Attributes
        -----------

        market : str
        Index representing the market, like snp500 or nifty50

        stocks : dic
        Name of the Stock --> Stock dataframe
        benchmark must be given at the time of instatiation

        market_return : Int
        Returns of the market

        merged_stocks : pd.DataFrame
        Merged dataframe of all the stocks, by default there is only benchmark

        betas : pd.DataFrame
        DataFrame of Beta of all the stocks

        alphas : pd.DataFrame
        DataFrame of Alpha of all the stocks

        returns : pd.DataFrame
        DataFrame of Returns of all the stocks

        Eg :
        apple = pd.read_csv(f"data/AAPL.csv", index_col="Date", parse_dates=True)
        stocks['APPLE'] = apple

        Returns
        -------
        None
        """

        benchmark = copy.deepcopy(benchmark)

        self.market = list(benchmark.keys())[0]

        # Check whether a given benchmark have the pd.Datetime as index or not
        if isinstance(benchmark[self.market].index, pd.DatetimeIndex):
            pass
        else:
            benchmark[self.market].index = benchmark[self.market]["Date"]
            benchmark[self.market].index = pd.to_datetime(benchmark[self.market].index)
            benchmark[self.market].drop("Date", axis=1, inplace=True)


        self.stocks = benchmark
        self.freq = freq
        if risk_free_rate is None:
            self.risk_free_rate = 0.225
        elif isinstance(risk_free_rate, float):
            self.risk_free_rate = risk_free_rate
        else:
            self.risk_free_rate = risk_free_rate["Close"].asfreq(self.freq, method="ffill").pct_change().dropna().mean()*100
        self.merged_stocks = self.stocks[self.market]
        self.market_return = self.merged_stocks.asfreq(self.freq, method="ffill").pct_change().dropna()["Close"].mean()*100 # monthly returns of the market
        self.betas = None
        self.alphas = None
        self.returns = None

        cprint.print(f"Portfolio Created with {self.market} as the benchmark", "blue")
        

    
    def __repr__(self) -> str:
        """
        prints list of stocks in the portfolio when the object is called as it is
        """
        return str(f"Stocks: {self.stocks.keys()}")

    def __str__(self):
        """
        Prints the list of the stocks in the Portfolio
        """
        return str(list(self.stocks.keys()))

    def __len__(self):
        """
        Returns the number of stocks in the Portfolio except the benchmark
        """
        return len(self.stocks)-1

    def __update(self):
        """
        Updates a stock in the dictionary

        Parameters
        ----------
        name : str
            Name of the stock
        stock : pd.DataFrame
            Stock dataframe

        Returns
        -------
        None
        """
        self.merged_stocks = self.merge_dfs()
        self.returns = self.cal_returns()
        self.betas = self.cal_beta()
        self.alphas = self.cal_alpha()

    def portfolio_stocks(self ):
        return [key for key in self.stocks.keys() if key not in [self.market]]

    def merge_dfs(self, stocks=None, join="inner", columns=["Close"]):
        """
        Merges a list of dataframes into one. Uses the index as the key and `pd.concat` to merge the dataframes

        Parameters
        ----------
        stocks : dictionary
            keys : Stocks name
            values : Stocks DataFrame
        join : str, optional
            How to join the dataframes, by default "inner"
        columns : str, optional
            Which columns to merge, by default "Close"
            
        Returns
        -------
        pd.DataFrame
            Merged dataframe
        """
        if stocks:
            pass
        else:
            stocks = self.stocks

        df_names = stocks.keys()
        dfs = stocks.values()
        
        dfs = [df[columns] for df in dfs]
        df = pd.concat(dfs, axis=1, join=join)
        if df_names:
            df.columns = df_names
        else:
            df.columns = [f"df_{i}" for i in range(len(dfs))]

        self.merged_stocks = df
        return df

    def cal_returns(self, df=None, freq="M"):

        """
        Calculates the returns of a dataframe
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to calculate returns
        Returns
        -------
        pd.DataFrame
            Returns
        """
        if df is None:
            df = self.merged_stocks
            freq = self.freq
        else:
            freq = freq

        self.returns = df.asfreq(freq=freq , method="ffill").pct_change().dropna()
        return self.returns

    def add_stock(self, stock_dic):
        """
        Adds a stock to the dictionary

        Parameters
        ----------

        stock_dic : dict
            stock_dic = {name : stock}
                name : Name of the stock
                stock : Stock dataframe

        Returns
        -------
        None
        """
        for value in stock_dic.values():
            if isinstance(value.index, pd.DatetimeIndex):
                pass
            else:
                value.index = value["Date"]
                value.index = pd.to_datetime(value.index)
                value.drop("Date", axis=1, inplace=True)
        self.stocks.update(stock_dic)
        self.__update()

        # print that stock has been added
        cprint.print(f"{list(stock_dic.keys())} has been added to the portfolio", "Magenta")


    
    def __setitem__(self, name, stock):
        """
        Adds a stock to the dictionary

        Parameters
        ----------
        name : str
            Name of the stock
        stock : pd.DataFrame
            Stock dataframe

        Returns
        -------
        None

        Eg :
        -------
        obj["Apple"] = apple
        """

        if isinstance(stock.index, pd.DatetimeIndex):
            pass
        else:
            stock.index = stock["Date"]
            stock.index = pd.to_datetime(stock.index)
            stock.drop("Date", axis=1, inplace=True)
        self.stocks[name] = stock
        self.__update()

        # print that stock has been added
        cprint.print(f"{name} has been added to the portfolio", "Magenta")
        

    def remove_stock(self, names):
        """
        Removes a stock from the dictionary

        Parameters
        ----------
        names : list
            Names of the stocks

        Returns
        -------
        None
        """
        for name in names:
            if  name in [self.market]:
                cprint.print(f"You can't remove your benchmark from the portfolio", "warning")
            elif name in self.stocks.keys():
                self.stocks.pop(name)
                self.__update()
                cprint.print(f"{name} stock is removed from Portfolio", "Magenta")
            else:
                cprint.print(f"{name} stock is not present in Portfolio", "warning")
        
        

    def __delitem__(self, name):
        """
        Removes a stock from the dictionary

        Parameters
        ----------
        name : str
            Name of the stock

        Returns
        -------
        None

        Eg :
        --------
        del obj["apple"]
        """

        if  name in [self.market]:
            cprint.print(f"You can't remove your benchmark from the portfolio", "warning")
        elif name in self.stocks.keys():
            self.stocks.pop(name)
            self.__update()
            cprint.print(f"{name} stock is removed from Portfolio", "Magenta")
        else:
            cprint.print(f"{name} stock is not present in Portfolio", "warning")

    
    def __getitem__(self, name):
        """
        Returns the stock dataframe

        Parameters
        ----------
        name : str
            Name of the stock

        Returns
        -------
        pd.DataFrame
            Stock dataframe

        Eg :
        --------
        obj["apple"]
        """
        if name in self.stocks.keys():
            return self.stocks[name]
        else:
            cprint.print(f"{name} stock is not present in Portfolio", "red")


    

    def cal_beta(self, df=None, benchmark="snp500", freq="M"):
        """
        Calculates the alpha and beta of a dataframe
        Parameters
        ----------
        df : pd.DataFrame
            Merged Dataframe to calculate alpha and beta
        benchmark : pd.DataFrame
            Benchmark to calculate alpha and beta, by default "snp500"
        Returns
        -------
        pd.DataFrame
            Beta
        """
        if df is None:
            df = self.merged_stocks
            returns = self.returns
            benchmark = self.market
            freq = self.freq
        else:
            freq = freq
            returns = self.cal_returns(df, freq=freq)
            benchmark = benchmark
            
        betas = pd.DataFrame()
        for col in returns.columns:
            if col != benchmark:
                betas.loc[col, "beta"] = returns[col].corr(returns[benchmark]) * (returns[col].std()/returns[benchmark].std())
        
        self.betas = betas
        return betas

    def cal_alpha(self, df=None, benchmark="snp500", freq="M"):
        """
        Calculates the alpha and beta of a dataframe
        Parameters
        ----------
        df : pd.DataFrame
            Merged Dataframe to calculate alpha and beta
        benchmark : pd.DataFrame
            Benchmark to calculate alpha and beta, by default "snp500"
        Returns
        -------
        pd.DataFrame
            Alpha
        """
        if df is None:
            df = self.merged_stocks
            returns = self.returns
            betas = self.betas
            benchmark = self.market
            freq = self.freq
        else:
            freq = freq
            returns = self.cal_returns(df, freq=freq)
            betas = self.cal_beta(df, benchmark=benchmark, freq=freq)
            benchmark = benchmark
            

        alphas = pd.DataFrame()
        for col in returns.columns:
            if col != benchmark:
                alphas.loc[col, "alpha"] = returns[col].mean() - betas.loc[col, "beta"] * returns[benchmark].mean()

        self.alphas = alphas
        return alphas
    
    def cov_matrix(self, df=None, freq="M"):
        """
        Calculates the covariance matrix of a dataframe
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to calculate covariance matrix
        Returns
        -------
        pd.DataFrame
            Covariance matrix
            
        """
        if df is None:
            df = self.merged_stocks
            returns = self.returns
            freq = self.freq
        else:
            freq = freq
            returns = self.cal_returns(df, freq=freq)
        
        return returns.cov()

    def expected_returns(self, risk_free_rate=0.225, market_return=0.98,  beta=None, model="CAPM"):
        
        """
        
        Parameters
        ----------
        risk_free_rate : float
        
        
        market_return : float
        
        
        beta : pd.DataFrame, optional
            Beta of the stocks, by default None 


        model : str, optional
            Model to calculate expected returns, by default "CAPM"

        Returns
        -------
        pd.DataFrame
            Expected returns of the stocks

        Eg :
        -------
        obj.expected_returns(0.225, 0.89, beta=np.array([0.5, 0.6]))

        """

        if beta is None:
            if self.betas.shape[0] == 0:
                self.betas = self.cal_beta()
            beta = np.array(self.betas['beta']) 
            risk_free_rate = self.risk_free_rate
            market_return = self.market_return
            if self.alphas.shape[0] == 0:
                self.alphas = self.cal_alpha()
            alpha = np.array(self.alphas['alpha'])

        else:
            freq = freq

        if model.upper() == "CAPM":
            excepted_returns = risk_free_rate + np.multiply(beta, (market_return - risk_free_rate))
        elif model.upper() == "SIM":
            excepted_returns = alpha + risk_free_rate + np.multiply(beta, (market_return - risk_free_rate))
        return excepted_returns


    def portfolio_expected_return(self, weights="equals", expected_returns=None, model="CAPM"):
        """
        Calculates the portfolio return

        Parameters
        ----------
        weights : np.array
            Weights of the stocks
            or "equals" for equal weights

        expected_returns : np.array
            Expected returns of the stocks

        model : str, optional
            Model to calculate expected returns, by default "CAPM"

        Returns
        -------
        float
            Portfolio return

        Eg :
        -------
        obj.portfolio_return(weights, expected_returns)
        """
        
        if expected_returns is None:
            expected_returns = self.expected_returns(model=model)
        
        if type(weights) == str:
            weights = np.array([1/len(expected_returns)]*len(expected_returns))
        else:
            weights = weights

        return np.dot(weights, expected_returns)

    def portfolio_variance(self, weights="equals", cov_matrix=None):
        """
        Calculates the portfolio variance

        Parameters
        ----------
        weights : np.array
            Weights of the stocks
            or "equals" for equal weights

        cov_matrix : pd.DataFrame
            Covariance matrix of the stocks

        Returns
        -------
        float
            Portfolio variance

        Eg :
        -------
        obj.portfolio_variance(weights, cov_matrix)
        """
        if cov_matrix is None:
            cov_matrix = self.cov_matrix().drop(columns=self.market).drop(index=self.market).to_numpy()
        else:
            cov_matrix = cov_matrix.to_numpy()

        
        if isinstance(weights, str):
            weights = np.array([1/len(cov_matrix)]*len(cov_matrix))

        return np.dot(weights, np.dot(cov_matrix, weights))*100

    def portfolio_std(self, weights="equals", cov_matrix=None):
        """
        Calculates the portfolio standard deviation

        Parameters
        ----------
        weights : np.array
            Weights of the stocks
            or "equals" for equal weights

        cov_matrix : pd.DataFrame
            Covariance matrix of the stocks

        Returns
        -------
        float
            Portfolio standard deviation

        Eg :
        -------
        obj.portfolio_std(weights, cov_matrix)
        """

        return np.sqrt(self.portfolio_variance(weights, cov_matrix))

    def portfolio_summary(self,  expected_returns=None, cov_matrix=None):
        """
        Calculates the portfolio summary

        Parameters
        ----------
        weights : np.array
            Weights of the stocks
            or "equals" for equal weights

        expected_returns : np.array
            Expected returns of the stocks

        cov_matrix : pd.DataFrame
            Covariance matrix of the stocks

        Returns
        -------
        pd.DataFrame
            Portfolio summary

        Eg :
        -------
        obj.portfolio_summary(weights, expected_returns, cov_matrix)
        """
        if expected_returns is None:
            expected_returns = self.expected_returns()
        if cov_matrix is None:
            cov_matrix = self.cov_matrix().drop(columns=self.market).drop(index=self.market)
        else:
            cov_matrix = cov_matrix

        headers = [key for key in self.stocks.keys() if key not in [self.market]]

        cprint.print("Portfolio Summary", "header")
        cprint.print("*****************\n", "header")
        cprint.print(f"Stocks in the Portfolio : ", "blue", end="")
        cprint.print("*************************", "blue")
        cprint.print({str(headers)}, "green")
        print()
        
        cprint.print("Beta :", "blue")
        cprint.print("******", "blue")
        cprint.print(tabulate(self.betas.T, headers,  tablefmt="orgtbl"), "green")
        print()


        cprint.print("Expected Returns :", "blue")
        cprint.print("******************", "blue")
        cprint.print(tabulate(pd.DataFrame(expected_returns).T, headers, tablefmt="orgtbl"), "green")
        print()



        cprint.print("The covariance matrix is as follows", "blue")
        cprint.print("***********************************", "blue")
        cprint.print(tabulate(cov_matrix, headers=headers, tablefmt="orgtbl"), "green")
        print()

        cprint.print(f"Portfolio Returns at equals weights: {self.portfolio_expected_return(weights='equals')}", "MAGENTA")
        cprint.print(f"Portfolio Risk at equals weights: {self.portfolio_variance(weights='equals')}", "MAGENTA")
        