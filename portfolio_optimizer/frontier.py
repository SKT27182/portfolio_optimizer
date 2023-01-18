# Make a Efficient Frontier

from portfolio_optimizer.portfolio import Portfolio
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as py
py.init_notebook_mode(connected=True)

import numpy as np
import pandas as pd

import warnings

pd.options.mode.chained_assignment = None


class CurveFitting:
    """
    A class to implement mth order polynomial regression using the least squares method.
    Use the `fit` method to fit the model. Then predict the Y values given X values using\\
    the `predict` method.
    """

    def __init__(self) -> None:
        self.beta = None
        self.stats = None

    def fit(self, X, Y, order=3):
        """
        Polynomial regression of order m using least squares method.
        Parameters
        ----------
        X : array_like
            Independent variable.
        Y : array_like
            Dependent variable.
        order : int, optional
            Order of the polynomial. Default is 3.
        Returns
        -------
        beta : array_like
            Coefficients of the polynomial regression model.
        """
        self.n = len(X)
        Xis = np.zeros(2 * order + 1)
        Yis = np.zeros(order + 1)
        for i in range(0, 2 * order + 1):
            if i == 0:
                Xis[i] = self.n
                continue
            xi = np.sum(X**i)
            Xis[i] = xi

        for i in range(1, order + 2):
            yi = np.sum(Y * (X ** (i - 1)))
            Yis[i - 1] = yi
        A = np.zeros((order + 1, order + 1))
        for i in range(0, order + 1):
            A[i] = Xis[i : i + order + 1]
        beta = np.linalg.solve(A, Yis)
        self.beta = beta
        return beta

    def predict(self, X_l):
        """
        Predict the Y values given X values.
        Parameters
        ----------
        X_l : array_like
            Independent variable.
        Returns
        -------
        Y_l : array_like
            Predicted Y values.
        """
        Y_l = np.zeros(len(X_l))
        for i in range(0, len(self.beta)):
            Y_l += self.beta[i] * X_l**i
        return Y_l


class EfficientFrontier:
    def __init__(self):
        pass

    def add_portfolio(self, portfolio:Portfolio):
        """
        
        Parameters
        ----------
        portfolio : Portfolio
            Portfolio object.

        Returns
        -------
        None, just add the portfolio to the EfficientFrontier object.
        
        """
        self.portfolio = portfolio

    def __cal_required_parameters(self, weights, model):

        """
        Calculate the required parameters for the efficient frontier.
        Parameters
        ----------
        weights : array_like
            Weights of the portfolio.
        model : str
            The model to use for calculating the expected returns. Default is "mean".

        Returns
        -------
        p_std : float
            Standard deviation of the portfolio.
        p_returns : float
            Expected returns of the portfolio.


        """

        p_std = self.portfolio.portfolio_std(weights=weights)
        p_returns = self.portfolio.portfolio_expected_return(weights=weights, model=model)

        return p_std, p_returns

    def __simulate_frontier(self, model, short=False):

        """
        Simulate the efficient frontier.
        Parameters
        ----------
        model : str
            The model to use for calculating the expected returns. Default is "mean".
        short : bool, optional
            Whether to allow shorting of stocks. Default is False.

        Returns
        -------
        data : pandas.DataFrame
            Dataframe containing the simulated portfolios.
        weights : array_like
            Weights of the simulated portfolios.

        """

        n = 7000

        if short:
            weights = np.random.uniform(low=-2.0, high=2.0, size=(n, len(self.portfolio) ) )
            # weights /= np.sum(weights)
        else:
            weights = np.random.uniform(low=0.0, high=1.0, size=(n, len(self.portfolio) ) )
            # weights /= np.sum(weights)
            
            
        for _ in range(n):
            weights[_] /= np.sum(weights[_])


        p_returns = np.zeros(len(weights))
        p_risk = np.zeros(len(weights))
        sharpe_ratios = np.zeros(len(weights))
        for i in range(n):
            p_std_, p_returns_ = self.__cal_required_parameters(weights[i], model)
            p_returns[i] = p_returns_
            p_risk[i] = p_std_
            sharpe_ratios[i] = (p_returns_ - self.portfolio.risk_free_rate) / p_std_


        data = pd.DataFrame(
            {
                "Risk": p_risk,
                "Returns": p_returns,
                "Sharpe ratio": sharpe_ratios,
            }
        )

        return data, weights
        

    def plot_sim(self, model, short=False):

        """
        Plot the efficient frontier.
        Parameters
        ----------
        model : str
            The model to use for calculating the expected returns. Default is "mean".
        short : bool, optional
            Whether to allow shorting of stocks. Default is False.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Plotly figure object.

        """

        data , weights = self.__simulate_frontier(model, short=short)
        p_risk = data["Risk"]
        p_expected_returns = data["Returns"]
        store_weight = weights

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.array(p_risk), y=p_expected_returns, mode='markers', name="Portfolio", customdata=store_weight))

        fig.update_layout(
            title="Efficient frontier",
            xaxis_title="Standard Deviation",
            yaxis_title="Expected Returns",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            ), 
            
        )

        template = "Standard deviation: %{x:.4f}%<br>Expected return: %{y:.4f}%"
        for i, col in enumerate(self.portfolio.portfolio_stocks()):
            template += f"<br>{col} weight: %{{customdata[{i}]:.4f}}"

        fig.update_traces(
            hovertemplate=template
        )

        fig.show()


    def __pre_process(self, data):

        """
        Pre-process the data.
        Parameters
        ----------
        data : pandas.DataFrame
            Dataframe containing the simulated portfolios.

        Returns
        -------
        data_new : pandas.DataFrame
            Dataframe containing the processed portfolios.
        index : array_like
            Index of the processed portfolios.

        """

        data["Returns2"] = data["Returns"].apply(lambda x: np.round(x, 3))
        returns_unique = data["Returns2"].unique()
        min_risk = []
        index = []
        for i in returns_unique:
            min_risk.append(data[data["Returns2"] == i]["Risk"].min())
            index.append(data[data["Returns2"] == i]["Risk"].idxmin())
        min_risk = np.array(min_risk)
        index = np.array(index)
        data_new = data.iloc[index]

        data_new.sort_values(by="Returns", inplace=True)
        data_new["Returns_Change"] = data_new["Returns"].diff(1).fillna(0)
        # data_final = self.__remove_outliers(data_new, "Returns_Change")
        return data_new, index


    def __fit(self, X, Y, order=3):

        """
        Fit the data.
        Parameters
        ----------
        X : array_like
            X values.
        Y : array_like
            Y values.
        order : int, optional
            Order of the polynomial. Default is 3.

        Returns
        -------
        cf : CurveFitting
            CurveFitting object.

        """
        cf = CurveFitting()
        _ = cf.fit(X=X, Y=Y, order=order)
        Y_pred = cf.predict(X)

        return Y_pred


    def plot_frontier(self, short=False, model="capm"):
        """
        Creates a plot of the efficient frontier.

        Parameters
        ----------
        short : bool, optional
            Whether to allow shorting, by default False
        model : str, optional
            The model to use, by default "capm". Supported models are 'capm', 'sim'
            (If you want to use 'fff3' or 'fff5', first load the fff parameters.)

        Returns
        -------
        fig : plotly.graph_objects.Figure

        Examples
        --------
        >>> ef = EfficientFrontier(portfolio)
        >>> ef.plot_frontier()
        Notes
        -----
        
        You have to create a Portoflio object first. Then you need to load data. Only then you can call `plot_frontier()`.
        """



        data , weights = self.__simulate_frontier(model=model, short=short)
        data_final, _ = self.__pre_process(data=data)

        Y = data_final["Risk"]
        X = data_final["Returns"]

        Y_pred = self.__fit(X, Y, 3)

        data_final["Final_Returns"] = X
        data_final["Final_Risk"] = Y_pred
        weights_final = weights[data_final.index]
        customdata = weights_final.T * 100
        sharpe_ratios = data_final["Sharpe ratio"].astype(float).values

        tempelate = "Risk Deviation: %{x:.4f}<br>Expected Returns: %{y:.4f}"
        for i, name in enumerate(self.portfolio.portfolio_stocks()):
            tempelate += f"<br>{name}: %{{customdata{[i]}:.4f}}"

        fig = px.scatter(
            x=data_final["Final_Risk"],
            y=data_final["Final_Returns"],
            labels={"x": "Standard Deviation (Risk)", "y": "Expected Returns"},
            custom_data=customdata,
            color=sharpe_ratios,
            color_continuous_scale="Viridis",
        )
        fig.update_traces(
            hovertemplate=tempelate,
            marker=dict(
                size=5,
                color=sharpe_ratios,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sharpe ratio"),
            ),
        )

        
        
        # Update the color bar name
        fig.update_coloraxes(colorbar_title_text="Sharpe ratio")

        fig.show()
        return fig

