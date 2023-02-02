# Library to load data  from data folder or yfinance

from portfolio_optimizer.style import cprint
import yfinance as yf
import pandas as pd
import tqdm

class LoadData:


    """
    Load the data from the tickers
    
    Load Index data from nasdaq only, For Indian Index price is in INR (mostly)

    Few Important Tickers are :

    # Indices
    | ^GSPC | S&P500 | USD |
    | ^NSEI | Nifty 50 | INR |
    | ^NSEBANK | NiftyBank | INR |
    | ^BSESN | Sensex | INR |

    # Crypto
    | BTC-USD | Bitcoin | USD |
    | ETH-USD | Ethereum | USD |
    | LTC-USD | Litecoin | USD |
    | DOGE-USD | Dogecoin | USD |

    # Commodities
    | GC=F | Gold | USD |
    | SI=F | Silver | USD |
    | CL=F | Crude Oil | USD |
    | HG=F | Copper | USD |
    | ZC=F | Corn | USD |

    # US treasury
    | ^IRX | 13 Week Treasury Bill | USD |
    | ^FVX | 5 Year Treasury Note | USD |
    | ^TNX | 10 Year Treasury Note | USD |
    | ^TYX | 30 Year Treasury Bond | USD |

    # Indian NSE (National Stock Exchange)
    | RELIANCE.NS | Reliance Industries | INR |
    | TCS.NS | Tata Consultancy Services | INR | 
    | HDFCBANK.NS | HDFC Bank | INR |

    # Indian BSE (Bombay Stock Exchange)
    | RELIANCE.BO | Reliance Industries | INR |
    | TCS.BO | Tata Consultancy Services | INR |
    | HDFCBANK.BO | HDFC Bank | INR |
    """



    def __init__(self) -> None:
        pass

    def get_ticker_names(self, exchange):
        """
        Parameters
        ----------
        exchange : str
            Exchange to get the ticker name from

        Returns
        -------
        DataFrame:
            Return the DataFrame of the Ticker and Company name listed on the specified exchanged
        """

        url_nse = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        url_bse = "https://www.bseindia.com/corporates/List_Scrips.html"
        url_nasdaq = "https://www.nasdaq.com/market-activity/stocks/screener"

        if exchange.lower() == "nse":
            return pd.read_csv(url_nse)[["SYMBOL","NAME OF COMPANY"]]
        elif exchange.lower() == "bse":
            cprint.print("BSE not supported yet. You can use NSE instead or check the companies listed there for the returned url", "red")
            return url_bse
        elif exchange.lower() == "nasdaq":
            cprint.print("Nasdaq not supported yet. You can use NSE instead or check the companies listed there for the returned url", "red")
            return url_nasdaq


    def get_company_name(self, ticker, exchange):
        """
        Get the company name from the ticker

        Parameters
        ----------
        ticker : str
            Ticker of the company

        exchange : str
            Exchange to get the ticker name from

        Returns
        -------
        str
            Return the company name
        """

        if exchange.lower() == "nse":
            try:
                cprint.print(self.get_ticker_names(exchange).loc[self.get_ticker_names(exchange)["SYMBOL"]==ticker]["NAME OF COMPANY"].values[0], "blue")
            except:
                cprint.print("Not Found", "fail")
        elif exchange.lower() == "bse":
            try:
                cprint.print(yf.Ticker(ticker).info['longName'], "blue")
            except TypeError:
                cprint.print("Not Found", "fail")
        elif exchange.lower() == "nasdaq":
            try:
                cprint.print(yf.Ticker(ticker).info['longName'], "blue")
            except TypeError:
                cprint.print("Not Found", "fail")
                
        

    def load_data(self, tickers, exchange="nse"):
        """
        Load the data from the tickers

        Load Index data from nasdaq only, For Indian Index price is in INR (mostly)

        Parameters
        ----------
        tickers : list
            List of the tickers to load the data from
            
        exchange : str
            Exchange to get the ticker name from

        Returns
        -------
        dict
            Return the dictionary of the dataframes of the tickers

        if exchange = "nse" or "bse":
            Returns the historical data in INR
        if exchange = "nasdaq":
            Returns the historical data in USD

        by default exchange = "nse"
        indices of nse are in INR and are available in nasdaq only

        """
        data = {}
        if exchange.lower() == "nse":
            tickers = [_+".NS" for _ in tickers]
        elif exchange.lower() == "bse":
            tickers = [_+".BO" for _ in tickers]
        elif exchange.lower() == "nasdaq":
            tickers = tickers

        for ticker in tickers:
            cprint.print(f"Downloading...{ticker}", "green")
            data_ = yf.download(ticker, period="max")
            if data_.shape[0] !=0:
                data_ = data_.dropna()
                data.update({ticker:data_})
            else:
                cprint.print(f"{ticker} Ticker Not Found.", "fail")
            
        return data

    def save_data(self, data, path):
        """
        Save the data to the path

        Parameters
        ----------
        data : dict
            Dictionary of the dataframes of the tickers

        path : str
            Path to save the data

        Returns
        -------
        None
        """
        for ticker in data.keys():
            data[ticker].to_csv(f"{path}/{ticker}.csv")
            cprint.print(f"{ticker} Saved.", "blue")
        