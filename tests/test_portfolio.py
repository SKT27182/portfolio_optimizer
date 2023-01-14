#Pytest for Portfolio class

import pytest
import pandas as pd
import numpy as np
import sys 
from pandas.testing import assert_frame_equal
sys.path.append("/home/shailja/Courses/Notes/portfolio_optimizer")

from portfolio_optimizer.portfolio import Portfolio


market_ = "Nifty"
df_ = pd.read_csv("data/^NSEI.csv", parse_dates=["Date"], index_col="Date")
benchmark_ = {market_: df_}
freq_ = "M"

data_dic_ = {"Nifty": pd.read_csv("data/^NSEI.csv", parse_dates=["Date"], index_col="Date"), "TCS": pd.read_csv("data/TCS.csv", parse_dates=["Date"], index_col="Date"), "INFY": pd.read_csv("data/INFY.csv", parse_dates=["Date"], index_col="Date")}

@pytest.fixture
def portfolio_base():
    return Portfolio("M", benchmark = {"Nifty": pd.read_csv("../benchmark/^NSEI.csv")})

@pytest.fixture
def portfolio_full():
    port = Portfolio("M", benchmark = {"Nifty": pd.read_csv("../benchmark/^NSEI.csv")})
    port.add_stock({"TCS": pd.read_csv("data/TCS.csv"), "INFY": pd.read_csv("data/INFY.csv")})
    return port


def test_portfolio_baseinit(portfolio_base):
    
    # Assert that the object is created correctly
    assert isinstance(portfolio_base, Portfolio)
    # Assert that the attributes are set correctly
    assert portfolio_base.market == market_
    assert_frame_equal(portfolio_base.stocks[portfolio_base.market], benchmark_[market_])
    assert set(portfolio_base.stocks.keys()) == set(benchmark_.keys())
    assert portfolio_base.risk_free_rate == 0.225
    assert portfolio_base.market_return == benchmark_[market_].asfreq(freq_, method="ffill").pct_change().dropna()["Close"].mean()*100
    assert portfolio_base.merged_stocks.equals(benchmark_[market_])
    assert portfolio_base.betas is None
    assert portfolio_base.alphas is None
    assert portfolio_base.returns is None

# test base len function
def test_portfolio_baselen(portfolio_base):
    assert len(portfolio_base) == len(benchmark_.keys())-1

# test add_stock function
def test_portfolio_base_add_stock(portfolio_base):
    benchmark = benchmark_.copy()
    benchmark["TCS"] = pd.read_csv("data/TCS.csv")
    portfolio_base.add_stock({"TCS": pd.read_csv("data/TCS.csv")})
    assert portfolio_base.market == market_
    assert_frame_equal(portfolio_base.stocks[portfolio_base.market], benchmark[market_])
    assert set(portfolio_base.stocks.keys()) == set(benchmark.keys())
    assert portfolio_base.risk_free_rate == 0.225
    assert portfolio_base.market_return == benchmark[market_].asfreq(freq_, method="ffill").pct_change().dropna()["Close"].mean()*100
    assert portfolio_base.betas is not None
    assert portfolio_base.alphas is not None
    assert portfolio_base.returns is not None

# test __setitem__ function
def test_portfolio_base_setitem(portfolio_base):
    benchmark = benchmark_.copy()
    benchmark["TCS"] = pd.read_csv("data/TCS.csv")
    portfolio_base["TCS"] = pd.read_csv("data/TCS.csv")
    assert portfolio_base.market == market_
    assert_frame_equal(portfolio_base.stocks[portfolio_base.market], benchmark[market_])
    assert set(portfolio_base.stocks.keys()) == set(benchmark.keys())
    assert portfolio_base.risk_free_rate == 0.225
    assert portfolio_base.market_return == benchmark[market_].asfreq(freq_, method="ffill").pct_change().dropna()["Close"].mean()*100
    assert portfolio_base.betas is not None
    assert portfolio_base.alphas is not None
    assert portfolio_base.returns is not None

# test __getitem__ function
def test_portfolio_base_getitem(portfolio_base):
    assert portfolio_base["Nifty"].equals(benchmark_[market_])

# test __delitem__ function
def test_portfolio_full_delitem(portfolio_full):
    del portfolio_full["TCS"]
    data_dic = data_dic_.copy()
    del data_dic["TCS"]
    assert portfolio_full.market == market_
    assert len(portfolio_full) == len(data_dic.keys())-1
    # assert_frame_equal(portfolio_full.stocks[portfolio_full.market], data_dic[market_]), as data_dic isn't merged
    assert set(portfolio_full.stocks.keys()) == set(data_dic.keys())
    assert portfolio_full.risk_free_rate == 0.225
    assert portfolio_full.market_return == data_dic[market_].asfreq(freq_, method="ffill").pct_change().dropna()["Close"].mean()*100
    assert portfolio_full.betas is not None
    assert portfolio_full.alphas is not None
    assert portfolio_full.returns is not None

# test remove_stock function
def test_portfolio_full_remove_stock(portfolio_full):
    portfolio_full.remove_stock("TCS")
    data_dic = data_dic_.copy()
    del data_dic["TCS"]
    assert portfolio_full.market == market_
    assert len(portfolio_full) == len(data_dic.keys())-1
    # assert_frame_equal(portfolio_full.stocks[portfolio_full.market], data_dic[market_]), as data_dic isn't merged
    assert set(portfolio_full.stocks.keys()) == set(data_dic.keys())
    assert portfolio_full.risk_free_rate == 0.225
    assert portfolio_full.market_return == data_dic[market_].asfreq(freq_, method="ffill").pct_change().dropna()["Close"].mean()*100
    assert portfolio_full.betas is not None
    assert portfolio_full.alphas is not None
    assert portfolio_full.returns is not None

# test cal_beta function
def test_portfolio_full_cal_beta(portfolio_full):
    portfolio_full.cal_beta()
    assert portfolio_full.betas is not None

# test cal_alpha function
def test_portfolio_full_cal_alpha(portfolio_full):
    portfolio_full.cal_alpha()
    assert portfolio_full.alphas is not None

# test cal_return function
def test_portfolio_full_cal_return(portfolio_full):
    portfolio_full.cal_returns()
    assert portfolio_full.returns is not None

# test cov_matrix function
def test_portfolio_full_cov_matrix(portfolio_full):
    cov_matrix = portfolio_full.cov_matrix()
    assert isinstance(cov_matrix, pd.DataFrame)
    assert cov_matrix.shape == (len(portfolio_full)+1, len(portfolio_full)+1) # added 1 b/c here we haven't removed benchmark in cov_matrix
    assert cov_matrix.index.equals(portfolio_full.returns.columns)
    assert cov_matrix.columns.equals(portfolio_full.returns.columns)

# test expected_returns function
def test_portfolio_full_expected_returns(portfolio_full):
    expected_return = portfolio_full.expected_returns()
    assert isinstance(expected_return, np.ndarray)
    assert expected_return.shape == ((len(portfolio_full)),)  # made a tuple b/c LHS is a tuple
    
# test portfolio_expected_return function
def test_portfolio_full_portfolio_expected_return(portfolio_full):
    portfolio_expected_return = portfolio_full.portfolio_expected_return()
    assert isinstance(portfolio_expected_return, np.float64)

# test portfolio_variance function
def test_portfolio_full_portfolio_variance(portfolio_full):
    portfolio_variance = portfolio_full.portfolio_variance()
    assert isinstance(portfolio_variance, np.float64)

# test portfolio_volatility function
def test_portfolio_full_portfolio_volatility(portfolio_full):
    portfolio_std = portfolio_full.portfolio_std()
    assert isinstance(portfolio_std, np.float64)