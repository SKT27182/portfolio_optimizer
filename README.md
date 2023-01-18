# portfolio_optimizer

[![PyPI version](https://badge.fury.io/py/portfolio-optimizer0.svg)](https://badge.fury.io/py/portfolio-optimizer0)
[![Downloads](https://static.pepy.tech/badge/portfolio-optimizer0)](https://pepy.tech/project/portfolio-optimizer0)



A small python library for analysis of historical data of stocks. Especially for portfolio management and optimization.

## Installation

```bash
pip install portfolio-optimizer0
```

> Note: The library is still in development, so the version number is 0. You will need to call `pip install portfolio-optimizer0` to install the library. However, you can import the library as `import portfolio_optimizer`.

After installation, you can import the library as follows:

```python
import portfolio_optimizer as po
```

## Usage

The end goal of the library is to provide a simple interface for portfolio management and optimization. The library is still in development, so the interface is not yet stable. The following example shows how to use the library to optimize a portfolio of stocks.

```python
from po.load_data import LoadData   # for loading data
from po.portfolio import Portfolio  # for portfolio management
from po.optimizer import Optimizer  # for optimization purpose
from po.frontier import EfficientFrontier  # for the plot of Efficient Frontier

# loading the data from NSE exchange you can use BSE also
ld = LoadData()
data_dic = ld.load_data(['HDFCBANK', 'RELIANCE', 'TCS', 'INFY', 'BAJAJ-AUTO', 'TATAMOTORS'], 'NSE')

# loading the benchmark separately, Indices are always downloaded from nasdaq
benchmark = ld.load_data(['^NSEI'], 'nasdaq')

# Creating a Portfolio object, with the freq="M" and benchmark as NSEI (Nifty 50)
portfolio = Portfolio(freq="M", benchmark=benchmark)


# Adding the stocks to the portfolio
portfolio.add_stock(data_dic)

# You can view you Portfolio summary also by
portfolio.portfolio_summary()

# Creating a Optimizer object and adding the portfolio
model = Optimizer()
model.add_portfolio(portfolio=portfolio)

# Optimizing the portfolio using CAPM
risk = 1
model_ = "capm"
optimized_res = model.optimize_portfolio(model=model, max_risk=risk)
print(optimized_res)
```

```output
Optimized successfully.
Expected Portfolio's Returns : 1.3079
Risk : 1.0000
Expected weights:
--------------------
['HDFCBANK.NS']: 22.85%
['RELIANCE.NS']: 27.83%
['TCS.NS']: 0.00%
['INFY.NS']: 0.00%
['BAJAJ-AUTO.NS']: 0.00%
['TATAMOTORS.NS']: 49.32%
```

## More Examples

For more detailed go through of the library, please refer to the notebook [Walk through portfolio_optimizer](https://github.com/shailjakant-3245/portfolio_optimizer/blob/main/go_through_portfolio_optimizer.ipynb)

## Documentation

The documentation is available at [Documentation](https://shailjakant-3245.github.io/portfolio_optimizer/)
