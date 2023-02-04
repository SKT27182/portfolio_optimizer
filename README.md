# portfolio_optimizer

[![PyPI version](https://badge.fury.io/py/portfolio-optimizer0.svg)](https://badge.fury.io/py/portfolio-optimizer0)
[![Downloads](https://static.pepy.tech/personalized-badge/portfolio-optimizer0?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/portfolio-optimizer0)
<!-- [![Downloads](https://static.pepy.tech/badge/portfolio-optimizer0)](https://pepy.tech/project/portfolio-optimizer0) -->




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
data_dic = ld.load_data(['HDFCBANK', 'RELIANCE', 'INFY', 'BAJAJ-AUTO', 'TATAMOTORS'], 'NSE')

# loading the benchmark separately, Indices are always downloaded from nasdaq
benchmark = ld.load_data(['^NSEI'], 'nasdaq')

# Creating a Portfolio object, with the freq="M" and benchmark as NSEI (Nifty 50)
portfolio = Portfolio(freq="M", benchmark=benchmark)


# Adding the stocks to the portfolio
portfolio.add_stock(data_dic)

# You can view you Portfolio summary also by
portfolio.portfolio_summary()
```
    
```output
Portfolio Summary
*****************

Stocks in the Portfolio : *************************
{"['HDFCBANK', 'RELIANCE', 'INFY', 'BAJAJ-AUTO', 'TATAMOTORS']"}

Beta :
******
|      |   HDFCBANK |   RELIANCE |     INFY |   BAJAJ-AUTO |   TATAMOTORS |
|------+------------+------------+----------+--------------+--------------|
| beta |    1.00174 |    1.04458 | 0.543611 |      1.01237 |      1.59273 |

Expected Returns :
******************
|    |   HDFCBANK |   RELIANCE |     INFY |   BAJAJ-AUTO |   TATAMOTORS |
|----+------------+------------+----------+--------------+--------------|
|  0 |    1.05617 |    1.09172 | 0.676051 |        1.065 |      1.54654 |

The covariance matrix is as follows
***********************************
|            |   HDFCBANK |   RELIANCE |       INFY |   BAJAJ-AUTO |   TATAMOTORS |
|------------+------------+------------+------------+--------------+--------------|
| HDFCBANK   | 0.00565169 | 0.00383894 | 0.00144691 |   0.00458971 |   0.00610401 |
| RELIANCE   | 0.00383894 | 0.00795272 | 0.00227559 |   0.00466296 |   0.0060295  |
| INFY       | 0.00144691 | 0.00227559 | 0.00679807 |   0.00154204 |   0.00296804 |
| BAJAJ-AUTO | 0.00458971 | 0.00466296 | 0.00154204 |   0.0110113  |   0.00849118 |
| TATAMOTORS | 0.00610401 | 0.0060295  | 0.00296804 |   0.00849118 |   0.0228973  |

Portfolio Returns at equals weights: 1.0870947316063777
Portfolio Risk at equals weights: 0.5528354886819963
```

```python
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
Expected Portfolio's Returns : 1.1792
Risk : 0.8001
Expected weights:
--------------------
['HDFCBANK']: 47.44%
['RELIANCE']: 29.62%
['INFY']: 0.00%
['BAJAJ-AUTO']: 0.00%
['TATAMOTORS']: 22.94%
```

## More Examples

For more detailed go through of the library, please refer to the notebook [Walk through portfolio_optimizer](https://github.com/SKT27182/portfolio_optimizer/blob/main/go_through_portfolio_optimizer.ipynb)

## Documentation

The documentation is available at [Documentation](https://SKT27182.github.io/portfolio_optimizer/)
