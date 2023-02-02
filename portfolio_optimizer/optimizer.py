# Optimize the portfolio

import numpy as np
from portfolio_optimizer.portfolio import Portfolio
from scipy.optimize import minimize
from portfolio_optimizer.style import cprint

class Optimizer:

    """
    Optimize the portfolio

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio to optimize

    Attributes
    ----------
    portfolio : Portfolio
        Portfolio to optimize

    Methods
    -------
    model(weights, model)
        Calculate the expected return of the portfolio
    optimize_portfolio(model, risk, short=False)
        Optimize the portfolio
        
    """

    def __init__(self) -> None:
        pass

    def add_portfolio(self, portfolio:Portfolio):
        """
        Add the portfolio to optimize

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio to optimize
        """
        self.portfolio = portfolio


    def model(self,  weights, model):

        """
        CAPM Model

        Parameters
        ----------
        weights : list
            weights of each stocks

        model : str
            Model to use for the the calculation of the expected return


        Returns
        -------
        float
            Return of the expected return of the portfolio
        """
        
        return self.portfolio.portfolio_expected_return(weights=weights, model=model)
        


    def optimize_portfolio(self, model, max_risk, short=False):


        """
        Optimize the portfolio
        

        Parameters
        ----------
        model : str
            Model to use for the optimization
        max_risk : float
            Risk of the portfolio
        short : bool, optional
            Allow shorting, by default False


        Returns
        -------
        Returns the optimized weights of the portfolio and maximized portfolio_returns

        """

        ini_weights = [1/len(self.portfolio)] * len(self.portfolio)

        if short:
            bounds = tuple([(-2, 2)] * len(self.portfolio))
        else:
            bounds = tuple([(0, 1)] * len(self.portfolio))

        def objective_function(weights):
                return -self.model(weights=weights, model=model)

        
        cons = (
            {'type': 'eq', 'fun': lambda x: sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x:max_risk - self.portfolio.portfolio_std(weights=x) }
                )

        res = minimize(objective_function, ini_weights, method='SLSQP', bounds=bounds, constraints=cons, tol=0.0001)

        weights = res["x"]
        var = self.portfolio.portfolio_variance(weights=res.x)
        std = self.portfolio.portfolio_std(weights=res.x)
        portfolio_expected_return = -res.fun

        if res["success"]:
            cprint.print("Optimized successfully.", "green")
        else:
            cprint.print(f"Optimization failed. {res['message']}", "fail")
            cprint.print("Here are the last results:", "fail")
        
        cprint.print(f"Expected Portfolio's Returns : {portfolio_expected_return:.4f}", "green")
        cprint.print(f"Risk : {std:.4f}", "red")

        cprint.print("Expected weights:", "green")
        cprint.print("-" * 20, "green")
        for i, stock in enumerate(self.portfolio.portfolio_stocks()):
            cprint.print(f"{[stock]}: {weights[i]*100:.2f}%", "green")

        

        return {
            "weights": weights,
            "portfolio_expected_return": portfolio_expected_return,
            "portfolio_variance": var,
            "portfolio_std": std,
        }
