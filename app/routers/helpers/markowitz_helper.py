import numpy as np
import yfinance as yahoo
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from scipy.optimize import OptimizeResult
from typing import Dict, Any, List, Tuple
import plotly.graph_objects as go


class PortfolioOptimizer:
    NUM_TRADING_DAYS = 252  # Average number of trading days in a year
    NUM_PORTFOLIOS = 10000  # Number of random portfolios to generate

    def __init__(self, stocks: List[str], start_date: str, end_date: str):
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.dataset = self.download_data()
        self.log_daily_returns = self.calculate_return()

    def download_data(self) -> pd.DataFrame:
        stock_data = {}
        for stock in self.stocks:
            ticker = yahoo.Ticker(stock)
            stock_data[stock] = ticker.history(start=self.start_date, end=self.end_date)['Close']
        return pd.DataFrame(stock_data)

    def show_data(self):
        self.dataset.plot(figsize=(10,6))
        plt.show()

    def calculate_return(self) -> pd.DataFrame:
        log_return = np.log(self.dataset / self.dataset.shift(1))
        return log_return[1:]

    def show_statistics(self):
        print(self.log_daily_returns.mean() * self.NUM_TRADING_DAYS)
        print(self.log_daily_returns.cov() * self.NUM_TRADING_DAYS)

    @staticmethod
    def show_mean_and_variance(returns: pd.DataFrame, weights: np.array):
        portfolio_return = np.sum(returns.mean() * weights) * PortfolioOptimizer.NUM_TRADING_DAYS
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * PortfolioOptimizer.NUM_TRADING_DAYS, weights)))
        print("Expected portfolio mean (return): ", portfolio_return)
        print("Expected portfolio volatility (standard deviation): ", portfolio_volatility)

    @staticmethod
    def show_portfolios(returns: np.array, volatilities: np.array):
        plt.figure(figsize=(10,6))
        plt.scatter(volatilities, returns, c=returns / volatilities, marker='o')
        plt.grid(True)
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.colorbar(label='Sharpe Ratio')
        plt.show()

    def generate_portfolios(self) -> Tuple[np.array, np.array, np.array]:
        portfolio_means = []
        portfolio_risks = []
        portfolio_weights = []

        for _ in range(self.NUM_PORTFOLIOS):
            w = np.random.random(len(self.stocks))
            w /= np.sum(w)
            portfolio_weights.append(w)
            portfolio_means.append(np.sum(self.log_daily_returns.mean() * w) * self.NUM_TRADING_DAYS)
            portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(self.log_daily_returns.cov() * self.NUM_TRADING_DAYS, w))))

        return np.array(portfolio_means), np.array(portfolio_risks), np.array(portfolio_weights)

    @staticmethod
    def statistics(weights: np.array, returns: pd.DataFrame) -> np.array:
        portfolio_return = np.sum(returns.mean() * weights) * PortfolioOptimizer.NUM_TRADING_DAYS
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * PortfolioOptimizer.NUM_TRADING_DAYS, weights)))
        return np.array([portfolio_return, portfolio_volatility, portfolio_return/portfolio_volatility])

    @staticmethod
    def min_function_sharpe(weights: np.array, returns: pd.DataFrame) -> float:
        return -PortfolioOptimizer.statistics(weights, returns)[2]

    def optimize_portfolio(self, weights: np.array) -> OptimizeResult:
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0,1) for _ in range(len(self.stocks)))
        return optimization.minimize(fun=self.min_function_sharpe,
                                     x0=weights[0],
                                     args=self.log_daily_returns,
                                     method='SLSQP',
                                     bounds=bounds,
                                     constraints=constraints)

    def optimal_portfolio(self, optimum: OptimizeResult) -> Dict[str, float]:
        return dict(zip(self.stocks, optimum['x'].round(3).tolist()))

    def jsonify_optimal_portfolio(self, optimum: OptimizeResult, means, risks, plot) -> Dict[str, Any]:
        stats = self.statistics(optimum['x'].round(3), self.log_daily_returns)
        optimal_port = self.optimal_portfolio(optimum)

        # Calculate percent change over the last trading day
        if len(self.dataset) >= 2:
            percent_change_1d = ((self.dataset.iloc[-1] - self.dataset.iloc[-2]) / self.dataset.iloc[-2]) * 100
        else:
            percent_change_1d = {stock: None for stock in self.stocks}  # Handle insufficient data

        print("Last two closing prices:\n", self.dataset.tail(2))  # Diagnostic print

        # Fetch additional company information
        company_info = {}
        for stock in self.stocks:
            ticker = yahoo.Ticker(stock)
            info = ticker.info
            company_info[stock] = {
                "sector": info.get("sector"),
                "website": info.get("website"),
                "longName": info.get("longName"),
                "shortName": info.get("shortName"),
                "previousClose": info.get("previousClose"),
                "percent_change_1d": percent_change_1d[stock] if isinstance(percent_change_1d, dict) else None
            }

        # Create the portfolio structure with tickers as keys
        portfolio = {
            stock: {
                "weight": round(optimal_port[stock], 3),
                "info": company_info[stock]  # Add corresponding company info
            }
            for stock in self.stocks
        }

        print(self.log_daily_returns)

        result = {
            "result": {
                "portfolio": portfolio,
                "expected_return": round(stats[0], 3),
                "expected_volatility": round(stats[1], 4),
                "expected_sharpe_ratio": round(stats[2], 4),
                "optimum": optimum['x'].tolist(),  # Convert the NumPy array to a list
                "returns": self.log_daily_returns.reset_index().to_dict(orient='records'),  # Convert to list of dicts
                "means": means.tolist(),  # Convert NumPy array to list
                "risks": risks.tolist(),  # Convert NumPy array to list
                "plot" : plot,
            }
        }

        return result

    # def show_optimal_portfolios(self, optimum: OptimizeResult, portfolio_returns: np.array, portfolio_volatilities: np.array):
    #     plt.figure(figsize=(10,6))
    #     plt.scatter(portfolio_volatilities, portfolio_returns, c=portfolio_returns / portfolio_volatilities, marker='o')
    #     plt.grid(True)
    #     plt.xlabel('Expected Volatility')
    #     plt.ylabel('Expected Return')
    #     plt.colorbar(label='Sharpe Ratio')
    #     plt.plot(self.statistics(optimum['x'], self.log_daily_returns)[1], self.statistics(optimum['x'], self.log_daily_returns)[0], 'g*', markersize=20.0)
    #     plt.show()

    def show_optimal_portfolios(self, optimum, returns, portfolio_returns, portfolio_volatilities):
        '''
        Plots the efficient frontier and highlights the optimal portfolio as an interactive JSON object.

        Args:
            optimum (OptimizeResult): Result of portfolio optimization.
            returns (pd.DataFrame): DataFrame containing daily returns.
            portfolio_returns (np.array): Array of portfolio returns.
            portfolio_volatilities (np.array): Array of portfolio volatilities.
        '''
        # Calculate Sharpe Ratios for the portfolios
        sharpe_ratios = portfolio_returns / portfolio_volatilities

        # Create a scatter plot
        fig = go.Figure()

        # Add scatter points for portfolios
        fig.add_trace(go.Scatter(
            x=portfolio_volatilities,
            y=portfolio_returns,
            mode='markers',
            marker=dict(
                color=sharpe_ratios,
                colorscale='Viridis',
                colorbar=dict(title='Sharpe Ratio'),
                size=10,
                opacity=0.7
            ),
            text=[f'Sharpe: {sharpe:.2f}' for sharpe in sharpe_ratios],
            hoverinfo='text'
        ))

        # Highlight the optimal portfolio
        optimal_stats = self.statistics(optimum['x'], returns)  # Ensure statistics function is defined
        fig.add_trace(go.Scatter(
            x=[optimal_stats[1]],
            y=[optimal_stats[0]],
            mode='markers',
            marker=dict(color='green', size=20, symbol='star'),
            name='Optimal Portfolio'
        ))

        # Update layout
        fig.update_layout(
            title='Efficient Frontier with Optimal Portfolio',
            xaxis_title='Expected Volatility',
            yaxis_title='Expected Return',
            showlegend=True
        )

        # Convert the figure to JSON
        plot_json = fig.to_json()

        return plot_json

def start_model(params) -> Dict[str, Any]:
    optimizer = PortfolioOptimizer(params["stocks"], params["start_date"], params["end_date"])
    means, risks, pweights = optimizer.generate_portfolios()
    optimum = optimizer.optimize_portfolio(pweights)
    log_daily_returns = optimizer.log_daily_returns
    plot = optimizer.    show_optimal_portfolios(optimum, log_daily_returns, means, risks)
    result = optimizer.jsonify_optimal_portfolio(optimum, means, risks, plot)
    return result

# Usage example
if __name__ == "__main__":

    params = {
        "stocks" : ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB'],
        "start_date": "2014-01-01",
        "end_date": "2024-09-01",
    }

    result = start_model(params)
    # print(result)