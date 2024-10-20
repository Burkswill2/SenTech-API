'''
Portfolio Optimization using Modern Portfolio Theory

This script performs portfolio optimization based on the Modern Portfolio Theory (MPT).
It downloads historical stock data, calculates returns, generates random portfolios,
and finds the optimal portfolio composition that maximizes the Sharpe ratio.

Key Features:
1. Downloads historical stock data using yfinance
2. Calculates logarithmic daily returns
3. Generates random portfolios
4. Finds the optimal portfolio using the Sharpe ratio
5. Visualizes the efficient frontier and the optimal portfolio

Important Concepts:
1. Modern Portfolio Theory: A framework for constructing optimal investment portfolios
2. Sharpe Ratio: A measure of risk-adjusted return
3. Efficient Frontier: The set of optimal portfolios that offer the highest expected return for a defined level of risk

Note: This script uses a sample set of stocks and a fixed date range. Adjust these parameters as needed.
'''

import numpy as np
import yfinance as yahoo
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from scipy.optimize import OptimizeResult
from typing import Dict, Any

# Constants
NUM_TRADING_DAYS = 252  # Average number of trading days in a year
NUM_PORTFOLIOS = 10000  # Number of random portfolios to generate

# List of stocks to analyze
stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']

# Date range for historical data
start_date = '2014-01-01'
end_date = '2024-09-01'

def download_data():
    '''
    Downloads historical stock data for the specified stocks and date range.

    Returns:
        pd.DataFrame: A DataFrame containing the closing prices for each stock.
    '''
    stock_data = {}
    for stock in stocks:
        ticker = yahoo.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']
    return pd.DataFrame(stock_data)

def show_data(data):
    '''
    Plots the historical stock prices.

    Args:
        data (pd.DataFrame): DataFrame containing stock prices.
    '''
    data.plot(figsize=(10,6))
    plt.show()

def calculate_return(data):
    '''
    Calculates logarithmic daily returns.

    Args:
        data (pd.DataFrame): DataFrame containing stock prices.

    Returns:
        pd.DataFrame: DataFrame containing logarithmic daily returns.
    '''
    log_return = np.log(data / data.shift(1))
    return log_return[1:]

def show_statistics(returns):
    '''
    Prints annualized mean returns and covariance matrix.

    Args:
        returns (pd.DataFrame): DataFrame containing daily returns.
    '''
    print(returns.mean() * NUM_TRADING_DAYS)
    print(returns.cov() * NUM_TRADING_DAYS)

def show_mean_and_variance(returns, weights):
    '''
    Calculates and prints the expected return and volatility for a given portfolio.

    Args:
        returns (pd.DataFrame): DataFrame containing daily returns.
        weights (np.array): Array of portfolio weights.
    '''
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
    print("Expected portfolio mean (return): ", portfolio_return)
    print("Expected portfolio volatility (standard deviation): ", portfolio_volatility)

def show_portfolios(returns, volatilities):
    '''
    Plots the efficient frontier.

    Args:
        returns (np.array): Array of portfolio returns.
        volatilities (np.array): Array of portfolio volatilities.
    '''
    plt.figure(figsize=(10,6))
    plt.scatter(volatilities, returns, c=returns / volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

def generate_portfolios(returns):
    '''
    Generates random portfolios and calculates their returns and risks.

    Args:
        returns (pd.DataFrame): DataFrame containing daily returns.

    Returns:
        tuple: Arrays of portfolio means, risks, and weights.
    '''
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUM_TRADING_DAYS, w))))

    return np.array(portfolio_means), np.array(portfolio_risks), np.array(portfolio_weights)

def statistics(weights, returns):
    '''
    Calculates portfolio statistics: return, volatility, and Sharpe ratio.

    Args:
        weights (np.array): Array of portfolio weights.
        returns (pd.DataFrame): DataFrame containing daily returns.

    Returns:
        np.array: Array containing portfolio return, volatility, and Sharpe ratio.
    '''
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
    return np.array([portfolio_return, portfolio_volatility, portfolio_return/portfolio_volatility])

def min_function_sharpe(weights, returns):
    '''
    Function to minimize (negative Sharpe ratio) for optimization.

    Args:
        weights (np.array): Array of portfolio weights.
        returns (pd.DataFrame): DataFrame containing daily returns.

    Returns:
        float: Negative Sharpe ratio.
    '''
    return -statistics(weights, returns)[2]

def optimize_portfolio(weights, returns):
    '''
    Optimizes the portfolio to maximize the Sharpe ratio.

    Args:
        weights (np.array): Initial guess for portfolio weights.
        returns (pd.DataFrame): DataFrame containing daily returns.

    Returns:
        OptimizeResult: Result of the optimization.
    '''
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0,1) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_function_sharpe,
                                 x0=weights[0],
                                 args=returns,
                                 method='SLSQP',
                                 bounds=bounds,
                                 constraints=constraints)

def optimal_portfolio(optimum):
    '''
    Creates a dictionary of optimal portfolio weights.

    Args:
        optimum (OptimizeResult): Result of portfolio optimization.

    Returns:
        dict: Dictionary of stocks and their optimal weights.
    '''
    return dict(zip(stocks, optimum['x'].round(3).tolist()))

def jsonify_optimal_portfolio(optimum: OptimizeResult, returns: pd.DataFrame) -> Dict[str, Any]:
    """
    Creates a JSON-serializable dictionary with the details of the optimal portfolio.

    Args:
        optimum (OptimizeResult): Result of portfolio optimization.
        returns (pd.DataFrame): DataFrame containing daily returns.

    Returns:
        Dict[str, Any]: A dictionary containing the optimal portfolio details.
    """
    stats = statistics(optimum['x'].round(3), returns)
    optimal_port = optimal_portfolio(optimum)

    result = {
        "optimal_portfolio": {
            asset: weight for asset, weight in optimal_port.items()
        },
        "expected_return": round(stats[0], 4),
        "expected_volatility": round(stats[1], 4),
        "expected_sharpe_ratio": round(stats[2], 4)
    }

    return result

def show_optimal_portfolios(optimum, returns, portfolio_returns, portfolio_volatilities):
    '''
    Plots the efficient frontier and highlights the optimal portfolio.

    Args:
        optimum (OptimizeResult): Result of portfolio optimization.
        returns (pd.DataFrame): DataFrame containing daily returns.
        portfolio_returns (np.array): Array of portfolio returns.
        portfolio_volatilities (np.array): Array of portfolio volatilities.
    '''
    plt.figure(figsize=(10,6))
    plt.scatter(portfolio_volatilities, portfolio_returns, c=portfolio_returns / portfolio_volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(optimum['x'], returns)[1], statistics(optimum['x'], returns)[0], 'g*', markersize=20.0)
    plt.show()

def start_model():
    # Download and process data
    dataset = download_data()
    # show_data(dataset)
    log_daily_returns = calculate_return(dataset)

    # Generate random portfolios
    means, risks, pweights = generate_portfolios(log_daily_returns)

    # Find the optimal portfolio
    optimum = optimize_portfolio(pweights, log_daily_returns)
    result = jsonify_optimal_portfolio(optimum, log_daily_returns)

    return result

    # Visualize results
    # show_optimal_portfolios(optimum, log_daily_returns, means, risks)