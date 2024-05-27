import yfinance as yf
import pandas as pd 
from datetime import datetime, timedelta
import numpy as np 
from scipy.optimize import minimize
from fredapi import Fred
import matplotlib.pyplot as plt 

#Tickers in portfolio
tickers = ['SPY', 'BND', 'GLD', 'QQQ', 'VTI']

end_date = datetime.today()

start_date = end_date - timedelta(5 * 365) # 5 is five years 

print(start_date)


#Download adjusted close price (which include dividends and stock splits)
adj_close_df = pd.DataFrame()

for ticker in tickers:
  data = yf.download(ticker, start= start_date, end= end_date)
  adj_close_df[ticker] = data['Adj Close']

print(adj_close_df)



#Calculate the log return
log_returns = np.log(adj_close_df / adj_close_df.shift(1))

print(log_returns)


#drop missing values 
log_returns = log_returns.dropna()

print(log_returns)


#Calculate the covariance matrix using annualised log returns 
cov_matrix = log_returns.cov()*252

print(cov_matrix)


#Define the portfolio performance metrics 

#Calculate the standard deviation
def standard_deviation(weights, cov_matrix):
  return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  


#Calculate the expected return
def expected_return(weights, log_returns):
    return np.sum(log_returns.mean()*weights)*252

#Calculate the sharpe ratio
def sharpe_ratio(weights, log_returns, risk_free_rate, cov_matrix):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)


#Define the risk free rate
fred = Fred(api_key='ff13d3aca000abd2734f8e45dff1472b')

ten_year_tresury_rate = fred.get_series_latest_release('GS10') / 100

risk_free_rate = ten_year_tresury_rate.iloc[-1]

print(risk_free_rate)

#Defining the function to mimimize the negative sharpe ratio
def negative_sharpe_ratio(weights, log_returns, risk_free_rate, cov_matrix):
    return -sharpe_ratio(weights, log_returns, risk_free_rate, cov_matrix)


#set the constraints and bounds 
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = [(0, 0.5) for _ in range(len(tickers))]


#Inital weights
initial_weights = np.array([1/len(tickers)]*len(tickers))
print(initial_weights)

#Optimize the portfolio
result = minimize(negative_sharpe_ratio, initial_weights, args=(log_returns, risk_free_rate, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = result.x


#Analyse the portfolio

print("optimal_weights:")

for ticker, weight in zip(tickers, optimal_weights):
  print(f"{ticker}: {weight:.4f}")

print()

optimal_portfolio_return = expected_return(optimal_weights, log_returns)
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
optimal_portfolio_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, risk_free_rate, cov_matrix)

print(f"Expected return: {optimal_portfolio_return:.4f}")
print(f"Standard deviation: {optimal_portfolio_volatility:.4f}")
print(f"Sharpe ratio: {optimal_portfolio_sharpe_ratio:.4f}")


#Display the portfolio as a plot


plt.figure(figsize=(10, 6))
plt.bar(tickers, optimal_weights)

plt.xlabel('Assets')
plt.ylabel('Optimal Weights')
plt.title('Optimal Portfolio Weights')

plt.show()


#Incorporate Different Risk Models

def calculate_var(returns, alpha=0.05):
    """
    Calculate the Value at Risk (VaR) at a specified confidence level
    :param returns: array-like, portfolio returns
    :param alpha: float, confidence level
    :return: float, VaR value
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.values.flatten()
    sorted_returns = np.sort(returns)
    index = int(alpha * len(sorted_returns))
    var = -sorted_returns[index]
    return var

def calculate_cvar(returns, alpha=0.05):
    """
    Calculate the Conditional Value at Risk (CVaR) at a specified confidence level
    :param returns: array-like, portfolio returns
    :param alpha: float, confidence level
    :return: float, CVaR value
    """
    if isinstance(returns, pd.DataFrame):
        returns = returns.values.flatten()
    sorted_returns = np.sort(returns)
    index = int(alpha * len(sorted_returns))
    cvar = -sorted_returns[:index].mean()
    return cvar

def negative_cvar(weights, log_returns, alpha=0.05):
    """
    Function to minimize (negative CVaR)
    :param weights: array-like, portfolio weights
    :param log_returns: DataFrame, log returns of assets
    :param alpha: float, confidence level for CVaR
    :return: float, negative CVaR of the portfolio
    """
    portfolio_returns = log_returns.dot(weights)
    return -calculate_cvar(portfolio_returns, alpha)

# Optimization settings
result_cvar = minimize(negative_cvar, initial_weights, args=(log_returns, 0.05), method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights_cvar = result_cvar.x

optimal_portfolio_return_cvar = expected_return(optimal_weights_cvar, log_returns)
optimal_portfolio_cvar = calculate_cvar(log_returns.dot(optimal_weights_cvar))

print(f"Optimized Portfolio Return: {optimal_portfolio_return_cvar:.4f}")
print(f"Optimized Portfolio CVaR: {optimal_portfolio_cvar:.4f}")
