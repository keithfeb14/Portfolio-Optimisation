import yfinance as yf
import pandas as pd 
from datetime import datetime, timedelta
import numpy as np 
from scipy.optimize import minimize

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

