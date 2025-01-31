import yfinance as yf
import pandas as pd
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm

def bs_call_price(S, K, r, sigma, T):
    d1 = (log(S/K) + (r + sigma**2/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

def bs_put_price(S, K, r, sigma, T):
    d1 = (log(S/K) + (r + sigma**2/2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

tickers = ["AAPL", "NVDA"]
start_date = "2023-01-01"
end_date = "2025-01-01" 
r = 0.05
option_days_to_expiry = 365
T = option_days_to_expiry / 365.0
df_list = []

# download historical stock data
stock_data = yf.download(tickers, start_date, end_date)
adj_close = stock_data['Adj Close'].dropna()

for ticker in tickers:
    # estimate volatility from historical data
    returns = adj_close[ticker].pct_change().dropna()
    sigma = np.std(returns)*np.sqrt(252) # annualized volatility

    # generate yearly contract start dates (start of each year)
    yearly_starts = pd.date_range(start_date, end_date, freq='YS')
    yearly_starts = yearly_starts[yearly_starts < end_date]

    for contract_start in yearly_starts:
        # compute expiration date one year from contract_start
        expiration_date = contract_start + pd.Timedelta(days=option_days_to_expiry)
        if expiration_date > adj_close.index[-1]:
            # ff we don't have data until the expiration, break 
            break

        # jf contract_start is not a trading day, find the nearest available date
        if contract_start not in adj_close.index:
            #find the closest trading day:
            idx = adj_close.index.get_indexer([contract_start], method='nearest')
            contract_start = adj_close.index[idx][0]


        K = adj_close.loc[contract_start, ticker]

        # get all trading days from contract_start to expiration_date
        contract_dates = adj_close.loc[contract_start:expiration_date].index


        base_price = adj_close.loc[contract_start, ticker]
        call_strike = base_price   
        put_strike = base_price   

    for current_date in contract_dates:
        days_to_exp = (expiration_date - current_date).days
        if days_to_exp <= 0:
            continue
        current_price = adj_close.loc[current_date, ticker]
        T_current = days_to_exp / 365.0

        # compute call price at call_strike
        call_price = bs_call_price(current_price, call_strike, r, sigma, T_current)
        # compute put price at put_strike
        put_price = bs_put_price(current_price, put_strike, r, sigma, T_current)
        
        df_list.append({
            "Date": current_date,
            "Ticker": ticker,
            "Expiration": expiration_date,
            "Strike": call_strike,
            "Type": "call",
            "Price": call_price,
            "Adj_Close": current_price
        })
        df_list.append({
            "Date": current_date,
            "Ticker": ticker,
            "Expiration": expiration_date,
            "Strike": put_strike,
            "Type": "put",
            "Price": put_price,
            "Adj_Close": current_price
        })
options_df = pd.DataFrame(df_list)
options_df.to_csv('data/synthetic_options_data_yearly.csv', index=False)
