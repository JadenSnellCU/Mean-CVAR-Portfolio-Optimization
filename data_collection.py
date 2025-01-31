import yfinance as yf
import pandas as pd

tickers = ["AAPL", "NVDA"]
stock_data = {}

# download one year's worth of stock data
for ticker in tickers:
    data = yf.download(ticker,'2023-12-01','2024-12-01')
    stock_data[ticker] = data

df = pd.concat(stock_data, axis=1)
df.columns = ['_'.join(col).strip() for col in df.columns.values]
df.to_csv('sp500_stocks.csv', index=True)

merged_datasets = []

# parameters for filtering
min_volume = 10  # minimum volume to ensure some liquidity
min_open_interest = 10  # minimum open interest
max_spread = 5.0  # maximum allowed absolute bid-ask spread in dollars
strike_range_factor = 0.5  # keep strikes within ±50% of current price

for ticker in tickers:
    stock = yf.Ticker(ticker)
    expirations = stock.options

    # filter to standard monthly expirations 
    monthly_expirations = []
    for exp in expirations:
        exp_date = pd.to_datetime(exp)
        if exp_date.weekday() == 4 and 14 <= exp_date.day <= 21:
            monthly_expirations.append(exp)

    stock_data_df = pd.read_csv('sp500_stocks.csv', parse_dates=['Date'])
    adj_close_column = f"{ticker}_Adj Close_{ticker}"
    if adj_close_column not in stock_data_df.columns:
        print(f"Column {adj_close_column} not found in stock data. Skipping {ticker}.")
        continue

    #approximate current price to filter strikes
    current_price = stock_data_df[adj_close_column].iloc[-1]

    for expiration_date in monthly_expirations:
        options_chain = stock.option_chain(expiration_date)
        calls = options_chain.calls.copy()
        puts = options_chain.puts.copy()

        # add columns for Type and Expiration
        calls['Type'] = 'calls'
        puts['Type'] = 'puts'
        calls['Expiration'] = expiration_date
        puts['Expiration'] = expiration_date

        # combine calls and puts
        options_data = pd.concat([calls, puts])

        # convert lastTradeDate to a date
        options_data['Date'] = pd.to_datetime(options_data['lastTradeDate']).dt.date
        options_data['Date'] = pd.to_datetime(options_data['Date'])

        # filter based on liquidity and strike range
        options_data = options_data[(options_data['volume'] >= min_volume) & (options_data['openInterest'] >= min_open_interest)]
        lower_strike = current_price * (1 - strike_range_factor)
        upper_strike = current_price * (1 + strike_range_factor)
        options_data = options_data[(options_data['strike'] >= lower_strike) & (options_data['strike'] <= upper_strike)]
        options_data = options_data[(options_data['ask'] - options_data['bid']) <= max_spread]

        if options_data.empty:
            continue

        # merge with stock data
        ticker_stock_data = stock_data_df[['Date', adj_close_column]].rename(columns={adj_close_column: 'Adj_Close'})
        merged_data = pd.merge(ticker_stock_data, options_data, on='Date', how='inner')


        merged_datasets.append(merged_data)

# combine all data
if merged_datasets:
    final_combined_data = pd.concat(merged_datasets)
    final_combined_data.to_csv('sp500_combined_monthly_filtered.csv', index=False)
