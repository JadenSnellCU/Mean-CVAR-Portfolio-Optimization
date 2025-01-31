import pandas as pd

# load dataset
data = pd.read_csv("synthetic_options_data.csv")

# pivot the table to restructure data
pivoted_data = data.pivot_table(
    index=['Date', 'Ticker', 'Expiration', 'Strike', 'Adj_Close'], 
    columns='Type', 
    values='Price'
).reset_index()

# rename columns for clarity
pivoted_data.columns.name = None
pivoted_data.rename(columns={'call': 'Call_Price', 'put': 'Put_Price'}, inplace=True)

# save the reorganized dataset
pivoted_data.to_csv("restructured_dataset.csv", index=False)
