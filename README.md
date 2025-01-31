# Mean-CVAR-Portfolio-Optimization

This project implements a Mean-CVaR Portfolio Optimization model incorporating stocks and options, leveraging MATLAB and Python to construct an efficient frontier and identify optimal portfolio allocations based on risk-adjusted returns. It integrates real S&P 500 market data and synthetic options datasets to test different portfolio configurations and detect potential arbitrage opportunities.

## Features

-Portfolio Optimization using Mean-CVaR Framework: Implements convex optimization with Conditional Value at Risk (CVaR) constraints.
-Real-Time Data Collection: Utilizes yfinance to fetch stock and options market data, filtering by liquidity, bid-ask spread, and strike price ranges.
-Efficient Frontier Construction: Uses MATLAB’s fmincon optimizer to generate portfolios minimizing risk for a given return level.
-Arbitrage Detection in Options Markets: Tests for pricing inefficiencies across expiration dates and detects arbitrage opportunities.
-Support for Multiple Asset Classes: Includes stocks, long/short call and put options, and risk-free assets.

## Structure

├── data_collection.py      # Fetches stock and options data from yfinance
├── data_clean.py           # Cleans and restructures the dataset
├── gradproject.m           # MATLAB script for Mean-CVaR portfolio optimization (stocks & calls)
├── gradproject_all_options.m # Includes long/short calls and puts
├── real_SP500_all_options.m  # Uses real S&P 500 stock and options data
├── real_SP500_call.m        # Simplified model with only calls
├── real_SP500_synthetic.m   # Uses synthetic options dataset
├── README.md               # Documentation

## Future Work

-Analyze out-of-sample performance using tick data from CBOE for an underlying asset
-Refine data collection and cleaning pipelines
-Implement some machine learning principles to potentially train the model

