# Pairs Trading Strategy

A statistical arbitrage project implementing a **pairs trading strategy** in Python. The strategy identifies cointegrated stock pairs, constructs a mean-reverting spread, and trades using z-score signals.

## Overview

- **Data Source:** Yahoo Finance (via `yfinance`)  
- **Assets:** Example pair – XLE (Energy) vs XLB (Materials)  
- **Methodology:**  
  - Estimate hedge ratio via OLS regression  
  - Compute spread and rolling z-score  
  - Generate entry/exit signals based on thresholds  
  - Backtest with position sizing, holding limits, and performance tracking  

## Features

- Automated data download for chosen tickers  
- Hedge ratio estimation and spread calculation  
- Trading signals from rolling z-score  
- Backtesting with PnL, Sharpe Ratio, drawdowns, and trade count  
- Visualization of prices, spread, z-score, and cumulative returns  

## Results (XLE – XLB, 2022–2025)

- **Hedge Ratio (β):** 0.6598  
- **Entry Threshold:** ±1.0  
- **Exit Threshold:** ±0.3  

### Performance Metrics

- Total Return: 15.25%  
- Annualized Return: 4.88%  
- Volatility: 3.94%  
- Sharpe Ratio: 1.24  
- Maximum Drawdown: -3.34%  
- Total Trades: 45  
- Win Rate: 53.09%  

### Example Plots

- Stock prices  
- Spread with mean line  
- Z-score with thresholds  
- Cumulative returns  
