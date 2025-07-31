import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

# Step 1: Download historical data
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)


# DAX 40 tickers on Yahoo Finance
tickers = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE", "CBK.DE", "CON.DE",
    "1COV.DE", "DHER.DE", "DBK.DE", "DB1.DE", "DTE.DE", "DTG.DE", "EOAN.DE", "FME.DE", "FRE.DE", "HEI.DE",
    "HEN3.DE", "IFX.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "PAH3.DE", "P911.DE", "QIA.DE", "RWE.DE", "SAP.DE",
    "SIE.DE", "SHL.DE", "SY1.DE", "SRT3.DE", "VOW3.DE", "VNA.DE", "ZAL.DE", "RHM.DE", "MUV2.DE", "ENR.DE"
]


data = yf.download(
    tickers=tickers,
    start=start_date.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    interval='1d',
    group_by='ticker',
    auto_adjust=True
)

# Step 2: Extract adjusted close prices
adj_close = pd.DataFrame({ticker: data[ticker]['Close'] for ticker in tickers})

# Step 3: Drop rows with missing values
adj_close.dropna(inplace=True)

# Step 4: Perform cointegration test on all pairs
results = []
for i in range(len(tickers)):
    for j in range(i + 1, len(tickers)):
        score, pvalue, _ = coint(adj_close[tickers[i]], adj_close[tickers[j]])
        results.append({
            'Pair': (tickers[i], tickers[j]),
            'Cointegration Score': score,
            'p-value': pvalue
        })

# Convert results to DataFrame and sort by p-value
coint_results = pd.DataFrame(results).sort_values('p-value')
print(coint_results)


# Step 5: Spread and Z-score for best pair
best_pair = min(results, key=lambda x: x['p-value'])['Pair']
stock_x = adj_close[best_pair[0]]
stock_y = adj_close[best_pair[1]]

# Estimate hedge ratio (beta) via linear regression
model = sm.OLS(stock_x, sm.add_constant(stock_y)).fit()
beta = model.params[1]

# Calculate spread
spread = stock_x - beta * stock_y

# Calculate z-score
spread_mean = spread.mean()
spread_std = spread.std()
zscore = (spread - spread_mean) / spread_std

# Optional: print last few z-scores

print(f"Best pair: {best_pair}")
print(zscore.tail())
# Define trading signal thresholds
entry_threshold = 2.0
exit_threshold = 0.5


# Step 6: Generate trading signals
entry_threshold = 2.0
exit_threshold = 0.5

signals = pd.DataFrame(index=zscore.index)
signals['zscore'] = zscore
signals['long'] = zscore < -entry_threshold
signals['short'] = zscore > entry_threshold
signals['exit'] = zscore.abs() < exit_threshold

# Step 7: Backtest strategy
positions = pd.DataFrame(index=signals.index)
positions['position'] = 0  # 1 for long, -1 for short

current_position = 0
for date in signals.index:
    if signals.loc[date, 'long'] and current_position == 0:
        current_position = 1
    elif signals.loc[date, 'short'] and current_position == 0:
        current_position = -1
    elif signals.loc[date, 'exit'] and current_position != 0:
        current_position = 0
    positions.loc[date, 'position'] = current_position

# Calculate daily returns of the spread
spread_returns = spread.pct_change().fillna(0)

# Strategy returns
strategy_returns = positions['position'].shift(1) * spread_returns
cumulative_returns = (1 + strategy_returns).cumprod()

# Output final result
print(f"Best pair: {best_pair}")
print("Final cumulative return:", cumulative_returns.iloc[-1])



