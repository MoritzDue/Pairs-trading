import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint

# Step 1: Download historical data
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)

tickers = ["BMW.DE", "DTE.DE", "SAP.DE", "ALV.DE", "BAYN.DE"]  # Example subset of DAX 40

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

