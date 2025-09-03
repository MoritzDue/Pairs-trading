import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class PairsTradingStrategy:
    def __init__(self, tickers, start_date=None, end_date=None, min_data_threshold=0.8):
        """
        Single-pair trading strategy
        """
        self.tickers = tickers
        self.end_date = end_date or datetime.today()
        self.start_date = start_date or (self.end_date - timedelta(days=3*365))
        self.min_data_threshold = min_data_threshold
        self.adj_close = None
        self.spread = None
        self.zscore = None
        self.signals = None
        self.positions = None
        self.strategy_returns = None
        self.cumulative_returns = None

    def download_data(self):
        """Download data"""
        print("Downloading data...")
        data = yf.download(
            tickers=self.tickers,
            start=self.start_date.strftime('%Y-%m-%d'),
            end=self.end_date.strftime('%Y-%m-%d'),
            interval='1d',
            group_by='ticker',
            auto_adjust=True,
            progress=False
        )

        adj_close = pd.DataFrame()
        for ticker in self.tickers:
            if ticker in data.columns.levels[0]:
                price_data = data[ticker]['Close']
                data_availability = price_data.dropna().shape[0] / price_data.shape[0]
                if data_availability >= self.min_data_threshold:
                    adj_close[ticker] = price_data
                else:
                    print(f"Warning: {ticker} excluded - only {data_availability:.1%} data available")

        self.adj_close = adj_close.dropna()
        print(f"Successfully loaded data for {len(self.adj_close.columns)} tickers")
        print(f"Data period: {self.adj_close.index[0].strftime('%Y-%m-%d')} to {self.adj_close.index[-1].strftime('%Y-%m-%d')}")

    def analyze_pair(self, entry_threshold=1.0, exit_threshold=0.3):
        """Analyze chosen pair"""
        stock_x = self.adj_close[self.tickers[0]]
        stock_y = self.adj_close[self.tickers[1]]

        print(f"Analyzing pair: {self.tickers[0]} - {self.tickers[1]}")

        # Simple OLS hedge ratio
        model = sm.OLS(stock_x, sm.add_constant(stock_y)).fit()
        self.beta = model.params[1]

        # Spread
        self.spread = stock_x - self.beta * stock_y

        # Rolling z-score
        window = min(60, len(self.spread) // 4)
        rolling_mean = self.spread.rolling(window=window).mean()
        rolling_std = self.spread.rolling(window=window).std()
        self.zscore = (self.spread - rolling_mean) / rolling_std

        # Trading signals
        self.signals = pd.DataFrame(index=self.zscore.index)
        self.signals['zscore'] = self.zscore
        self.signals['long'] = self.zscore < -entry_threshold
        self.signals['short'] = self.zscore > entry_threshold
        self.signals['exit'] = self.zscore.abs() < exit_threshold

        print(f"Hedge ratio (beta): {self.beta:.4f}")
        print(f"Entry threshold: ±{entry_threshold}, Exit threshold: ±{exit_threshold}")

    def backtest_strategy(self, max_position_size=0.1):
        """Simple backtest"""
        self.positions = pd.DataFrame(index=self.signals.index)
        self.positions['position'] = 0.0

        current_position = 0.0
        trade_count = 0
        days_in_trade = 0
        max_days = 20

        for date in self.signals.index:
            if pd.isna(self.signals.loc[date, 'zscore']):
                self.positions.loc[date, 'position'] = current_position
                continue

            if current_position != 0:
                days_in_trade += 1

            # Exit conditions
            should_exit = (self.signals.loc[date, 'exit'] or 
                          days_in_trade >= max_days or
                          abs(self.signals.loc[date, 'zscore']) > 2.5)

            if should_exit and current_position != 0:
                current_position = 0.0
                days_in_trade = 0
            elif current_position == 0:
                if self.signals.loc[date, 'long']:
                    current_position = max_position_size
                    trade_count += 1
                    days_in_trade = 1
                elif self.signals.loc[date, 'short']:
                    current_position = -max_position_size
                    trade_count += 1
                    days_in_trade = 1

            self.positions.loc[date, 'position'] = current_position

        # Returns
        spread_returns = self.spread.pct_change().fillna(0)
        self.strategy_returns = (self.positions['position'].shift(1) * spread_returns).fillna(0)
        self.cumulative_returns = (1 + self.strategy_returns).cumprod()

        # Performance metrics
        total_return = self.cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(self.strategy_returns)) - 1
        volatility = self.strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        rolling_max = self.cumulative_returns.expanding().max()
        drawdown = (self.cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        win_rate = len(self.strategy_returns[self.strategy_returns > 0]) / len(self.strategy_returns[self.strategy_returns != 0]) if len(self.strategy_returns[self.strategy_returns != 0]) > 0 else 0

        print("\n=== SINGLE-PAIR CONSERVATIVE RESULTS ===")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annual_return:.2%}")
        print(f"Volatility: {volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Total Trades: {trade_count}")
        print(f"Win Rate: {win_rate:.2%}")

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': trade_count,
            'win_rate': win_rate
        }

    def plot_results(self):
        """Plot prices, spread, zscore, cumulative returns"""
        stock_x = self.adj_close[self.tickers[0]]
        stock_y = self.adj_close[self.tickers[1]]

        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

        # Prices
        axes[0].plot(stock_x, label=self.tickers[0])
        axes[0].plot(stock_y, label=self.tickers[1])
        axes[0].set_title("Stock Prices")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Spread
        axes[1].plot(self.spread, label='Spread', color='purple')
        axes[1].axhline(self.spread.mean(), color='red', linestyle='--', label='Mean')
        axes[1].set_title("Spread")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Z-score
        axes[2].plot(self.zscore, label='Z-score', color='black')
        axes[2].axhline(1.0, color='red', linestyle='--')
        axes[2].axhline(-1.0, color='green', linestyle='--')
        axes[2].axhline(0, color='blue', linestyle=':')
        axes[2].set_title("Z-score")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Cumulative returns
        if self.cumulative_returns is not None:
            axes[3].plot(self.cumulative_returns, label='Cumulative Returns', color='blue')
            axes[3].set_title("Cumulative Strategy Returns")
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    pair_strategy = PairsTradingStrategy(tickers=["XLE", "XLB"])
    pair_strategy.download_data()
    pair_strategy.analyze_pair(entry_threshold=1.0, exit_threshold=0.3)
    performance = pair_strategy.backtest_strategy(max_position_size=0.1)
    pair_strategy.plot_results()
