import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PairsTradingStrategy:
    def __init__(self, tickers, start_date=None, end_date=None, min_data_threshold=0.8):
        """
        Initialize the pairs trading strategy
        
        Parameters:
        - tickers: list of stock tickers
        - start_date: start date for data collection
        - end_date: end date for data collection
        - min_data_threshold: minimum data availability threshold (0.8 = 80%)
        """
        self.tickers = tickers
        self.end_date = end_date or datetime.today()
        self.start_date = start_date or (self.end_date - timedelta(days=5*365))
        self.min_data_threshold = min_data_threshold
        self.adj_close = None
        self.coint_results = None
        self.best_pair = None
        self.spread = None
        self.zscore = None
        self.signals = None
        self.positions = None
        self.strategy_returns = None
        self.cumulative_returns = None
        
    def download_data(self):
        """Download data with error handling and data quality checks"""
        print("Downloading data...")
        try:
            data = yf.download(
                tickers=self.tickers,
                start=self.start_date.strftime('%Y-%m-%d'),
                end=self.end_date.strftime('%Y-%m-%d'),
                interval='1d',
                group_by='ticker',
                auto_adjust=True,
                progress=False
            )
            
            # Extract adjusted close prices
            adj_close = pd.DataFrame()
            valid_tickers = []
            
            for ticker in self.tickers:
                try:
                    if ticker in data.columns.levels[0]:  # Check if ticker exists in data
                        price_data = data[ticker]['Close']
                        # Check data availability
                        data_availability = price_data.dropna().shape[0] / price_data.shape[0]
                        
                        if data_availability >= self.min_data_threshold:
                            adj_close[ticker] = price_data
                            valid_tickers.append(ticker)
                        else:
                            print(f"Warning: {ticker} excluded - only {data_availability:.1%} data available")
                    else:
                        print(f"Warning: {ticker} not found in downloaded data")
                except Exception as e:
                    print(f"Error processing {ticker}: {str(e)}")
            
            self.adj_close = adj_close.dropna()
            self.valid_tickers = valid_tickers
            
            print(f"Successfully loaded data for {len(valid_tickers)} tickers")
            print(f"Data period: {self.adj_close.index[0].strftime('%Y-%m-%d')} to {self.adj_close.index[-1].strftime('%Y-%m-%d')}")
            
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
            raise
    
    def find_cointegrated_pairs(self, significance_level=0.05):
        """Find cointegrated pairs with improved statistics"""
        print("Testing for cointegration...")
        results = []
        
        for i in range(len(self.valid_tickers)):
            for j in range(i + 1, len(self.valid_tickers)):
                ticker_i, ticker_j = self.valid_tickers[i], self.valid_tickers[j]
                
                try:
                    # Perform cointegration test
                    score, pvalue, crit_values = coint(
                        self.adj_close[ticker_i], 
                        self.adj_close[ticker_j]
                    )
                    
                    # Calculate correlation for additional insight
                    correlation = self.adj_close[ticker_i].corr(self.adj_close[ticker_j])
                    
                    results.append({
                        'Pair': (ticker_i, ticker_j),
                        'Cointegration Score': score,
                        'p-value': pvalue,
                        'Critical Value (1%)': crit_values[0],
                        'Critical Value (5%)': crit_values[1],
                        'Critical Value (10%)': crit_values[2],
                        'Correlation': correlation,
                        'Cointegrated (5%)': pvalue < significance_level
                    })
                except Exception as e:
                    print(f"Error testing pair {ticker_i}-{ticker_j}: {str(e)}")
        
        self.coint_results = pd.DataFrame(results).sort_values('p-value')
        
        # Show summary
        cointegrated_pairs = self.coint_results[self.coint_results['Cointegrated (5%)']].shape[0]
        print(f"Found {cointegrated_pairs} cointegrated pairs at 5% significance level")
        
        return self.coint_results
    
    def analyze_best_pair(self, entry_threshold=2.0, exit_threshold=0.5):
        """Analyze the best cointegrated pair"""
        if self.coint_results is None:
            raise ValueError("Must run find_cointegrated_pairs first")
        
        # Get best pair (lowest p-value)
        self.best_pair = self.coint_results.iloc[0]['Pair']
        stock_x = self.adj_close[self.best_pair[0]]
        stock_y = self.adj_close[self.best_pair[1]]
        
        print(f"Analyzing best pair: {self.best_pair[0]} - {self.best_pair[1]}")
        print(f"P-value: {self.coint_results.iloc[0]['p-value']:.6f}")
        
        # Estimate hedge ratio using linear regression
        model = sm.OLS(stock_x, sm.add_constant(stock_y)).fit()
        self.beta = model.params[1]
        self.alpha = model.params[0]
        
        # Calculate spread
        self.spread = stock_x - self.beta * stock_y
        
        # Calculate z-score
        spread_mean = self.spread.mean()
        spread_std = self.spread.std()
        self.zscore = (self.spread - spread_mean) / spread_std
        
        # Generate trading signals
        self.signals = pd.DataFrame(index=self.zscore.index)
        self.signals['zscore'] = self.zscore
        self.signals['long'] = self.zscore < -entry_threshold
        self.signals['short'] = self.zscore > entry_threshold
        self.signals['exit'] = self.zscore.abs() < exit_threshold
        
        print(f"Hedge ratio (beta): {self.beta:.4f}")
        print(f"Alpha: {self.alpha:.4f}")
        print(f"Spread mean: {spread_mean:.4f}")
        print(f"Spread std: {spread_std:.4f}")
        
    def backtest_strategy(self):
        """Backtest the pairs trading strategy with enhanced metrics"""
        if self.signals is None:
            raise ValueError("Must run analyze_best_pair first")
        
        # Generate positions
        self.positions = pd.DataFrame(index=self.signals.index)
        self.positions['position'] = 0
        
        current_position = 0
        trade_count = 0
        
        for date in self.signals.index:
            if self.signals.loc[date, 'long'] and current_position == 0:
                current_position = 1
                trade_count += 1
            elif self.signals.loc[date, 'short'] and current_position == 0:
                current_position = -1
                trade_count += 1
            elif self.signals.loc[date, 'exit'] and current_position != 0:
                current_position = 0
            
            self.positions.loc[date, 'position'] = current_position
        
        # Calculate returns
        spread_returns = self.spread.pct_change().fillna(0)
        self.strategy_returns = self.positions['position'].shift(1) * spread_returns
        self.cumulative_returns = (1 + self.strategy_returns).cumprod()
        
        # Calculate performance metrics
        total_return = self.cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(self.strategy_returns)) - 1
        volatility = self.strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        rolling_max = self.cumulative_returns.expanding().max()
        drawdown = (self.cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate calculation
        winning_trades = self.strategy_returns[self.strategy_returns > 0].count()
        losing_trades = self.strategy_returns[self.strategy_returns < 0].count()
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        print("\n=== BACKTEST RESULTS ===")
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
    
    def plot_analysis(self, figsize=(15, 12)):
        """Create comprehensive visualization of the strategy"""
        if self.cumulative_returns is None:
            raise ValueError("Must run backtest_strategy first")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Pairs Trading Analysis: {self.best_pair[0]} - {self.best_pair[1]}', fontsize=16)
        
        # 1. Price series
        ax1 = axes[0, 0]
        self.adj_close[self.best_pair[0]].plot(ax=ax1, label=self.best_pair[0], alpha=0.8)
        self.adj_close[self.best_pair[1]].plot(ax=ax1, label=self.best_pair[1], alpha=0.8)
        ax1.set_title('Price Series')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Spread and Z-score
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()
        
        ax2.plot(self.spread.index, self.spread, color='blue', alpha=0.7, label='Spread')
        ax2_twin.plot(self.zscore.index, self.zscore, color='red', alpha=0.7, label='Z-score')
        
        # Add threshold lines
        ax2_twin.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Entry threshold')
        ax2_twin.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
        ax2_twin.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Exit threshold')
        ax2_twin.axhline(y=-0.5, color='orange', linestyle='--', alpha=0.5)
        ax2_twin.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax2.set_ylabel('Spread', color='blue')
        ax2_twin.set_ylabel('Z-score', color='red')
        ax2.set_title('Spread and Z-score')
        ax2.grid(True, alpha=0.3)
        
        # 3. Trading positions
        ax3 = axes[1, 0]
        position_colors = self.positions['position'].map({1: 'green', -1: 'red', 0: 'gray'})
        ax3.scatter(self.positions.index, self.positions['position'], 
                   c=position_colors, alpha=0.6, s=1)
        ax3.set_ylabel('Position')
        ax3.set_title('Trading Positions')
        ax3.set_ylim(-1.5, 1.5)
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative returns
        ax4 = axes[1, 1]
        self.cumulative_returns.plot(ax=ax4, color='green', linewidth=2)
        ax4.axhline(y=1, color='black', linestyle='-', alpha=0.3)
        ax4.set_ylabel('Cumulative Return')
        ax4.set_title('Strategy Performance')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_cointegration_heatmap(self, top_n=20):
        """Plot heatmap of top cointegrated pairs"""
        if self.coint_results is None:
            raise ValueError("Must run find_cointegrated_pairs first")
        
        # Create matrix for heatmap
        top_pairs = self.coint_results.head(top_n)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # P-values heatmap
        pvalue_matrix = pd.DataFrame(index=self.valid_tickers, columns=self.valid_tickers)
        for _, row in top_pairs.iterrows():
            ticker1, ticker2 = row['Pair']
            pvalue_matrix.loc[ticker1, ticker2] = row['p-value']
            pvalue_matrix.loc[ticker2, ticker1] = row['p-value']
        
        # Fill diagonal with NaN
        for ticker in self.valid_tickers:
            pvalue_matrix.loc[ticker, ticker] = np.nan
        
        # Convert to float
        pvalue_matrix = pvalue_matrix.astype(float)
        
        sns.heatmap(pvalue_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   ax=ax1, cbar_kws={'label': 'P-value'})
        ax1.set_title('Cointegration P-values (Top Pairs)')
        
        # Correlation heatmap
        corr_matrix = self.adj_close[self.valid_tickers].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                   ax=ax2, cbar_kws={'label': 'Correlation'})
        ax2.set_title('Price Correlation Matrix')
        
        plt.tight_layout()
        plt.show()
    
    def run_full_analysis(self, entry_threshold=2.0, exit_threshold=0.5, 
                         significance_level=0.05, plot_results=True):
        """Run complete pairs trading analysis"""
        print("Starting full pairs trading analysis...")
        print("=" * 50)
        
        # Step 1: Download data
        self.download_data()
        
        # Step 2: Find cointegrated pairs
        coint_results = self.find_cointegrated_pairs(significance_level)
        
        # Check if we have any cointegrated pairs
        if coint_results.empty or not any(coint_results['Cointegrated (5%)']):
            print("No cointegrated pairs found at 5% significance level!")
            return None
        
        # Step 3: Analyze best pair
        self.analyze_best_pair(entry_threshold, exit_threshold)
        
        # Step 4: Backtest strategy
        performance = self.backtest_strategy()
        
        # Step 5: Plot results
        if plot_results:
            self.plot_analysis()
            self.plot_cointegration_heatmap()
        
        return performance

# DAX 40 tickers
tickers = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE", "CBK.DE", "CON.DE",
    "1COV.DE", "DHER.DE", "DBK.DE", "DB1.DE", "DTE.DE", "DTG.DE", "EOAN.DE", "FME.DE", "FRE.DE", "HEI.DE",
    "HEN3.DE", "IFX.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "PAH3.DE", "P911.DE", "QIA.DE", "RWE.DE", "SAP.DE",
    "SIE.DE", "SHL.DE", "SY1.DE", "SRT3.DE", "VOW3.DE", "VNA.DE", "ZAL.DE", "RHM.DE", "ENR.DE"
]

# Example usage
if __name__ == "__main__":
    # Initialize strategy
    strategy = PairsTradingStrategy(
        tickers=tickers, 
        min_data_threshold=0.8  # Require 80% data availability
    )
    
    # Run complete analysis
    performance = strategy.run_full_analysis(
        entry_threshold=2.0,
        exit_threshold=0.5,
        significance_level=0.05,
        plot_results=True
    )
    
    # Display top cointegrated pairs
    if strategy.coint_results is not None:
        print("\n=== TOP 10 COINTEGRATED PAIRS ===")
        print(strategy.coint_results.head(10)[['Pair', 'p-value', 'Correlation', 'Cointegrated (5%)']].to_string(index=False))
        
        # Show additional statistics for best pair
        print(f"\n=== BEST PAIR DETAILS ===")
        print(f"Pair: {strategy.best_pair}")
        print(f"Hedge Ratio: {strategy.beta:.4f}")
        print(f"Recent Z-scores:")
        print(strategy.zscore.tail().to_string())
