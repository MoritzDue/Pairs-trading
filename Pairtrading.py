import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PairsTradingStrategy:
    def __init__(self, tickers, start_date=None, end_date=None, min_data_threshold=0.8):
        """
        Single-pair trading strategy (conservative baseline)
        """
        self.tickers = tickers
        self.end_date = end_date or datetime.today()
        self.start_date = start_date or (self.end_date - timedelta(days=3*365))
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
                    if ticker in data.columns.levels[0]:
                        price_data = data[ticker]['Close']
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
        """Find cointegrated pairs"""
        print("Testing for cointegration...")
        results = []
        
        for i in range(len(self.valid_tickers)):
            for j in range(i + 1, len(self.valid_tickers)):
                ticker_i, ticker_j = self.valid_tickers[i], self.valid_tickers[j]
                
                try:
                    score, pvalue, crit_values = coint(
                        self.adj_close[ticker_i], 
                        self.adj_close[ticker_j]
                    )
                    
                    correlation = self.adj_close[ticker_i].corr(self.adj_close[ticker_j])
                    
                    results.append({
                        'Pair': (ticker_i, ticker_j),
                        'p-value': pvalue,
                        'Correlation': correlation,
                        'Cointegrated (5%)': pvalue < significance_level
                    })
                except Exception as e:
                    print(f"Error testing pair {ticker_i}-{ticker_j}: {str(e)}")
        
        self.coint_results = pd.DataFrame(results).sort_values('p-value')
        
        cointegrated_pairs = self.coint_results[self.coint_results['Cointegrated (5%)']].shape[0]
        print(f"Found {cointegrated_pairs} cointegrated pairs at {significance_level*100}% significance level")
        
        return self.coint_results
    
    def analyze_best_pair(self, entry_threshold=1.0, exit_threshold=0.3):
        """Analyze the best cointegrated pair with conservative approach"""
        if self.coint_results is None:
            raise ValueError("Must run find_cointegrated_pairs first")
        
        # Get best pair (lowest p-value) with good correlation
        filtered_pairs = self.coint_results[
            (self.coint_results['p-value'] < 0.05) &
            (self.coint_results['Correlation'].abs() > 0.6) &
            (self.coint_results['Correlation'].abs() < 0.98)
        ]
        
        if filtered_pairs.empty:
            print("No suitable pairs found, using best available...")
            self.best_pair = self.coint_results.iloc[0]['Pair']
        else:
            self.best_pair = filtered_pairs.iloc[0]['Pair']
        
        stock_x = self.adj_close[self.best_pair[0]]
        stock_y = self.adj_close[self.best_pair[1]]
        
        print(f"Analyzing pair: {self.best_pair[0]} - {self.best_pair[1]}")
        
        # Simple OLS hedge ratio
        model = sm.OLS(stock_x, sm.add_constant(stock_y)).fit()
        self.beta = model.params[1]
        
        # Calculate spread
        self.spread = stock_x - self.beta * stock_y
        
        # Calculate rolling z-score (more responsive)
        window = min(60, len(self.spread) // 4)
        rolling_mean = self.spread.rolling(window=window).mean()
        rolling_std = self.spread.rolling(window=window).std()
        self.zscore = (self.spread - rolling_mean) / rolling_std
        
        # Generate conservative signals
        self.signals = pd.DataFrame(index=self.zscore.index)
        self.signals['zscore'] = self.zscore
        self.signals['long'] = self.zscore < -entry_threshold
        self.signals['short'] = self.zscore > entry_threshold
        self.signals['exit'] = self.zscore.abs() < exit_threshold
        
        print(f"Hedge ratio (beta): {self.beta:.4f}")
        print(f"Entry threshold: ±{entry_threshold}, Exit threshold: ±{exit_threshold}")
    
    def backtest_strategy(self, max_position_size=0.1):
        """Conservative backtest with small position sizes"""
        if self.signals is None:
            raise ValueError("Must run analyze_best_pair first")
        
        self.positions = pd.DataFrame(index=self.signals.index)
        self.positions['position'] = 0.0
        
        current_position = 0.0
        trade_count = 0
        days_in_trade = 0
        max_days = 20  # Short holding period
        
        for date in self.signals.index:
            if pd.isna(self.signals.loc[date, 'zscore']):
                self.positions.loc[date, 'position'] = current_position
                continue
            
            zscore_today = self.signals.loc[date, 'zscore']
            
            if current_position != 0:
                days_in_trade += 1
            
            # Exit conditions
            should_exit = (self.signals.loc[date, 'exit'] or 
                          days_in_trade >= max_days or
                          abs(zscore_today) > 2.5)
            
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
        
        # Calculate returns with transaction costs
        spread_returns = self.spread.pct_change().fillna(0)
        transaction_cost = 0.001  # 10 bps
        position_changes = self.positions['position'].diff().abs()
        transaction_costs = position_changes * transaction_cost
        
        self.strategy_returns = (self.positions['position'].shift(1) * spread_returns - transaction_costs).fillna(0)
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

class MultiPairsTradingStrategy:
    def __init__(self, tickers, start_date=None, end_date=None, min_data_threshold=0.8):
        """
        Multi-pair trading strategy with portfolio approach
        """
        self.tickers = tickers
        self.end_date = end_date or datetime.today()
        self.start_date = start_date or (self.end_date - timedelta(days=3*365))
        self.min_data_threshold = min_data_threshold
        self.adj_close = None
        self.selected_pairs = []
        self.portfolio_returns = None
        self.portfolio_cumulative = None
        
    def download_data(self):
        """Download data with error handling and data quality checks"""
        print("Downloading data for multi-pair strategy...")
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
            
            adj_close = pd.DataFrame()
            valid_tickers = []
            
            for ticker in self.tickers:
                try:
                    if ticker in data.columns.levels[0]:
                        price_data = data[ticker]['Close']
                        data_availability = price_data.dropna().shape[0] / price_data.shape[0]
                        
                        if data_availability >= self.min_data_threshold:
                            adj_close[ticker] = price_data
                            valid_tickers.append(ticker)
                        else:
                            print(f"Warning: {ticker} excluded - only {data_availability:.1%} data available")
                except Exception as e:
                    print(f"Error processing {ticker}: {str(e)}")
            
            self.adj_close = adj_close.dropna()
            self.valid_tickers = valid_tickers
            
            print(f"Successfully loaded data for {len(valid_tickers)} tickers")
            
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
            raise
    
    def find_tradeable_pairs(self, min_correlation=0.6, max_correlation=0.95):
        """Find pairs with good trading characteristics"""
        print("Finding tradeable pairs...")
        
        # Find cointegrated pairs
        results = []
        for i in range(len(self.valid_tickers)):
            for j in range(i + 1, len(self.valid_tickers)):
                ticker_i, ticker_j = self.valid_tickers[i], self.valid_tickers[j]
                
                try:
                    score, pvalue, _ = coint(
                        self.adj_close[ticker_i], 
                        self.adj_close[ticker_j]
                    )
                    
                    correlation = self.adj_close[ticker_i].corr(self.adj_close[ticker_j])
                    
                    results.append({
                        'pair': (ticker_i, ticker_j),
                        'pvalue': pvalue,
                        'correlation': correlation
                    })
                except:
                    continue
        
        # Filter by quality criteria
        good_pairs = []
        for result in results:
            if (result['pvalue'] < 0.05 and 
                min_correlation <= abs(result['correlation']) <= max_correlation):
                
                # Test mean reversion quality
                try:
                    half_life = self.calculate_half_life(result['pair'])
                    if 5 <= half_life <= 40:  # Reasonable half-life
                        good_pairs.append({
                            'pair': result['pair'],
                            'pvalue': result['pvalue'],
                            'correlation': result['correlation'],
                            'half_life': half_life
                        })
                except:
                    continue
        
        # Sort by p-value and take top 5
        good_pairs.sort(key=lambda x: x['pvalue'])
        self.selected_pairs = good_pairs[:5]
        
        print(f"Selected {len(self.selected_pairs)} pairs for trading:")
        for pair_info in self.selected_pairs:
            print(f"  {pair_info['pair']}: p-val={pair_info['pvalue']:.4f}, half-life={pair_info['half_life']:.1f}d")
        
        return self.selected_pairs
    
    def calculate_half_life(self, pair):
        """Calculate half-life of mean reversion"""
        stock_x = self.adj_close[pair[0]]
        stock_y = self.adj_close[pair[1]]
        
        # Calculate spread
        model = sm.OLS(stock_x, sm.add_constant(stock_y)).fit()
        spread = stock_x - model.params[1] * stock_y
        
        # Half-life calculation
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        common_idx = spread_lag.index.intersection(spread_diff.index)
        spread_lag = spread_lag[common_idx]
        spread_diff = spread_diff[common_idx]
        
        try:
            reg_model = sm.OLS(spread_diff, sm.add_constant(spread_lag)).fit()
            beta = reg_model.params[1]
            half_life = -np.log(2) / np.log(1 + beta) if beta < 0 else float('inf')
            return max(1, half_life)
        except:
            return float('inf')
    
    def create_pair_strategy(self, pair_info):
        """Create trading strategy for a single pair"""
        pair = pair_info['pair']
        stock_x = self.adj_close[pair[0]]
        stock_y = self.adj_close[pair[1]]
        
        # Calculate spread
        model = sm.OLS(stock_x, sm.add_constant(stock_y)).fit()
        beta = model.params[1]
        spread = stock_x - beta * stock_y
        
        # Rolling z-score
        window = 60
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        zscore = (spread - rolling_mean) / rolling_std
        
        # Generate signals
        entry_threshold = 1.2
        exit_threshold = 0.3
        
        signals = pd.DataFrame(index=zscore.index)
        signals['zscore'] = zscore
        signals['long'] = zscore < -entry_threshold
        signals['short'] = zscore > entry_threshold
        signals['exit'] = zscore.abs() < exit_threshold
        
        # Backtest
        positions = pd.DataFrame(index=signals.index)
        positions['position'] = 0.0
        
        current_position = 0.0
        days_in_trade = 0
        max_days = 25
        
        for date in signals.index:
            if pd.isna(signals.loc[date, 'zscore']):
                positions.loc[date, 'position'] = current_position
                continue
            
            if current_position != 0:
                days_in_trade += 1
            
            should_exit = (signals.loc[date, 'exit'] or 
                          days_in_trade >= max_days or
                          abs(signals.loc[date, 'zscore']) > 2.5)
            
            if should_exit and current_position != 0:
                current_position = 0.0
                days_in_trade = 0
            elif current_position == 0:
                if signals.loc[date, 'long']:
                    current_position = 0.15  # 15% position size
                    days_in_trade = 1
                elif signals.loc[date, 'short']:
                    current_position = -0.15
                    days_in_trade = 1
            
            positions.loc[date, 'position'] = current_position
        
        # Calculate returns
        spread_returns = spread.pct_change().fillna(0)
        pair_returns = positions['position'].shift(1) * spread_returns
        
        return {
            'pair': pair,
            'returns': pair_returns,
            'positions': positions
        }
    
    def backtest_portfolio(self):
        """Backtest portfolio of pairs"""
        if not self.selected_pairs:
            raise ValueError("Must select pairs first")
        
        print("Backtesting portfolio of pairs...")
        
        # Create strategies for each pair
        pair_strategies = []
        for pair_info in self.selected_pairs:
            try:
                strategy = self.create_pair_strategy(pair_info)
                pair_strategies.append(strategy)
            except Exception as e:
                print(f"Failed to create strategy for {pair_info['pair']}: {e}")
        
        if not pair_strategies:
            print("No successful pair strategies created")
            return None
        
        # Combine returns (equal weight)
        all_returns = pd.DataFrame()
        for i, strategy in enumerate(pair_strategies):
            all_returns[f'pair_{i}'] = strategy['returns']
        
        # Portfolio returns
        self.portfolio_returns = all_returns.mean(axis=1, skipna=True)
        
        # Apply transaction costs
        self.portfolio_returns = self.portfolio_returns - 0.0005  # 5 bps daily cost
        self.portfolio_cumulative = (1 + self.portfolio_returns).cumprod()
        
        # Calculate metrics
        total_return = self.portfolio_cumulative.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(self.portfolio_returns)) - 1
        volatility = self.portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        rolling_max = self.portfolio_cumulative.expanding().max()
        drawdown = (self.portfolio_cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        win_rate = len(self.portfolio_returns[self.portfolio_returns > 0]) / len(self.portfolio_returns[self.portfolio_returns != 0]) if len(self.portfolio_returns[self.portfolio_returns != 0]) > 0 else 0
        
        print("\n=== MULTI-PAIR PORTFOLIO RESULTS ===")
        print(f"Number of Pairs: {len(pair_strategies)}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annual_return:.2%}")
        print(f"Volatility: {volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Daily Win Rate: {win_rate:.2%}")
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_pairs': len(pair_strategies)
        }
    
    def plot_results(self, figsize=(12, 8)):
        """Plot portfolio results"""
        if self.portfolio_returns is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Multi-Pair Portfolio Analysis', fontsize=14)
        
        # Cumulative returns
        self.portfolio_cumulative.plot(ax=axes[0,0], title='Cumulative Returns')
        axes[0,0].grid(True, alpha=0.3)
        
        # Rolling Sharpe
        rolling_sharpe = (self.portfolio_returns.rolling(252).mean() * 252) / (self.portfolio_returns.rolling(252).std() * np.sqrt(252))
        rolling_sharpe.plot(ax=axes[0,1], title='Rolling 1Y Sharpe Ratio')
        axes[0,1].grid(True, alpha=0.3)
        
        # Drawdown
        rolling_max = self.portfolio_cumulative.expanding().max()
        drawdown = (self.portfolio_cumulative - rolling_max) / rolling_max
        drawdown.plot(ax=axes[1,0], title='Drawdown', color='red', area=True, alpha=0.3)
        axes[1,0].grid(True, alpha=0.3)
        
        # Return distribution
        self.portfolio_returns.hist(ax=axes[1,1], bins=50, title='Return Distribution', alpha=0.7)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# US SECTOR ETFs - Perfect for pairs trading
sector_etfs = [
    "XLK",   # Technology Select Sector SPDR Fund
    "XLF",   # Financial Select Sector SPDR Fund  
    "XLE",   # Energy Select Sector SPDR Fund
    "XLV",   # Health Care Select Sector SPDR Fund
    "XLI",   # Industrial Select Sector SPDR Fund
    "XLP",   # Consumer Staples Select Sector SPDR Fund
    "XLY",   # Consumer Discretionary Select Sector SPDR Fund
    "XLB",   # Materials Select Sector SPDR Fund
    "XLRE",  # Real Estate Select Sector SPDR Fund
    "XLU",   # Utilities Select Sector SPDR Fund
    "XLC",   # Communication Services Select Sector SPDR Fund
    # Additional broad market ETFs for more pairs
    "SPY",   # SPDR S&P 500 ETF Trust
    "QQQ",   # Invesco QQQ Trust (Nasdaq-100)
    "IWM",   # iShares Russell 2000 ETF
    "EFA",   # iShares MSCI EAFE ETF (International)
    "EEM",   # iShares MSCI Emerging Markets ETF
    "VNQ",   # Vanguard Real Estate ETF
    "GLD",   # SPDR Gold Shares
    "TLT",   # iShares 20+ Year Treasury Bond ETF
    "HYG"    # iShares iBoxx $ High Yield Corporate Bond ETF
]

# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("SECTOR ETF PAIRS TRADING STRATEGY")
    print("=" * 70)
    
    # Multi-pair strategy with sector ETFs
    multi_strategy = MultiPairsTradingStrategy(
        tickers=sector_etfs, 
        min_data_threshold=0.8
    )
    
    # Download data
    multi_strategy.download_data()
    
    # Find tradeable pairs
    pairs = multi_strategy.find_tradeable_pairs()
    
    if pairs:
        # Backtest portfolio
        multi_performance = multi_strategy.backtest_portfolio()
        
        # Plot results
        if multi_performance:
            multi_strategy.plot_results()
    
    print("\n" + "=" * 70)
    print("SINGLE-PAIR CONSERVATIVE STRATEGY (COMPARISON)")
    print("=" * 70)
    
    # Conservative single-pair strategy for comparison
    single_strategy = PairsTradingStrategy(
        tickers=sector_etfs,
        min_data_threshold=0.8
    )
    
    # Run analysis
    single_strategy.download_data()
    single_strategy.find_cointegrated_pairs(significance_level=0.05)
    
    if single_strategy.coint_results is not None and not single_strategy.coint_results.empty:
        single_strategy.analyze_best_pair(entry_threshold=1.0, exit_threshold=0.3)
        single_performance = single_strategy.backtest_strategy(max_position_size=0.1)
        
        # Display comparison
        print(f"\n=== STRATEGY COMPARISON ===")
        if 'multi_performance' in locals() and multi_performance:
            print(f"Multi-pair Sharpe: {multi_performance['sharpe_ratio']:.2f}")
            print(f"Single-pair Sharpe: {single_performance['sharpe_ratio']:.2f}")
            print(f"Multi-pair Max DD: {multi_performance['max_drawdown']:.2%}")
            print(f"Single-pair Max DD: {single_performance['max_drawdown']:.2%}")
        
        # Show top pairs with sector descriptions
        print(f"\n=== TOP COINTEGRATED SECTOR PAIRS ===")
        top_pairs = single_strategy.coint_results.head(10).copy()
        
        # Add sector descriptions
        sector_names = {
            'XLK': 'Technology', 'XLF': 'Financials', 'XLE': 'Energy', 'XLV': 'Healthcare',
            'XLI': 'Industrials', 'XLP': 'Staples', 'XLY': 'Discretionary', 'XLB': 'Materials',
            'XLRE': 'Real Estate', 'XLU': 'Utilities', 'XLC': 'Communications',
            'SPY': 'S&P 500', 'QQQ': 'Nasdaq-100', 'IWM': 'Small Cap', 'EFA': 'International',
            'EEM': 'Emerging Markets', 'VNQ': 'REITs', 'GLD': 'Gold', 'TLT': 'Long Bonds', 'HYG': 'High Yield'
        }
        
        def format_pair(pair_tuple):
            ticker1, ticker2 = pair_tuple
            name1 = sector_names.get(ticker1, ticker1)
            name2 = sector_names.get(ticker2, ticker2)
            return f"{ticker1}({name1}) - {ticker2}({name2})"
        
        top_pairs['Pair_Description'] = top_pairs['Pair'].apply(format_pair)
        print(top_pairs[['Pair_Description', 'p-value', 'Correlation']].to_string(index=False))
    else:
        print("No cointegrated pairs found for single-pair strategy")
