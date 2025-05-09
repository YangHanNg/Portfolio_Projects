import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from collections import deque
import os
from tabulate import tabulate  
import numba as nb  
import multiprocessing as mp  
import pandas_ta as ta 

# --------------------------------------------------------------------------------------------------------------------------

# Base parameters
TICKER = ['SPY']
DEFAULT_LOOKBACK = 4500

# Moving average strategy parameters
FAST = 20
SLOW = 50

# MACD strategy parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SPAN = 9

# RSI strategy parameters
RSI_LENGTH = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# MFI strategy parameters
MFI_LENGTH = 14
MFI_OVERBOUGHT = 70
MFI_OVERSOLD = 30

# Bollinger Bands strategy parameters
BB_LEN = 20
DEVS = 2

# Risk & Reward
DEFAULT_RISK = 0.03  # 2% risk per trade
DEFAULT_REWARD = 0.05  # 4% target profit per trade
DEFAULT_POSITION_SIZE = 0.01  # 10% of portfolio per trade
MAX_OPEN_POSITIONS = 100  # Maximum number of concurrent open positions

# --------------------------------------------------------------------------------------------------------------------------

def optimize_parameters(ticker='SPY', visualize_best=True, optimization_type='basic'):
    """
    Optimize strategy parameters using parallel processing
    
    Parameters:
    ticker (str): Ticker symbol to test
    visualize_best (bool): Whether to create charts for the best parameter set
    optimization_type (str): Type of optimization to run
                            'basic': Just risk/reward/position size
                            'technical': Technical indicator parameters
                            'comprehensive': Both basic and technical parameters
    
    Returns:
    tuple: (best_params, all_results)
    """
    
    print(f"Starting parameter optimization for {ticker}...")
    
    if optimization_type == 'basic':
        # Basic parameter grid - risk management only
        risks = [0.01, 0.02, 0.03, 0.04, 0.05]
        rewards = [0.02, 0.03, 0.05, 0.07, 0.10]
        position_sizes = [0.02, 0.05, 0.10, 0.15, 0.20]
        
        # Fixed technical parameters
        fast_periods = [FAST]  # Default value
        slow_periods = [SLOW]  # Default value
        rsi_lower_thresholds = [30]  # Default value
        rsi_upper_thresholds = [70]  # Default value
        bb_deviations = [2.0]  # Default value
        adx_thresholds = [25]  # Default value
        min_buy_scores = [3.0]  # Default value
        min_sell_scores = [3.0]  # Default value
        require_cloud_contexts = [True]  # Default value
        
    elif optimization_type == 'technical':
        # Fixed risk parameters (use defaults)
        risks = [DEFAULT_RISK]
        rewards = [DEFAULT_REWARD]
        position_sizes = [DEFAULT_POSITION_SIZE]
        
        # Technical parameter grid
        fast_periods = [10, 15, 20, 25]
        slow_periods = [40, 50, 60]
        rsi_lower_thresholds = [25, 30, 35]
        rsi_upper_thresholds = [65, 70, 75]
        bb_deviations = [1.8, 2.0, 2.2]
        adx_thresholds = [20, 25, 30]
        min_buy_scores = [2.5, 3.0, 3.5]
        min_sell_scores = [2.5, 3.0, 3.5]
        require_cloud_contexts = [True, False]
        
    else:  # comprehensive optimization
        # Full parameter grid - warning: this can be very computationally intensive!
        # Consider running this on a powerful machine or cloud instance
        risks = [0.02, 0.03, 0.04]
        rewards = [0.03, 0.05, 0.07]
        position_sizes = [0.05, 0.10, 0.15]
        
        fast_periods = [15, 20, 25]
        slow_periods = [40, 50, 60]
        rsi_lower_thresholds = [25, 30, 35]
        rsi_upper_thresholds = [65, 70, 75]
        bb_deviations = [1.8, 2.0, 2.2]
        adx_thresholds = [20, 25, 30]
        min_buy_scores = [2.5, 3.0, 3.5]
        min_sell_scores = [2.5, 3.0, 3.5]
        require_cloud_contexts = [True, False]
    
    # Create parameter combinations - this can grow very large, be careful
    param_grid = []
    for risk in risks:
        for reward in rewards:
            for position_size in position_sizes:
                for fast in fast_periods:
                    for slow in slow_periods:
                        if fast >= slow:  # Skip invalid combinations
                            continue
                        for rsi_lower in rsi_lower_thresholds:
                            for rsi_upper in rsi_upper_thresholds:
                                if rsi_lower >= rsi_upper:  # Skip invalid combinations
                                    continue
                                for bb_dev in bb_deviations:
                                    for adx_threshold in adx_thresholds:
                                        for min_buy_score in min_buy_scores:
                                            for min_sell_score in min_sell_scores:
                                                for require_cloud in require_cloud_contexts:
                                                    # Technical parameter set
                                                    tech_params = {
                                                        'FAST': fast,
                                                        'SLOW': slow,
                                                        'RSI_OVERSOLD': rsi_lower,
                                                        'RSI_OVERBOUGHT': rsi_upper,
                                                        'DEVS': bb_dev,
                                                        'ADX_THRESHOLD': adx_threshold,
                                                        'MIN_BUY_SCORE': min_buy_score,
                                                        'MIN_SELL_SCORE': min_sell_score,
                                                        'REQUIRE_CLOUD': require_cloud
                                                    }
                                                    
                                                    # Create charts only for the final best parameter set
                                                    create_charts = False
                                                    
                                                    param_grid.append((
                                                        ticker, risk, reward, position_size, 
                                                        create_charts, tech_params
                                                    ))
    
    # Use multiprocessing to test parameters in parallel
    start_time = datetime.now()
    print(f"Testing {len(param_grid)} parameter combinations using multiprocessing...")
    
    # Calculate estimated time based on number of combinations and cores
    seconds_per_test = 3  # Estimated time per test in seconds
    total_cores = max(1, mp.cpu_count() - 1)  # Leave one core for system
    estimated_seconds = len(param_grid) * seconds_per_test / total_cores
    estimated_time = timedelta(seconds=estimated_seconds)
    
    print(f"Estimated completion time: {estimated_time} "
          f"(using {total_cores} cores, ~{seconds_per_test}s per test)")
    
    results = []
    with mp.Pool(processes=total_cores) as pool:
        results = pool.map(run_parameter_test, param_grid)
    
    # Find the best parameter set based on Sharpe ratio
    best_result = max(results, key=lambda x: x['sharpe'])
    
    # Sort results by Sharpe ratio
    sorted_results = sorted(results, key=lambda x: x['sharpe'], reverse=True)
    
    # Display top 5 parameter combinations
    print("\nTop 5 Parameter Combinations:")
    headers = ["Risk", "Reward", "Pos Size", "Fast", "Slow", "RSI Low", "RSI High", "BB Dev",
              "ADX Thres", "Min Buy", "Min Sell", "Cloud Req", 
              "Win%", "Profit%", "MaxDD%", "Sharpe", "# Trades"]
    rows = []
    
    for i, result in enumerate(sorted_results[:5]):
        rows.append([
            f"{result['risk']*100:.1f}%", 
            f"{result['reward']*100:.1f}%", 
            f"{result['position_size']*100:.1f}%",
            f"{result['tech_params']['FAST']}",
            f"{result['tech_params']['SLOW']}",
            f"{result['tech_params']['RSI_OVERSOLD']}",
            f"{result['tech_params']['RSI_OVERBOUGHT']}",
            f"{result['tech_params']['DEVS']:.1f}",
            f"{result['tech_params']['ADX_THRESHOLD']}",
            f"{result['tech_params']['MIN_BUY_SCORE']:.1f}",
            f"{result['tech_params']['MIN_SELL_SCORE']:.1f}",
            f"{result['tech_params']['REQUIRE_CLOUD']}",
            f"{result['win_rate']:.1f}", 
            f"{result['net_profit_pct']:.1f}", 
            f"{result['max_drawdown']:.1f}", 
            f"{result['sharpe']:.2f}", 
            f"{result['num_trades']}"
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # If requested, run again with the best parameters and visualize
    if visualize_best:
        print(f"\nRunning detailed backtest with best parameters:")
        print(f"Risk: {best_result['risk']*100:.1f}%, Reward: {best_result['reward']*100:.1f}%, "
              f"Position Size: {best_result['position_size']*100:.1f}%")
        print(f"Technical Parameters: Fast={best_result['tech_params']['FAST']}, "
              f"Slow={best_result['tech_params']['SLOW']}, "
              f"RSI={best_result['tech_params']['RSI_OVERSOLD']}/{best_result['tech_params']['RSI_OVERBOUGHT']}, "
              f"BB Dev={best_result['tech_params']['DEVS']:.1f}, "
              f"ADX Threshold={best_result['tech_params']['ADX_THRESHOLD']}, "
              f"Min Scores: Buy {best_result['tech_params']['MIN_BUY_SCORE']:.1f}/Sell {best_result['tech_params']['MIN_SELL_SCORE']:.1f}")
        
        # Download data
        df = yf.download(ticker, period="5y")
        
        # Get the best technical parameters
        best_fast = best_result['tech_params']['FAST']
        best_slow = best_result['tech_params']['SLOW']
        best_rsi_oversold = best_result['tech_params']['RSI_OVERSOLD']
        best_rsi_overbought = best_result['tech_params']['RSI_OVERBOUGHT']
        best_devs = best_result['tech_params']['DEVS']
        
        # Reset indicator cache - force recalculation with new parameters
        cache_params = f"_{best_fast}_{best_slow}_{best_rsi_oversold}_{best_rsi_overbought}_{best_devs}"
        cache_file = f"{ticker}_indicators{cache_params}_cache.pkl"
        
        # Pass params explicitly instead of modifying globals
        df_with_indicators = adding_indicators(
            df,
            fast=best_fast,
            slow=best_slow,
            rsi_oversold=best_rsi_oversold,
            rsi_overbought=best_rsi_overbought,
            devs=best_devs,
            recalculate=True,
            cache_file=cache_file
        )
        
        # Run backtest with best parameters
        result = test_strategy(
            df_with_indicators,
            ticker,
            risk=best_result['risk'],
            reward=best_result['reward'],
            position_size=best_result['position_size']
        )
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"\nParameter optimization completed in {elapsed_time}")
    
    return best_result, sorted_results

# --------------------------------------------------------------------------------------------------------------------------
# Functions for all indicators
def adding_indicators(df, fast=None, slow=None, rsi_oversold=None, rsi_overbought=None, devs=None, 
                     recalculate=False, cache_file="indicator_cache.pkl"):
    """
    Calculate technical indicators with vectorized operations for better performance
    
    Parameters:
    df: DataFrame with OHLCV data
    fast: Fast period for moving average (default=None, uses global FAST)
    slow: Slow period for moving average (default=None, uses global SLOW)
    rsi_oversold: RSI oversold threshold (default=None, uses global RSI_OVERSOLD)
    rsi_overbought: RSI overbought threshold (default=None, uses global RSI_OVERBOUGHT) 
    devs: Bollinger Bands standard deviation (default=None, uses global DEVS)
    recalculate: Force recalculation even if cache exists
    cache_file: File to cache indicators to avoid recalculation
    
    Returns:
    DataFrame with added indicators
    """
    # Use provided parameters or fall back to global defaults
    fast = fast if fast is not None else FAST
    slow = slow if slow is not None else SLOW
    rsi_oversold = rsi_oversold if rsi_oversold is not None else RSI_OVERSOLD
    rsi_overbought = rsi_overbought if rsi_overbought is not None else RSI_OVERBOUGHT
    devs = devs if devs is not None else DEVS
    
    # Use caching to avoid recalculating indicators
    if cache_file and os.path.exists(cache_file) and not recalculate:
        try:
            return pd.read_pickle(cache_file)
        except Exception:
            pass  # If error reading cache, recalculate
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Check if the dataframe is empty
    if len(df) == 0:
        print("Warning: Empty dataframe provided to adding_indicators!")
        return df
    
    # --- Efficient vectorized calculations ---
    
    # Moving Averages - vectorized
    df.loc[:, f'{fast}_ma'] = df['Close'].rolling(fast).mean()
    df.loc[:, f'{slow}_ma'] = df['Close'].rolling(slow).mean()

    # MACD - calculated manually to avoid potential None return from pandas_ta
    df.loc[:, 'EMA_fast'] = df['Close'].ewm(span=MACD_FAST, adjust=False).mean()
    df.loc[:, 'EMA_slow'] = df['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df.loc[:, 'MACD'] = df['EMA_fast'] - df['EMA_slow']
    df.loc[:, 'Signal'] = df['MACD'].ewm(span=MACD_SPAN, adjust=False).mean()
    df.loc[:, 'MACD_hist'] = df['MACD'] - df['Signal']

    # RSI - vectorized
    rsi = ta.rsi(df['Close'], length=RSI_LENGTH)
    df.loc[:, 'RSI'] = rsi if rsi is not None else df['Close'].pct_change().rolling(RSI_LENGTH).apply(lambda x: 100 * (sum(max(y, 0) for y in x) / sum(abs(y) for y in x) if sum(abs(y) for y in x) != 0 else 50))

    # MFI - vectorized
    mfi = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=MFI_LENGTH)
    df.loc[:, 'MFI'] = mfi if mfi is not None else df['Close'].rolling(MFI_LENGTH).mean()  # Simplified fallback

    # Bollinger Bands - manually calculate if pandas_ta returns None
    bbands = ta.bbands(df['Close'], length=BB_LEN, std=devs)
    if bbands is not None:
        try:
            df.loc[:, 'BB_SMA'] = bbands['BBM_' + str(BB_LEN) + '_' + str(float(devs))]
            df.loc[:, 'Upper_Band'] = bbands['BBU_' + str(BB_LEN) + '_' + str(float(devs))]
            df.loc[:, 'Lower_Band'] = bbands['BBL_' + str(BB_LEN) + '_' + str(float(devs))]
        except (KeyError, TypeError):
            # Calculate manually
            bb_sma = df['Close'].rolling(window=BB_LEN).mean()
            bb_std = df['Close'].rolling(window=BB_LEN).std()
            df.loc[:, 'BB_SMA'] = bb_sma
            df.loc[:, 'Upper_Band'] = bb_sma + (bb_std * devs)
            df.loc[:, 'Lower_Band'] = bb_sma - (bb_std * devs)
    else:
        # Calculate manually
        bb_sma = df['Close'].rolling(window=BB_LEN).mean()
        bb_std = df['Close'].rolling(window=BB_LEN).std()
        df.loc[:, 'BB_SMA'] = bb_sma
        df.loc[:, 'Upper_Band'] = bb_sma + (bb_std * devs)
        df.loc[:, 'Lower_Band'] = bb_sma - (bb_std * devs)

    # ATR - vectorized
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df.loc[:, 'ATR'] = atr if atr is not None else (df['High'] - df['Low']).rolling(14).mean()  # Simplified fallback

    # ADX - vectorized
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    if adx is not None and 'ADX_14' in adx:
        df.loc[:, 'ADX'] = adx['ADX_14']
    else:
        # Calculate ADX manually if pandas_ta fails
        df.loc[:, 'ADX'] = df['High'].rolling(14).mean()  # Simplified fallback

    # Parabolic SAR - vectorized with error handling
    try:
        psar = ta.psar(df['High'], df['Low'], df['Close'])
        if psar is not None and 'PSARl_0.02_0.2' in psar:
            df.loc[:, 'SAR'] = psar['PSARl_0.02_0.2']
        else:
            df.loc[:, 'SAR'] = df['Close'].rolling(10).mean() * 0.98  # Simplified fallback
    except Exception:
        df.loc[:, 'SAR'] = df['Close'].rolling(10).mean() * 0.98  # Simplified fallback

    # ROC - vectorized
    roc = ta.roc(df['Close'], length=12)
    df.loc[:, 'ROC'] = roc if roc is not None else df['Close'].pct_change(12)

    # Ichimoku Cloud - vectorized with error handling
    try:
        ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])
        if ichimoku is not None:
            df.loc[:, 'Tenkan_sen'] = ichimoku.get('ITS_9', df['Close'].rolling(9).mean())  # Conversion line
            df.loc[:, 'Kijun_sen'] = ichimoku.get('IKS_26', df['Close'].rolling(26).mean())  # Base line
            df.loc[:, 'Senkou_span_A'] = ichimoku.get('ISA_9', df['Close'].rolling(9).mean())  # Leading span A
            df.loc[:, 'Senkou_span_B'] = ichimoku.get('ISB_26', df['Close'].rolling(26).mean())  # Leading span B
        else:
            # Simplified fallback calculations
            df.loc[:, 'Tenkan_sen'] = df['Close'].rolling(9).mean()
            df.loc[:, 'Kijun_sen'] = df['Close'].rolling(26).mean()
            df.loc[:, 'Senkou_span_A'] = df['Close'].rolling(9).mean()
            df.loc[:, 'Senkou_span_B'] = df['Close'].rolling(26).mean()
    except Exception:
        # Simplified fallback calculations
        df.loc[:, 'Tenkan_sen'] = df['Close'].rolling(9).mean()
        df.loc[:, 'Kijun_sen'] = df['Close'].rolling(26).mean()
        df.loc[:, 'Senkou_span_A'] = df['Close'].rolling(9).mean()
        df.loc[:, 'Senkou_span_B'] = df['Close'].rolling(26).mean()
    
    df.loc[:, 'Chikou_span'] = df['Close'].shift(-26)  # Lagging span

    # Save to cache if requested
    if cache_file:
        df.to_pickle(cache_file)
    
    return df.dropna()

# --------------------------------------------------------------------------------------------------------------------------

def test_strategy(df, ticker, risk=DEFAULT_RISK, reward=DEFAULT_REWARD, position_size=DEFAULT_POSITION_SIZE):
    
    # Run backtest with enhanced trade tracking
    trade_log, trade_stats, portfolio_equity, trade_returns = set_momentum_strat(df, risk=DEFAULT_RISK, reward=DEFAULT_REWARD, position_size=DEFAULT_POSITION_SIZE, 
                          max_positions=MAX_OPEN_POSITIONS, risk_free_rate=0.04)
    
    df = df.copy()
    # Add asset returns to dataframe for comparison
    df.loc[:, 'Asset_Returns'] = df['Close'].pct_change().fillna(0).cumsum()
    
    # Convert equity curve to returns for comparison
    df.loc[:, 'Strategy_Returns'] = (portfolio_equity / portfolio_equity.iloc[0] - 1)
    
    # Calculate additional metrics
    buy_hold_return = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
    peak_equity = portfolio_equity.max()
    exposure_time = ((df.index[-1] - df.index[0]).days)
    
    # Find the best and worst trades
    if trade_log:
        best_trade_pct = max([t['PnL'] for t in trade_log if t['PnL'] is not None]) / portfolio_equity.iloc[0] * 100 if trade_log else 0
        worst_trade_pct = min([t['PnL'] for t in trade_log if t['PnL'] is not None]) / portfolio_equity.iloc[0] * 100 if trade_log else 0
        avg_trade_pct = sum([t['PnL'] for t in trade_log if t['PnL'] is not None]) / len(trade_log) / portfolio_equity.iloc[0] * 100 if trade_log else 0
        max_duration = max([t['Duration'] for t in trade_log if t['Duration'] is not None]) if trade_log else 0
        avg_duration = sum([t['Duration'] for t in trade_log if t['Duration'] is not None]) / len(trade_log) if trade_log else 0
    else:
        best_trade_pct = worst_trade_pct = avg_trade_pct = max_duration = avg_duration = 0

    # Print summary stats with right alignment
    print(f"\n=== {ticker} STRATEGY SUMMARY ===")
    print(f"Risk: {risk*100:.1f}% | Reward: {reward*100:.1f}% | Position Size: {position_size*100:.1f}%")
    
    # Format financial metrics with tabulate
    metrics = [
        ["Starting Capital [$]", f"{portfolio_equity.iloc[0]:,.2f}"],
        ["Ending Capital [$]", f"{portfolio_equity.iloc[-1]:,.2f}"],
        ["Start", f"{df.index[0].strftime('%Y-%m-%d')}"],
        ["End", f"{df.index[-1].strftime('%Y-%m-%d')}"],
        ["Duration [days]", f"{exposure_time}"],
        ["Equity Final [$]", f"{portfolio_equity.iloc[-1]:,.2f}"],
        ["Equity Peak [$]", f"{peak_equity:,.2f}"],
        ["Return [%]", f"{trade_stats['Net Profit (%)']:.2f}"],
        ["Buy & Hold Return [%]", f"{buy_hold_return:.2f}"],
        ["Annual Return [%]", f"{trade_stats['Annualized Return (%)']:.2f}"],
        ["Sharpe Ratio", f"{trade_stats['Sharpe Ratio']:.2f}"],
        ["Sortino Ratio", f"{trade_stats['Sortino Ratio']:.2f}"],
        ["Max. Drawdown [%]", f"{trade_stats['Max Drawdown (%)']:.2f}"],
    ]
    
    print(tabulate(metrics, tablefmt="simple", colalign=("left", "right")))
    
    # Print trade summary with tabulate
    print(f"\n=== {ticker} TRADE SUMMARY ===")
    trade_metrics = [
        ["Win Rate [%]", f"{trade_stats['Win Rate']:.2f}"],
        ["Best Trade [%]", f"{best_trade_pct:.2f}"],
        ["Worst Trade [%]", f"{worst_trade_pct:.2f}"],
        ["Avg. Trade [%]", f"{avg_trade_pct:.2f}"],
        ["Max. Trade Duration [days]", f"{max_duration}"],
        ["Avg. Trade Duration [days]", f"{avg_duration:.1f}"],
        ["Profit Factor", f"{trade_stats['Profit Factor']:.2f}"],
        ["Expectancy [%]", f"{trade_stats['Expectancy (%)']:.2f}"]
    ]
    
    print(tabulate(trade_metrics, tablefmt="simple", colalign=("left", "right")))
    
    # Create visualizations
    # create_backtest_charts(df, portfolio_equity, trade_stats, trade_log, ticker)
    
    return {
        'ticker': ticker,
        'df': df,
        'equity': portfolio_equity,
        'trade_stats': trade_stats,
        'trade_log': trade_log
    }

# --------------------------------------------------------------------------------------------------------------------------

def set_momentum_strat(df, risk=DEFAULT_RISK, reward=DEFAULT_REWARD, position_size=DEFAULT_POSITION_SIZE, 
                      max_positions=MAX_OPEN_POSITIONS, risk_free_rate=0.04):
    """
    Execute the momentum strategy on the given DataFrame
    
    Uses optimized data structures and vectorized operations where possible
    """
    # Check if DataFrame is empty after preprocessing (e.g. after dropna())
    if len(df) == 0:
        print("Warning: Empty dataframe provided to set_momentum_strat. Returning empty results.")
        empty_equity = pd.Series(dtype='float64')
        empty_returns = pd.Series(dtype='float64')
        return [], {
            'Total Trades': 0,
            'Win Rate': 0,
            'Net Profit (%)': 0,
            'Profit Factor': 0,
            'Expectancy (%)': 0,
            'Max Drawdown (%)': 0,
            'Annualized Return (%)': 0,
            'Sharpe Ratio': 0,
            'Sortino Ratio': 0
        }, empty_equity, empty_returns
    
    # Initialize tracking variables
    initial_capital = 100000.0  # Add .0 to explicitly make this a float
    available_capital = initial_capital
    portfolio_value = initial_capital

    # Create equity curve series with same index as df and explicit float64 dtype
    equity = pd.Series(initial_capital, index=df.index, dtype='float64')
    trade_returns = pd.Series(0.0, index=df.index)
    
    # Track active trades with DataFrames instead of deques for better performance
    active_long_trades = pd.DataFrame(columns=['entry_date', 'multiplier', 'entry_price', 
                                             'stop_loss', 'take_profit', 'position_size', 'share_amount'])
    active_short_trades = pd.DataFrame(columns=['entry_date', 'multiplier', 'entry_price', 
                                              'stop_loss', 'take_profit', 'position_size', 'share_amount'])
    
    long_positions, short_positions = 0, 0

    # Track trade statistics
    trade_log = []
    wins, losses, lengths = [], [], []

    # Main loop through data
    for i in range(1, len(df)):  # Start from 1 to avoid index error on i-1
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        
        # Default equity to previous day (will be updated if trades exit)
        equity.iloc[i] = equity.iloc[i-1]
        
        # Parameters for numba function from current bar
        params = (
            df[f'{FAST}_ma'].iloc[i], df[f'{SLOW}_ma'].iloc[i], df['RSI'].iloc[i],
            df['MFI'].iloc[i], df['Close'].iloc[i], df['Lower_Band'].iloc[i],
            df['Upper_Band'].iloc[i], df['ATR'].iloc[i], df['ADX'].iloc[i],
            df['SAR'].iloc[i], df['ROC'].iloc[i], df['Tenkan_sen'].iloc[i],
            df['Kijun_sen'].iloc[i], df['Senkou_span_A'].iloc[i], 
            df['Senkou_span_B'].iloc[i], df['Close'].iloc[i-26] if i >= 26 else None
        )
        buy_signal, sell_signal, buy_score, sell_score = calculate_signals_numba(*params)
        
        
        # --- Exit short positions on buy signal ---
        if buy_signal and not active_short_trades.empty:
            for idx, trade in active_short_trades.iterrows():
                portfolio_value, available_capital, trade_log, wins, losses, lengths, trade_returns = handle_trade_exit(
                    trade['entry_date'], current_date, 
                    (trade['multiplier'], trade['entry_price'], trade['stop_loss'], 
                     trade['take_profit'], trade['position_size'], trade['share_amount']),
                    'Short', current_price, portfolio_value, available_capital, 
                    trade_log, wins, losses, lengths, trade_returns, 'Buy Signal'
                )
            
            active_short_trades = pd.DataFrame(columns=active_short_trades.columns)  # Clear all shorts
            short_positions = 0
            equity.iloc[i] = portfolio_value  # Update equity after exits

        # --- Exit long positions on sell signal ---
        if sell_signal and not active_long_trades.empty:
            for idx, trade in active_long_trades.iterrows():
                portfolio_value, available_capital, trade_log, wins, losses, lengths, trade_returns = handle_trade_exit(
                    trade['entry_date'], current_date, 
                    (trade['multiplier'], trade['entry_price'], trade['stop_loss'], 
                     trade['take_profit'], trade['position_size'], trade['share_amount']),
                    'Long', current_price, portfolio_value, available_capital, 
                    trade_log, wins, losses, lengths, trade_returns, 'Sell Signal'
                )
            
            active_long_trades = pd.DataFrame(columns=active_long_trades.columns)  # Clear all longs
            long_positions = 0
            equity.iloc[i] = portfolio_value  # Update equity after exits

            # --- Enter short if allowed ---
            if long_positions == 0 and short_positions < max_positions and available_capital > 0:
                trade_data = prepare_trade_entry('Short', current_price, df['ATR'].iloc[i], 
                                               portfolio_value, available_capital, position_size, risk, reward)
                if trade_data:
                    (entry_price, stop_loss, take_profit, trade_position_size, share_amount) = trade_data
                    
                    # Add trade to tracking DataFrame
                    new_trade = pd.DataFrame({
                        'entry_date': [current_date],
                        'multiplier': [-1],
                        'entry_price': [entry_price],
                        'stop_loss': [stop_loss],
                        'take_profit': [take_profit],
                        'position_size': [trade_position_size],
                        'share_amount': [share_amount]
                    })
                    active_short_trades = pd.concat([active_short_trades, new_trade], ignore_index=True)
                    
                    available_capital -= share_amount * entry_price
                    short_positions += 1
                    trade_log.append(create_trade_entry(current_date, 'Short', entry_price, trade_position_size, share_amount))

        # --- Check stop losses and take profits (Long) - vectorized approach ---
        if not active_long_trades.empty:
            # Calculate exit conditions for all trades at once
            take_profit_exits = ~active_long_trades['take_profit'].isna() & (current_price >= active_long_trades['take_profit'])
            stop_loss_exits = current_price <= active_long_trades['stop_loss']
            exits = take_profit_exits | stop_loss_exits
            
            if exits.any():
                # Process exiting trades
                exit_trades = active_long_trades[exits].copy()
                for idx, trade in exit_trades.iterrows():
                    exit_reason = 'Take Profit' if take_profit_exits.loc[idx] else 'Stop Loss'
                    
                    portfolio_value, available_capital, trade_log, wins, losses, lengths, trade_returns = handle_trade_exit(
                        trade['entry_date'], current_date, 
                        (trade['multiplier'], trade['entry_price'], trade['stop_loss'], 
                         trade['take_profit'], trade['position_size'], trade['share_amount']),
                        'Long', current_price, portfolio_value, available_capital, 
                        trade_log, wins, losses, lengths, trade_returns, exit_reason
                    )
                    long_positions -= 1
                
                # Keep only non-exiting trades
                active_long_trades = active_long_trades[~exits].reset_index(drop=True)
                equity.iloc[i] = portfolio_value  # Update equity after exits

        # --- Check stop losses and take profits (Short) - vectorized approach ---
        if not active_short_trades.empty:
            # Calculate exit conditions for all trades at once
            tactive_long_trades = active_long_trades.reset_index(drop=True)
            take_profit_exits = ~active_long_trades['take_profit'].isna() & (current_price >= active_long_trades['take_profit'])
            stop_loss_exits = current_price <= active_long_trades['stop_loss']
            exits = take_profit_exits | stop_loss_exits
            
            
            if exits.any():
                # Process exiting trades
                exit_trades = active_short_trades[exits].copy()
                for idx, trade in exit_trades.iterrows():
                    exit_reason = 'Take Profit' if take_profit_exits.loc[idx] else 'Stop Loss'
                    
                    portfolio_value, available_capital, trade_log, wins, losses, lengths, trade_returns = handle_trade_exit(
                        trade['entry_date'], current_date, 
                        (trade['multiplier'], trade['entry_price'], trade['stop_loss'], 
                         trade['take_profit'], trade['position_size'], trade['share_amount']),
                        'Short', current_price, portfolio_value, available_capital, 
                        trade_log, wins, losses, lengths, trade_returns, exit_reason
                    )
                    short_positions -= 1
                
                # Keep only non-exiting trades
                active_short_trades = active_short_trades[~exits].reset_index(drop=True)
                equity.iloc[i] = portfolio_value  # Update equity after exits

        # --- Enter long if buy signal ---
        if buy_signal and long_positions < max_positions and available_capital > 0:
            trade_data = prepare_trade_entry('Long', current_price, df['ATR'].iloc[i], 
                                           portfolio_value, available_capital, position_size, risk, reward)
            if trade_data:
                (entry_price, stop_loss, take_profit, trade_position_size, share_amount) = trade_data
                
                # Add trade to tracking DataFrame
                new_trade = pd.DataFrame({
                    'entry_date': [current_date],
                    'multiplier': [1],
                    'entry_price': [entry_price],
                    'stop_loss': [stop_loss],
                    'take_profit': [take_profit],
                    'position_size': [trade_position_size],
                    'share_amount': [share_amount]
                })
                active_long_trades = pd.concat([active_long_trades, new_trade], ignore_index=True)
                
                available_capital -= share_amount * entry_price
                long_positions += 1
                trade_log.append(create_trade_entry(current_date, 'Long', entry_price, trade_position_size, share_amount))

    # --- Final position exit ---
    # Check if DataFrame has any data before accessing the last row
    if len(df.index) > 0:
        final_date = df.index[-1]
        final_price = df['Close'].iloc[-1]

        # Close any remaining long positions
        if not active_long_trades.empty:
            for idx, trade in active_long_trades.iterrows():
                portfolio_value, available_capital, trade_log, wins, losses, lengths, trade_returns = handle_trade_exit(
                    trade['entry_date'], final_date, 
                    (trade['multiplier'], trade['entry_price'], trade['stop_loss'], 
                     trade['take_profit'], trade['position_size'], trade['share_amount']),
                    'Long', final_price, portfolio_value, available_capital, 
                    trade_log, wins, losses, lengths, trade_returns, 'End of Simulation'
                )

        # Close any remaining short positions
        if not active_short_trades.empty:
            for idx, trade in active_short_trades.iterrows():
                portfolio_value, available_capital, trade_log, wins, losses, lengths, trade_returns = handle_trade_exit(
                    trade['entry_date'], final_date, 
                    (trade['multiplier'], trade['entry_price'], trade['stop_loss'], 
                     trade['take_profit'], trade['position_size'], trade['share_amount']),
                    'Short', final_price, portfolio_value, available_capital, 
                    trade_log, wins, losses, lengths, trade_returns, 'End of Simulation'
                )

        # Update final equity value
        if len(equity) > 0:
            equity.iloc[-1] = portfolio_value

    # --- Compile trade stats ---
    trade_stats = compile_trade_statistics(equity, trade_log, wins, losses, risk_free_rate)

    return trade_log, trade_stats, equity, trade_returns

# --------------------------------------------------------------------------------------------------------------------------

def calculate_signals_numpy(fast_ma, slow_ma, rsi, mfi, close, lower_band, upper_band, atr, adx, sar, roc, tenkan, kijun, spanA, spanB, close_26_ago=None,
                           adx_threshold=25, min_buy_score=3.0, min_sell_score=3.0, require_cloud=True):
    """
    Signal calculation function using NumPy vectorization instead of Numba
    
    Parameters include optimizable thresholds:
    - adx_threshold: ADX level that indicates strong trend (default: 25)
    - min_buy_score: Minimum score threshold for buy signals (default: 3.0)
    - min_sell_score: Minimum score threshold for sell signals (default: 3.0)
    - require_cloud: Whether to require cloud context for signals (default: True)
    """
    # Convert inputs to numpy arrays if they're pandas Series
    inputs = [fast_ma, slow_ma, rsi, mfi, close, lower_band, upper_band, 
              atr, adx, sar, roc, tenkan, kijun, spanA, spanB]
    
    for i in range(len(inputs)):
        if hasattr(inputs[i], 'values'):
            inputs[i] = inputs[i].values
        if hasattr(inputs[i], 'item'):
            inputs[i] = inputs[i].item()
            
    # Unpack the converted inputs
    fast_ma, slow_ma, rsi, mfi, close, lower_band, upper_band, atr, adx, sar, roc, tenkan, kijun, spanA, spanB = inputs
    
    # Convert close_26_ago if it exists
    if close_26_ago is not None:
        if hasattr(close_26_ago, 'values'):
            close_26_ago = close_26_ago.values
        if hasattr(close_26_ago, 'item'):
            close_26_ago = close_26_ago.item()
    
    # Calculate all conditions
    # Note: For scalars these are simple booleans, for arrays these will be boolean arrays
    
    # Moving average crossover (needs special handling for the "and" operation)
    ma_cross_condition = fast_ma > slow_ma
    if hasattr(ma_cross_condition, '__len__') and len(ma_cross_condition) > 1:
        # For Series/arrays, we need to compare with shifted values
        ma_buy = np.logical_and(ma_cross_condition, np.logical_not(np.roll(ma_cross_condition, 1)))
    else:
        # For scalar values, just check the current condition 
        # (simplified as we can't check previous without context)
        ma_buy = ma_cross_condition
    
    ma_sell = np.logical_not(ma_cross_condition)
    if hasattr(ma_sell, '__len__') and len(ma_sell) > 1:
        ma_sell = np.logical_and(ma_sell, np.logical_not(np.roll(ma_sell, 1)))
    
    # RSI signals
    rsi_buy = rsi < 30
    rsi_sell = rsi > 70
    
    # Bollinger Bands
    bb_buy = close < lower_band
    bb_sell = close > upper_band
    
    # MFI
    mfi_buy = mfi < 20
    mfi_sell = mfi > 80
    
    # ADX trend strength
    strong_trend = adx > adx_threshold
    
    # Parabolic SAR
    sar_buy = close > sar
    sar_sell = close < sar
    
    # Rate of Change momentum
    roc_buy = roc > 0
    roc_sell = roc < 0
    
    # Ichimoku Cloud
    above_cloud = close > np.maximum(spanA, spanB)
    below_cloud = close < np.minimum(spanA, spanB)
    tenkan_kijun_buy = tenkan > kijun
    tenkan_kijun_sell = tenkan < kijun
    price_kijun_buy = close > kijun
    price_kijun_sell = close < kijun
    
    # Chikou span
    if close_26_ago is not None:
        chikou_buy = close > close_26_ago
        chikou_sell = close < close_26_ago
    else:
        chikou_buy = False
        chikou_sell = False
    
    # Calculate signal scores using vectorized operations
    buy_score = np.zeros_like(close, dtype=float) if hasattr(close, '__len__') else 0.0
    sell_score = np.zeros_like(close, dtype=float) if hasattr(close, '__len__') else 0.0
    
    # Add oscillator signals (work in any market)
    buy_score += np.where(rsi_buy, 1.0, 0.0)
    buy_score += np.where(bb_buy, 1.0, 0.0)
    buy_score += np.where(mfi_buy, 1.0, 0.0)
    
    sell_score += np.where(rsi_sell, 1.0, 0.0)
    sell_score += np.where(bb_sell, 1.0, 0.0)
    sell_score += np.where(mfi_sell, 1.0, 0.0)
    
    # Add trend signals, but only when trend is strong
    trend_buy_score = (
        np.where(ma_buy, 1.5, 0.0) +
        np.where(sar_buy, 1.25, 0.0) +
        np.where(roc_buy, 1.25, 0.0) +
        np.where(above_cloud, 1.5, 0.0) +
        np.where(tenkan_kijun_buy, 1.5, 0.0) +
        np.where(price_kijun_buy, 1.0, 0.0) +
        np.where(chikou_buy, 0.5, 0.0)
    )
    
    trend_sell_score = (
        np.where(ma_sell, 1.5, 0.0) +
        np.where(sar_sell, 1.25, 0.0) +
        np.where(roc_sell, 1.25, 0.0) +
        np.where(below_cloud, 1.5, 0.0) +
        np.where(tenkan_kijun_sell, 1.5, 0.0) +
        np.where(price_kijun_sell, 1.0, 0.0) +
        np.where(chikou_sell, 0.5, 0.0)
    )
    
    # Only add trend signals when trend is strong
    buy_score += np.where(strong_trend, trend_buy_score, 0.0)
    sell_score += np.where(strong_trend, trend_sell_score, 0.0)
    
    # Final signals with threshold parameters and cloud context requirement
    if require_cloud:
        buy_signal = np.logical_and(buy_score >= min_buy_score, 
                                   np.logical_or(strong_trend, above_cloud))
        sell_signal = np.logical_and(sell_score >= min_sell_score,
                                    np.logical_or(strong_trend, below_cloud))
    else:
        buy_signal = np.logical_and(buy_score >= min_buy_score, strong_trend)
        sell_signal = np.logical_and(sell_score >= min_sell_score, strong_trend)
    
    return buy_signal, sell_signal, buy_score, sell_score

# Keep the original function but rename it for backward compatibility
def calculate_signals_numba(fast_ma, slow_ma, rsi, mfi, close, lower_band, upper_band, atr, adx, sar, roc, tenkan, kijun, spanA, spanB, close_26_ago=None,
                           adx_threshold=25, min_buy_score=3.0, min_sell_score=3.0, require_cloud=True):
    """
    Signal calculation function - uses numpy implementation for better pandas Series compatibility
    """
    return calculate_signals_numpy(fast_ma, slow_ma, rsi, mfi, close, lower_band, upper_band, atr, adx, sar, roc, tenkan, kijun, spanA, spanB, close_26_ago,
                                  adx_threshold, min_buy_score, min_sell_score, require_cloud)

# --------------------------------------------------------------------------------------------------------------------------

def handle_trade_exit(entry_date, exit_date, trade_info, direction, exit_price,
                      portfolio_value, available_capital, trade_log, wins, losses, lengths, trade_returns, reason):
    # Unpack trade info
    if len(trade_info) < 5:  # Minimum required elements
        print(f"Error: Insufficient trade_info length. Expected at least 5, got {len(trade_info)}.")
        return portfolio_value, available_capital, trade_log, wins, losses, lengths, trade_returns
    
    trade_multiplier = trade_info[0]
    entry_price = trade_info[1]
    stop_loss = trade_info[2]
    
    # Handle optional take_profit
    if len(trade_info) > 3 and trade_info[3] is not None:
        take_profit = trade_info[3]
    else:
        take_profit = None
        
    position_size = trade_info[-2]
    share_amount = trade_info[-1]

    # Compute profit/loss
    if direction == 'Long':
        pnl = (exit_price - entry_price) * share_amount
    else:
        pnl = (entry_price - exit_price) * share_amount

    # Update capital
    available_capital += (share_amount * exit_price)
    portfolio_value += pnl

    # Track trade metrics
    duration = (exit_date - entry_date).days
    lengths.append(duration)

    if pnl > 0:
        wins.append(pnl)
    else:
        losses.append(pnl)

    # Update trade returns
    if exit_date in trade_returns.index:
        trade_returns[exit_date] += pnl
    
    # Log trade details
    trade_log.append({
        'Entry Date': entry_date,
        'Exit Date': exit_date,
        'Direction': direction,
        'Entry Price': entry_price,
        'Exit Price': exit_price,
        'Shares': share_amount,
        'PnL': pnl,
        'Duration': duration,
        'Exit Reason': reason
    })

    return portfolio_value, available_capital, trade_log, wins, losses, lengths, trade_returns

# --------------------------------------------------------------------------------------------------------------------------

def prepare_trade_entry(direction, price, atr, portfolio_value, available_capital, position_size, risk, reward=None):
    risk_amount = risk * portfolio_value
    trade_position_size = min(position_size, available_capital / portfolio_value)
    share_amount = int((trade_position_size * portfolio_value) // float(price))
    
    if share_amount == 0:
        return None

    if direction == 'Long':
        stop_loss = price - atr * 1.5
        take_profit = price + atr * 2 if reward else None
    elif direction == 'Short':
        stop_loss = price + atr * 1.5
        take_profit = price - atr * 2 if reward else None
    else:
        return None

    return price, stop_loss, take_profit, trade_position_size, share_amount

# --------------------------------------------------------------------------------------------------------------------------

def create_trade_entry(entry_date, direction, entry_price, position_size, share_amount):
    """Create a dictionary with trade entry details"""
    return {
        'Entry Date': entry_date,
        'Direction': direction,
        'Entry Price': entry_price,
        'Position Size': position_size,
        'Shares': share_amount,
        'Exit Date': None, 
        'Exit Price': None,
        'PnL': None,
        'Duration': None,
        'Exit Reason': None
    }

# --------------------------------------------------------------------------------------------------------------------------

def compile_trade_statistics(equity, trade_log, wins, losses, risk_free_rate=0.04):
    """
    Compile comprehensive statistics about the trading performance
    
    Parameters:
    equity (pd.Series): Portfolio equity curve
    trade_log (list): List of trade dictionaries containing trade details
    wins (list): List of winning trade amounts
    losses (list): List of losing trade amounts
    risk_free_rate (float): Annual risk-free rate (default 0.04 or 4%)
    
    Returns:
    dict: Dictionary of trading statistics
    """
    if not trade_log:
        return {
            'Total Trades': 0,
            'Win Rate': 0,
            'Net Profit (%)': 0,
            'Profit Factor': 0,
            'Expectancy (%)': 0,
            'Max Drawdown (%)': 0,
            'Annualized Return (%)': 0,
            'Sharpe Ratio': 0,
            'Sortino Ratio': 0
        }
    
    # Basic trade statistics
    total_trades = len(trade_log)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
    
    # Profit metrics
    gross_profit = sum(wins) if wins else 0
    gross_loss = sum(losses) if losses else 0
    net_profit = gross_profit + gross_loss
    
    # Starting and ending values
    initial_capital = equity.iloc[0]
    final_capital = equity.iloc[-1]
    net_profit_pct = ((final_capital / initial_capital) - 1) * 100
    
    # Risk metrics
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
    
    # Calculate average win and loss
    avg_win = gross_profit / win_count if win_count > 0 else 0
    avg_loss = gross_loss / loss_count if loss_count > 0 else 0
    
    # Calculate expectancy
    expectancy = ((win_rate/100) * avg_win) + ((1 - win_rate/100) * avg_loss)
    expectancy_pct = (expectancy / initial_capital) * 100 if initial_capital > 0 else 0
    
    # Calculate drawdown
    running_max = equity.cummax()
    drawdown = ((equity - running_max) / running_max) * 100
    max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
    
    # Calculate time-based metrics
    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25
    annualized_return = ((final_capital / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # Calculate daily returns for risk ratios
    daily_returns = equity.pct_change().dropna()
    
    # Calculate Sharpe ratio
    excess_returns = daily_returns - (risk_free_rate / 252)  # Daily risk-free rate
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0
    
    # Calculate Sortino ratio (only considering negative returns)
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = (daily_returns.mean() - (risk_free_rate / 252)) / downside_returns.std() * np.sqrt(252) if not downside_returns.empty and downside_returns.std() != 0 else 0
    
    return {
        'Total Trades': total_trades,
        'Win Rate': win_rate,
        'Net Profit (%)': net_profit_pct,
        'Profit Factor': profit_factor,
        'Expectancy (%)': expectancy_pct,
        'Max Drawdown (%)': max_drawdown,
        'Annualized Return (%)': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio
    }

# --------------------------------------------------------------------------------------------------------------------------
 
def create_backtest_charts(df, equity, trade_stats, trade_log, ticker):
    """Create visualization charts for the backtest results"""
    # Set style for plots
    plt.style.use('fivethirtyeight')
    
    # Create a 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Strategy Backtest Results for {ticker}', fontsize=16)
    
    # Plot 1: Equity Curve vs Buy & Hold
    ax1 = axes[0, 0]
    asset_returns = df['Close'] / df['Close'].iloc[0]
    equity_norm = equity / equity.iloc[0]
    
    ax1.plot(df.index, asset_returns, 'b-', label=f'{ticker} Buy & Hold')
    ax1.plot(df.index, equity_norm, 'g-', label='Strategy Equity')
    
    # Add buy and sell markers if available
    buys = [t for t in trade_log if t['Direction'] == 'Long']
    sells = [t for t in trade_log if t['Direction'] == 'Short']
    
    for trade in buys:
        if trade['Entry Date'] in df.index:
            ax1.scatter(trade['Entry Date'], asset_returns.loc[trade['Entry Date']], 
                      marker='^', color='green', s=100, alpha=0.7)
    
    for trade in sells:
        if trade['Entry Date'] in df.index:
            ax1.scatter(trade['Entry Date'], asset_returns.loc[trade['Entry Date']], 
                      marker='v', color='red', s=100, alpha=0.7)
            
    ax1.set_title('Equity Curve vs Buy & Hold')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Drawdown
    ax2 = axes[0, 1]
    running_max = equity.cummax()
    drawdown = ((equity - running_max) / running_max) * 100
    
    ax2.fill_between(df.index, drawdown, 0, color='red', alpha=0.3)
    ax2.set_title('Strategy Drawdown (%)')
    ax2.set_ylabel('Drawdown %')
    ax2.grid(True)
    
    # Plot 3: Trade Returns Distribution
    ax3 = axes[1, 0]
    all_returns = [t['PnL'] for t in trade_log if t['PnL'] is not None]
    if all_returns:
        sns.histplot(all_returns, kde=True, ax=ax3, color='blue')
        ax3.axvline(0, color='r', linestyle='--')
        ax3.set_title('Trade Returns Distribution')
        ax3.set_xlabel('Profit/Loss ($)')
    else:
        ax3.text(0.5, 0.5, 'No trade data available', ha='center', va='center')
    
    # Plot 4: Key Performance Metrics
    ax4 = axes[1, 1]
    metrics = [
        f"Total Trades: {trade_stats['Total Trades']}",
        f"Win Rate: {trade_stats['Win Rate']:.2f}%",
        f"Net Profit: {trade_stats['Net Profit (%)']:.2f}%",
        f"Max Drawdown: {trade_stats['Max Drawdown (%)']:.2f}%",
        f"Sharpe Ratio: {trade_stats['Sharpe Ratio']:.2f}",
        f"Profit Factor: {trade_stats['Profit Factor']:.2f}",
        f"Annualized Return: {trade_stats['Annualized Return (%)']:.2f}%"
    ]
    
    y_pos = range(len(metrics))
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(metrics)
    ax4.set_title('Key Performance Metrics')
    ax4.set_xlim([0, 1])  # Used only for spacing
    ax4.set_xticks([])  # Hide x-axis
    ax4.grid(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    return fig

# --------------------------------------------------------------------------------------------------------------------------

def run_parameter_test(params):
    """
    Run a single parameter test - designed to be used with multiprocessing
    
    Parameters:
    params (tuple): (ticker, risk, reward, position_size, create_charts, tech_params)
    
    Returns:
    dict: Results of the backtest with these parameters
    """
    
    # Unpack parameters
    if len(params) >= 6:
        ticker, risk, reward, position_size, create_charts, tech_params = params
    else:
        # Backward compatibility with older code
        ticker, risk, reward, position_size, create_charts = params
        tech_params = None  # Use default technical parameters
    
    # Get data for this test
    try:
        df = yf.download(ticker, period="5y", progress=False)  # Hide progress to avoid cluttering output
        
        # Get technical parameters
        fast = tech_params.get('FAST', FAST) if tech_params else FAST
        slow = tech_params.get('SLOW', SLOW) if tech_params else SLOW
        rsi_oversold = tech_params.get('RSI_OVERSOLD', RSI_OVERSOLD) if tech_params else RSI_OVERSOLD
        rsi_overbought = tech_params.get('RSI_OVERBOUGHT', RSI_OVERBOUGHT) if tech_params else RSI_OVERBOUGHT
        devs = tech_params.get('DEVS', DEVS) if tech_params else DEVS
            
        # Generate a unique cache filename based on technical parameters
        cache_params = f"_{fast}_{slow}_{rsi_oversold}_{rsi_overbought}_{devs}"
        cache_file = f"{ticker}_indicators{cache_params}_cache.pkl"
        
        # Add technical indicators with explicit parameters instead of using globals
        df_with_indicators = adding_indicators(
            df, 
            fast=fast,
            slow=slow,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            devs=devs,
            recalculate=False, 
            cache_file=cache_file
        )
        
        # Run backtest
        trade_log, trade_stats, portfolio_equity, trade_returns = set_momentum_strat(
            df_with_indicators, 
            risk=risk, 
            reward=reward, 
            position_size=position_size
        )
        
        # Create visualization only if requested
        if create_charts:
            create_backtest_charts(df_with_indicators, portfolio_equity, trade_stats, trade_log, ticker)
        
        # Return key metrics for comparison
        return {
            'ticker': ticker,
            'risk': risk,
            'reward': reward,
            'position_size': position_size,
            'tech_params': tech_params or {
                'FAST': FAST,
                'SLOW': SLOW,
                'RSI_OVERSOLD': RSI_OVERSOLD,
                'RSI_OVERBOUGHT': RSI_OVERBOUGHT,
                'DEVS': DEVS,
                'ADX_THRESHOLD': 25,  # Default value
                'MIN_BUY_SCORE': 3.0,  # Default value
                'MIN_SELL_SCORE': 3.0,  # Default value 
                'REQUIRE_CLOUD': True  # Default value
            },
            'win_rate': trade_stats['Win Rate'],
            'net_profit_pct': trade_stats['Net Profit (%)'],
            'max_drawdown': trade_stats['Max Drawdown (%)'],
            'sharpe': trade_stats['Sharpe Ratio'],
            'profit_factor': trade_stats['Profit Factor'],
            'num_trades': trade_stats['Total Trades']
        }
        
    except Exception as e:
        print(f"Error testing parameters for {ticker}: {e}")
        # Return a result with very poor performance so it won't be selected
        return {
            'ticker': ticker,
            'risk': risk,
            'reward': reward,
            'position_size': position_size,
            'tech_params': tech_params or {
                'FAST': FAST,
                'SLOW': SLOW,
                'RSI_OVERSOLD': RSI_OVERSOLD,
                'RSI_OVERBOUGHT': RSI_OVERBOUGHT,
                'DEVS': DEVS,
                'ADX_THRESHOLD': 25,
                'MIN_BUY_SCORE': 3.0,
                'MIN_SELL_SCORE': 3.0,
                'REQUIRE_CLOUD': True
            },
            'win_rate': 0.0,
            'net_profit_pct': -100.0,
            'max_drawdown': 100.0,
            'sharpe': -10.0,
            'profit_factor': 0.0,
            'num_trades': 0
        }

# --------------------------------------------------------------------------------------------------------------------------

def main():
    # Set working directory using absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(script_dir))
    
    # Create a command line argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Momentum Trading Strategy')
    parser.add_argument('--ticker', type=str, default='SPY', help='Stock ticker to analyze')
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    parser.add_argument('--backtest', action='store_true', help='Run backtest with default parameters')
    parser.add_argument('--risk', type=float, default=DEFAULT_RISK, help='Risk percentage per trade')
    parser.add_argument('--reward', type=float, default=DEFAULT_REWARD, help='Reward percentage per trade')
    parser.add_argument('--position', type=float, default=DEFAULT_POSITION_SIZE, help='Position size as percentage')
    args = parser.parse_args()
    
    ticker = args.ticker
    
    # Choose action based on command line arguments
    if args.optimize:
        # Run parameter optimization using multiprocessing
        print(f"Optimizing parameters for {ticker}...")
        best_result, all_results = optimize_parameters(ticker=ticker, visualize_best=True)
        print(f"Best parameters found: Risk={best_result['risk']*100:.1f}%, " +
              f"Reward={best_result['reward']*100:.1f}%, Position={best_result['position_size']*100:.1f}%")
    else:
        # Run standard backtest
        print(f"Running backtest for {ticker} with parameters: " +
              f"Risk={args.risk*100:.1f}%, Reward={args.reward*100:.1f}%, Position={args.position*100:.1f}%")
        
        # Get data with caching
        print(f"Loading data for {ticker}...")
        df = yf.download(ticker, period="5y")
        
        # Use the optimized indicator calculation with caching
        df_with_indicators = adding_indicators(
            df, 
            recalculate=False, 
            cache_file=f"{ticker}_indicators_cache.pkl"
        )
        
        # Run the backtest
        result = test_strategy(
            df_with_indicators,
            ticker,
            risk=args.risk,
            reward=args.reward, 
            position_size=args.position
        )
        
        return result

if __name__ == "__main__":
    main()