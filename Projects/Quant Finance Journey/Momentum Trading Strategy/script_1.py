import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from collections import deque
import os
from tabulate import tabulate  # Add this import
import numba as nb  # Add numba for JIT compilation
import multiprocessing as mp  # Add multiprocessing for parameter testing
import pandas_ta as ta  # Add pandas-ta for vectorized TA functions

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

def get_data(tickers, start_date=None, end_date=None, lookback=DEFAULT_LOOKBACK):
    if isinstance(tickers, str):
        tickers = [tickers]
    
    # Set default date range if not provided
    if end_date is None:
        end_date = datetime.now()
    else:
        end_date = pd.to_datetime(end_date)
        
    if start_date is None:
        start_date = end_date - timedelta(days=lookback)
    else:
        start_date = pd.to_datetime(start_date)
    
    data_dict = {}
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            # Ensure column names are simplified
            df.columns = df.columns.get_level_values(0) if hasattr(df.columns, 'get_level_values') else df.columns
            # Add ticker column for identification
            df['Ticker'] = ticker
            data_dict[ticker] = df
            print(f"Downloaded {len(df)} bars for {ticker}")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    
    return df, data_dict

# --------------------------------------------------------------------------------------------------------------------------
# Functions for all indicators
def adding_indicators(df, recalculate=False, cache_file="indicator_cache.pkl"):
    """
    Calculate technical indicators with vectorized operations for better performance
    
    Parameters:
    df: DataFrame with OHLCV data
    recalculate: Force recalculation even if cache exists
    cache_file: File to cache indicators to avoid recalculation
    
    Returns:
    DataFrame with added indicators
    """
    # Use caching to avoid recalculating indicators
    if cache_file and os.path.exists(cache_file) and not recalculate:
        try:
            return pd.read_pickle(cache_file)
        except Exception:
            pass  # If error reading cache, recalculate
    
    # --- Efficient vectorized calculations ---
    
    # Moving Averages - vectorized
    df[f'{FAST}_ma'] = df['Close'].rolling(FAST).mean()
    df[f'{SLOW}_ma'] = df['Close'].rolling(SLOW).mean()

    # MACD - vectorized
    macd = ta.macd(df['Close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SPAN)
    df['MACD'] = macd['MACD_' + str(MACD_FAST) + '_' + str(MACD_SLOW) + '_' + str(MACD_SPAN)]
    df['Signal'] = macd['MACDs_' + str(MACD_FAST) + '_' + str(MACD_SLOW) + '_' + str(MACD_SPAN)]
    df['MACD_hist'] = macd['MACDh_' + str(MACD_FAST) + '_' + str(MACD_SLOW) + '_' + str(MACD_SPAN)]

    # RSI - vectorized
    df['RSI'] = ta.rsi(df['Close'], length=RSI_LENGTH)

    # MFI - vectorized
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=MFI_LENGTH)

    # Bollinger Bands - vectorized
    bbands = ta.bbands(df['Close'], length=BB_LEN, std=DEVS)
    df['BB_SMA'] = bbands['BBM_' + str(BB_LEN) + '_' + str(float(DEVS))]
    df['Upper_Band'] = bbands['BBU_' + str(BB_LEN) + '_' + str(float(DEVS))]
    df['Lower_Band'] = bbands['BBL_' + str(BB_LEN) + '_' + str(float(DEVS))]

    # ATR - vectorized
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # ADX - vectorized
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx['ADX_14']

    # Parabolic SAR - vectorized
    df['SAR'] = ta.psar(df['High'], df['Low'], df['Close'])['PSARl_0.02_0.2']

    # ROC - vectorized
    df['ROC'] = ta.roc(df['Close'], length=12)

    # Ichimoku Cloud - vectorized
    ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])
    df['Tenkan_sen'] = ichimoku['ITS_9']  # Conversion line
    df['Kijun_sen'] = ichimoku['IKS_26']  # Base line
    df['Senkou_span_A'] = ichimoku['ISA_9']  # Leading span A
    df['Senkou_span_B'] = ichimoku['ISB_26']  # Leading span B
    df['Chikou_span'] = df['Close'].shift(-26)  # Lagging span

    # Save to cache if requested
    if cache_file:
        df.to_pickle(cache_file)
    
    return df.dropna()

# --------------------------------------------------------------------------------------------------------------------------

@nb.jit(nopython=True)
def calculate_signals_numba(fast_ma, slow_ma, rsi, mfi, close, lower_band, upper_band, 
                           atr, adx, sar, roc, tenkan, kijun, spanA, spanB, close_26_ago=None,
                           adx_threshold=25, min_buy_score=3.0, min_sell_score=3.0, require_cloud=True):
    """
    Numba-accelerated signal calculation function - much faster than non-compiled version
    
    Parameters now include optimizable thresholds:
    - adx_threshold: ADX level that indicates strong trend (default: 25)
    - min_buy_score: Minimum score threshold for buy signals (default: 3.0)
    - min_sell_score: Minimum score threshold for sell signals (default: 3.0)
    - require_cloud: Whether to require cloud context for signals (default: True)
    """
    # Moving average crossover
    ma_buy = fast_ma > slow_ma and fast_ma <= slow_ma
    ma_sell = fast_ma < slow_ma and fast_ma >= slow_ma
    
    # RSI signals
    rsi_buy = rsi < 30
    rsi_sell = rsi > 70
    
    # Bollinger Bands
    bb_buy = close < lower_band
    bb_sell = close > upper_band
    
    # MFI
    mfi_buy = mfi < 20
    mfi_sell = mfi > 80
    
    # ADX trend strength - now using the parameter
    strong_trend = adx > adx_threshold
    
    # Parabolic SAR
    sar_buy = close > sar
    sar_sell = close < sar
    
    # Rate of Change momentum
    roc_buy = roc > 0
    roc_sell = roc < 0
    
    # Ichimoku Cloud
    above_cloud = close > max(spanA, spanB)
    below_cloud = close < min(spanA, spanB)
    tenkan_kijun_buy = tenkan > kijun
    tenkan_kijun_sell = tenkan < kijun
    price_kijun_buy = close > kijun
    price_kijun_sell = close < kijun
    
    # Check if we have historical data for Chikou span
    chikou_buy = close > close_26_ago if close_26_ago is not None else False
    chikou_sell = close < close_26_ago if close_26_ago is not None else False
    
    # Calculate signal scores
    buy_score = 0.0
    sell_score = 0.0
    
    # Oscillator signals (work in any market)
    if rsi_buy: buy_score += 1.0
    if bb_buy: buy_score += 1.0
    if mfi_buy: buy_score += 1.0
    
    if rsi_sell: sell_score += 1.0
    if bb_sell: sell_score += 1.0
    if mfi_sell: sell_score += 1.0
    
    # Add trend signals
    if strong_trend:  # Only add trend signals in strong trends
        if ma_buy: buy_score += 1.5
        if sar_buy: buy_score += 1.25
        if roc_buy: buy_score += 1.25
        if above_cloud: buy_score += 1.5
        if tenkan_kijun_buy: buy_score += 1.5
        if price_kijun_buy: buy_score += 1.0
        if chikou_buy: buy_score += 0.5
        
        if ma_sell: sell_score += 1.5
        if sar_sell: sell_score += 1.25
        if roc_sell: sell_score += 1.25
        if below_cloud: sell_score += 1.5
        if tenkan_kijun_sell: sell_score += 1.5
        if price_kijun_sell: sell_score += 1.0
        if chikou_sell: sell_score += 0.5
    
    # Final signals with threshold parameters and cloud context requirement
    if require_cloud:
        buy_signal = buy_score >= min_buy_score and (strong_trend or above_cloud)
        sell_signal = sell_score >= min_sell_score and (strong_trend or below_cloud)
    else:
        # Don't require cloud context if parameter is set to False
        buy_signal = buy_score >= min_buy_score and strong_trend
        sell_signal = sell_score >= min_sell_score and strong_trend
    
    return buy_signal, sell_signal, buy_score, sell_score

# --------------------------------------------------------------------------------------------------------------------------

def prepare_trade_entry(direction, price, atr, portfolio_value, available_capital, position_size, risk, reward=None):
    risk_amount = risk * portfolio_value
    trade_position_size = min(position_size, available_capital / portfolio_value)
    share_amount = (trade_position_size * portfolio_value) // price

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

def calculate_signals(df, i):
    ma_buy = df[f'{FAST}_ma'].iloc[i] > df[f'{SLOW}_ma'].iloc[i] and df[f'{FAST}_ma'].iloc[i-1] <= df[f'{SLOW}_ma'].iloc[i-1]
    ma_sell = df[f'{FAST}_ma'].iloc[i] < df[f'{SLOW}_ma'].iloc[i] and df[f'{FAST}_ma'].iloc[i-1] >= df[f'{SLOW}_ma'].iloc[i-1]
    rsi_buy = df['RSI'].iloc[i] < 30
    rsi_sell = df['RSI'].iloc[i] > 70
    bb_buy = df['Close'].iloc[i] < df['Lower_Band'].iloc[i]
    bb_sell = df['Close'].iloc[i] > df['Upper_Band'].iloc[i]
    mfi_buy = df['MFI'].iloc[i] < 20
    mfi_sell = df['MFI'].iloc[i] > 80
    
    # New indicator signals
    # ADX trend strength filter (ADX > 25 indicates strong trend)
    strong_trend = df['ADX'].iloc[i] > 25
    
    # Parabolic SAR signals
    sar_buy = df['Close'].iloc[i] > df['SAR'].iloc[i] and df['Close'].iloc[i-1] <= df['SAR'].iloc[i-1]
    sar_sell = df['Close'].iloc[i] < df['SAR'].iloc[i] and df['Close'].iloc[i-1] >= df['SAR'].iloc[i-1]
    
    # Rate of Change momentum
    roc_buy = df['ROC'].iloc[i] > 0 and df['ROC'].iloc[i] > df['ROC'].iloc[i-1]  # Positive and increasing ROC
    roc_sell = df['ROC'].iloc[i] < 0 and df['ROC'].iloc[i] < df['ROC'].iloc[i-1]  # Negative and decreasing ROC
    
    # Ichimoku Cloud signals - adjusted to match your variable names
    # Price above cloud is bullish, below is bearish
    above_cloud = df['Close'].iloc[i] > max(df['Senkou_span_A'].iloc[i], df['Senkou_span_B'].iloc[i])
    below_cloud = df['Close'].iloc[i] < min(df['Senkou_span_A'].iloc[i], df['Senkou_span_B'].iloc[i])
    
    # Tenkan-sen/Kijun-sen crossover
    tenkan_kijun_buy = df['Tenkan_sen'].iloc[i] > df['Kijun_sen'].iloc[i] and df['Tenkan_sen'].iloc[i-1] <= df['Kijun_sen'].iloc[i-1]
    tenkan_kijun_sell = df['Tenkan_sen'].iloc[i] < df['Kijun_sen'].iloc[i] and df['Tenkan_sen'].iloc[i-1] >= df['Kijun_sen'].iloc[i-1]
    
    # Additional Ichimoku signal: Price crossing Kijun-sen (traditional entry signal)
    price_kijun_buy = df['Close'].iloc[i] > df['Kijun_sen'].iloc[i] and df['Close'].iloc[i-1] <= df['Kijun_sen'].iloc[i-1]
    price_kijun_sell = df['Close'].iloc[i] < df['Kijun_sen'].iloc[i] and df['Close'].iloc[i-1] >= df['Kijun_sen'].iloc[i-1]
    
    # Chikou span confirmation (current price vs price 26 periods ago)
    chikou_buy = df['Close'].iloc[i] > df['Close'].iloc[i-26] if i >= 26 else False
    chikou_sell = df['Close'].iloc[i] < df['Close'].iloc[i-26] if i >= 26 else False
    
    # Signal scoring system
    buy_score = 0
    sell_score = 0

    # Use oscillator-based signals in any market
    if rsi_buy: buy_score += 1
    if bb_buy: buy_score += 1
    if mfi_buy: buy_score += 1
    # same for sell signals...

    if strong_trend:  # ADX > 25
        # Add trend-following signals only when trend is strong
        if ma_buy: buy_score += 1.5
        if sar_buy: buy_score += 1.25
        if roc_buy: buy_score += 1.25
        if above_cloud: buy_score += 1.5
        if tenkan_kijun_buy: buy_score += 1.5
        if price_kijun_buy: buy_score += 1
        if chikou_buy: buy_score += 0.5
        
    # Add points for each bearish signal
    if ma_sell: sell_score += 1
    if rsi_sell: sell_score += 1
    if bb_sell: sell_score += 1
    if mfi_sell: sell_score += 1
    if sar_sell: sell_score += 1
    if roc_sell: sell_score += 1
    if below_cloud: sell_score += 1
    if tenkan_kijun_sell: sell_score += 1
    if price_kijun_sell: sell_score += 1
    if chikou_sell: sell_score += 1
    
    # Define minimum threshold for signal confirmation
    min_score = 3
    
    # Final signals with trend strength filter and cloud context
    buy_signal = buy_score >= min_score and (strong_trend or above_cloud)
    sell_signal = sell_score >= min_score and (strong_trend or below_cloud)
    
    return buy_signal, sell_signal, buy_score, sell_score

# --------------------------------------------------------------------------------------------------------------------------

def set_momentum_strat(df, risk=DEFAULT_RISK, reward=DEFAULT_REWARD, position_size=DEFAULT_POSITION_SIZE, 
                      max_positions=MAX_OPEN_POSITIONS, risk_free_rate=0.04):
    """
    Execute the momentum strategy on the given DataFrame
    
    Uses optimized data structures and vectorized operations where possible
    """
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
        
        # Use JIT-compiled signal calculation for improved performance when available
        try:
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
        except Exception:
            # Fallback to non-JIT version if there's an issue with numba
            buy_signal, sell_signal, buy_score, sell_score = calculate_signals(df, i)
        
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
            take_profit_exits = ~active_short_trades['take_profit'].isna() & (current_price <= active_short_trades['take_profit'])
            stop_loss_exits = current_price >= active_short_trades['stop_loss']
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
    equity.iloc[-1] = portfolio_value

    # --- Compile trade stats ---
    trade_stats = compile_trade_statistics(equity, trade_log, wins, losses, risk_free_rate)

    return trade_log, trade_stats, equity, trade_returns

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

def test_strategy(df, ticker, risk=DEFAULT_RISK, reward=DEFAULT_REWARD, position_size=DEFAULT_POSITION_SIZE):
    
    # Add technical indicators
    df = adding_indicators(df)
    
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
    create_backtest_charts(df, portfolio_equity, trade_stats, trade_log, ticker)
    
    return {
        'ticker': ticker,
        'df': df,
        'equity': portfolio_equity,
        'trade_stats': trade_stats,
        'trade_log': trade_log
    }

# --------------------------------------------------------------------------------------------------------------------------

def run_parameter_test(params):
    """
    Run a single parameter test - designed to be used with multiprocessing
    
    Parameters:
    params (tuple): (ticker, risk, reward, position_size, create_charts, tech_params)
    
    Returns:
    dict: Results of the backtest with these parameters
    """
    # Declare globals at the beginning of the function
    global FAST, SLOW, RSI_OVERSOLD, RSI_OVERBOUGHT, DEVS
    
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
        
        # Apply technical parameters if provided
        if tech_params:
            # Save original global values
            orig_fast, orig_slow = FAST, SLOW
            orig_rsi_oversold, orig_rsi_overbought = RSI_OVERSOLD, RSI_OVERBOUGHT
            orig_devs = DEVS
            
            # Temporarily modify global parameters for this run
            FAST = tech_params.get('FAST', FAST)
            SLOW = tech_params.get('SLOW', SLOW)
            RSI_OVERSOLD = tech_params.get('RSI_OVERSOLD', RSI_OVERSOLD)
            RSI_OVERBOUGHT = tech_params.get('RSI_OVERBOUGHT', RSI_OVERBOUGHT)
            DEVS = tech_params.get('DEVS', DEVS)
            
        # Generate a unique cache filename based on technical parameters
        cache_params = f"_{FAST}_{SLOW}_{RSI_OVERSOLD}_{RSI_OVERBOUGHT}_{DEVS}"
        cache_file = f"{ticker}_indicators{cache_params}_cache.pkl"
        
        # Add technical indicators with caching
        df_with_indicators = adding_indicators(df, recalculate=False, cache_file=cache_file)
        
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
        
        # Restore original global parameters if we modified them
        if tech_params:
            FAST, SLOW = orig_fast, orig_slow
            RSI_OVERSOLD, RSI_OVERBOUGHT = orig_rsi_oversold, orig_rsi_overbought
            DEVS = orig_devs
            
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
        
        # Apply the best technical parameters
        global FAST, SLOW, RSI_OVERSOLD, RSI_OVERBOUGHT, DEVS
        # Save original values
        orig_fast, orig_slow = FAST, SLOW
        orig_rsi_oversold, orig_rsi_overbought = RSI_OVERSOLD, RSI_OVERBOUGHT
        orig_devs = DEVS
        
        # Apply best parameters
        FAST = best_result['tech_params']['FAST']
        SLOW = best_result['tech_params']['SLOW']
        RSI_OVERSOLD = best_result['tech_params']['RSI_OVERSOLD']
        RSI_OVERBOUGHT = best_result['tech_params']['RSI_OVERBOUGHT']
        DEVS = best_result['tech_params']['DEVS']
        
        # Reset indicator cache - force recalculation with new parameters
        cache_params = f"_{FAST}_{SLOW}_{RSI_OVERSOLD}_{RSI_OVERBOUGHT}_{DEVS}"
        cache_file = f"{ticker}_indicators{cache_params}_cache.pkl"
        df_with_indicators = adding_indicators(df, recalculate=True, cache_file=cache_file)
        
        # Run backtest with best parameters
        result = test_strategy(
            df_with_indicators,
            ticker,
            risk=best_result['risk'],
            reward=best_result['reward'],
            position_size=best_result['position_size']
        )
        
        # Restore original parameters
        FAST, SLOW = orig_fast, orig_slow
        RSI_OVERSOLD, RSI_OVERBOUGHT = orig_rsi_oversold, orig_rsi_overbought
        DEVS = orig_devs
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"\nParameter optimization completed in {elapsed_time}")
    
    return best_result, sorted_results

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

