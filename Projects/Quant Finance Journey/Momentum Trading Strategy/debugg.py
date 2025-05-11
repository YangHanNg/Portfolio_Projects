import pandas as pd
import numpy as np
import yfinance as yf
import os
from tabulate import tabulate  
import pandas_ta as ta 
from datetime import datetime, timedelta
import multiprocessing as mp 
import itertools

# --------------------------------------------------------------------------------------------------------------------------

# Base parameters
TICKER = ['SPY']
INITIAL_CAPITAL = 10000.0

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
DEFAULT_LONG_RISK = 0.02  # 5% risk per long trade
DEFAULT_LONG_REWARD = 0.1  # 10% target profit per long trade
DEFAULT_SHORT_RISK = 0.01  # 5% risk per short trade
DEFAULT_SHORT_REWARD = 0.03  # 10% target profit per short trade
DEFAULT_POSITION_SIZE = 0.8  # 10% of portfolio per trade
MAX_OPEN_POSITIONS = 2  # Maximum number of concurrent open positions

TREND_STRONG = 1
TREND_WEAK = 0.75
ADX_THRESHOLD_DEFAULT = 25
MIN_BUY_SCORE_DEFAULT = 2.5
MIN_SELL_SCORE_DEFAULT = 5.5
REQUIRE_CLOUD_DEFAULT = True # Default for Ichimoku cloud requirement in signals

# Script Execution Defaults
OPTIMIZE_DEFAULT = False
VISUALIZE_BEST_DEFAULT = False # For optimize_parameters
OPTIMIZATION_TYPE_DEFAULT = 'basic'

# --------------------------------------------------------------------------------------------------------------------------

def indicators(df, fast=None, slow=None, rsi_oversold=None, rsi_overbought=None, devs=None):
    """Calculate technical indicators with NumPy operations for maximum performance"""
    # Use provided parameters or fall back to global defaults
    fast = fast if fast is not None else FAST
    slow = slow if slow is not None else SLOW
    rsi_oversold = rsi_oversold if rsi_oversold is not None else RSI_OVERSOLD
    rsi_overbought = rsi_overbought if rsi_overbought is not None else RSI_OVERBOUGHT
    devs = devs if devs is not None else DEVS
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Check if the dataframe is empty
    if len(df) == 0:
        print("Warning: Empty dataframe provided to adding_indicators!")
        return df
    
    try:
        # Extract numpy arrays for faster processing
        close_np = df['Close'].values
        high_np = df['High'].values
        low_np = df['Low'].values
        volume_np = df['Volume'].values
        
        # Precompute rolling windows using NumPy
        # Helper function for rolling windows
        def rolling_window(a, window):
            shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
            strides = a.strides + (a.strides[-1],)
            return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
            
        # Moving Averages with NumPy
        fast_ma = np.empty_like(close_np)
        slow_ma = np.empty_like(close_np)
        
        # Fill initial values with NaN
        fast_ma[:fast-1] = np.nan
        slow_ma[:slow-1] = np.nan
        
        # Calculate moving averages
        for i in range(fast-1, len(close_np)):
            fast_ma[i] = np.mean(close_np[i-fast+1:i+1])
        
        for i in range(slow-1, len(close_np)):
            slow_ma[i] = np.mean(close_np[i-slow+1:i+1])
        
        # MACD - using NumPy for EMA calculation
        alpha_fast = 2.0 / (MACD_FAST + 1)
        alpha_slow = 2.0 / (MACD_SLOW + 1)
        alpha_signal = 2.0 / (MACD_SPAN + 1)
        
        # Initialize EMA arrays
        ema_fast = np.empty_like(close_np)
        ema_slow = np.empty_like(close_np)
        
        # Set first value to SMA
        ema_fast[0] = close_np[0]
        ema_slow[0] = close_np[0]
        
        # Calculate EMAs
        for i in range(1, len(close_np)):
            ema_fast[i] = alpha_fast * close_np[i] + (1 - alpha_fast) * ema_fast[i-1]
            ema_slow[i] = alpha_slow * close_np[i] + (1 - alpha_slow) * ema_slow[i-1]
        
        # MACD and Signal Line
        macd = ema_fast - ema_slow
        signal = np.empty_like(macd)
        signal[0] = macd[0]
        
        for i in range(1, len(macd)):
            signal[i] = alpha_signal * macd[i] + (1 - alpha_signal) * signal[i-1]
        
        macd_hist = macd - signal
        
        # RSI implementation in NumPy
        try:
            # Delta of closing prices
            delta = np.diff(close_np)
            delta = np.append(np.nan, delta)  # Add NaN at the beginning to maintain shape
            
            # Separate gains and losses
            gains = np.copy(delta)
            losses = np.copy(delta)
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = abs(losses)
            
            # Use rolling window for average gain/loss
            rsi = np.empty_like(close_np)
            rsi[:RSI_LENGTH] = np.nan
            
            # First RSI uses SMA - add better error handling
            if len(gains[1:RSI_LENGTH+1]) > 0:  # Check if there are enough elements
                avg_gain = np.nanmean(gains[1:RSI_LENGTH+1])
                avg_loss = np.nanmean(losses[1:RSI_LENGTH+1])
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                else:
                    rs = 1.0
                
                rsi[RSI_LENGTH] = 100 - (100 / (1 + rs))
                
                # Subsequent RSI values use smoothing
                for i in range(RSI_LENGTH+1, len(close_np)):
                    avg_gain = (avg_gain * (RSI_LENGTH - 1) + gains[i]) / RSI_LENGTH
                    avg_loss = (avg_loss * (RSI_LENGTH - 1) + losses[i]) / RSI_LENGTH
                    
                    if avg_loss != 0:
                        rs = avg_gain / avg_loss
                    else:
                        rs = 1.0
                    
                    rsi[i] = 100 - (100 / (1 + rs))
            else:
                # Not enough data points, fill with default value
                rsi = np.full_like(close_np, 50)  # Default to middle value
                    
        except Exception as e:
            print(f"Error calculating RSI with NumPy: {e}")
            # Fall back to pandas-ta with better error handling
            try:
                rsi_series = ta.rsi(df['Close'], length=RSI_LENGTH)
                if rsi_series is not None:
                    rsi = rsi_series.values
                else:
                    rsi = np.full_like(close_np, 50)  # Default value
            except Exception:
                rsi = np.full_like(close_np, 50)  # Default value
        
        # MFI implementation in NumPy
        try:
            # Typical price
            typical_price = (high_np + low_np + close_np) / 3
            
            # Money flow
            money_flow = typical_price * volume_np
            
            # Delta for direction
            delta_tp = np.diff(typical_price)
            delta_tp = np.append(np.nan, delta_tp)
            
            # Positive and negative money flow
            pos_flow = np.where(delta_tp > 0, money_flow, 0)
            neg_flow = np.where(delta_tp < 0, money_flow, 0)
            
            # MFI array
            mfi = np.empty_like(close_np)
            mfi[:MFI_LENGTH] = np.nan
            
            # Calculate MFI
            for i in range(MFI_LENGTH, len(close_np)):
                pos_sum = np.sum(pos_flow[i-MFI_LENGTH+1:i+1])
                neg_sum = np.sum(neg_flow[i-MFI_LENGTH+1:i+1])
                
                if neg_sum != 0:
                    money_ratio = pos_sum / neg_sum
                    mfi[i] = 100 - (100 / (1 + money_ratio))
                else:
                    mfi[i] = 100
            
        except Exception as e:
            print(f"Error calculating MFI with NumPy: {e}")
            # Use RSI as fallback
            mfi = rsi
        
        # Bollinger Bands - using NumPy
        bb_sma = np.empty_like(close_np)
        bb_std = np.empty_like(close_np)
        upper_band = np.empty_like(close_np)
        lower_band = np.empty_like(close_np)
        
        # Fill initial values with NaN
        bb_sma[:BB_LEN-1] = np.nan
        bb_std[:BB_LEN-1] = np.nan
        upper_band[:BB_LEN-1] = np.nan
        lower_band[:BB_LEN-1] = np.nan
        
        # Calculate BB
        for i in range(BB_LEN-1, len(close_np)):
            window = close_np[i-BB_LEN+1:i+1]
            bb_sma[i] = np.mean(window)
            bb_std[i] = np.std(window)
            upper_band[i] = bb_sma[i] + (bb_std[i] * devs)
            lower_band[i] = bb_sma[i] - (bb_std[i] * devs)
        
        # ATR - using NumPy
        try:
            tr = np.zeros_like(close_np)
            
            # First TR value
            tr[0] = high_np[0] - low_np[0]
            
            # Calculate subsequent TR values
            for i in range(1, len(close_np)):
                hl = high_np[i] - low_np[i]
                hc = abs(high_np[i] - close_np[i-1])
                lc = abs(low_np[i] - close_np[i-1])
                tr[i] = max(hl, hc, lc)
            
            # Calculate ATR with simple moving average
            atr = np.empty_like(close_np)
            atr[:14] = np.nan
            atr[13] = np.mean(tr[:14])
            
            # Subsequent ATR values use smoothing
            for i in range(14, len(close_np)):
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14
                
        except Exception as e:
            print(f"Error calculating ATR with NumPy: {e}")
            # Fall back to pandas-ta
            atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).values
        
        # Simple proxy for ADX (would need more complex calculation for real ADX)
        adx = np.full_like(close_np, 25)  # Default to medium trend strength
        
        # Simple proxy for SAR
        sar = close_np * 0.97
        
        # ROC calculation using NumPy
        roc = np.empty_like(close_np)
        roc[:12] = np.nan
        
        for i in range(12, len(close_np)):
            roc[i] = ((close_np[i] / close_np[i-12]) - 1) * 100
        
        # Ichimoku Cloud components in NumPy
        # Convert Ichimoku calculations to NumPy
        tenkan_sen = np.empty_like(close_np)
        kijun_sen = np.empty_like(close_np)
        senkou_span_a = np.empty_like(close_np)
        senkou_span_b = np.empty_like(close_np)
        chikou_span = np.empty_like(close_np)
        
        # Initialize with NaNs
        tenkan_sen[:8] = np.nan
        kijun_sen[:25] = np.nan
        senkou_span_a[:] = np.nan  # Will be shifted 26 periods
        senkou_span_b[:] = np.nan  # Will be shifted 26 periods
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        for i in range(8, len(close_np)):
            tenkan_sen[i] = (np.max(high_np[i-8:i+1]) + np.min(low_np[i-8:i+1])) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        for i in range(25, len(close_np)):
            kijun_sen[i] = (np.max(high_np[i-25:i+1]) + np.min(low_np[i-25:i+1])) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 shifted 26 periods forward
        for i in range(25, len(close_np)):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods forward
        for i in range(51, len(close_np)):
            senkou_span_b[i] = (np.max(high_np[i-51:i+1]) + np.min(low_np[i-51:i+1])) / 2
        
        # Chikou Span (Lagging Span): Close price shifted 26 periods backward
        chikou_span[:-26] = close_np[26:]
        chikou_span[-26:] = np.nan

        # Apply calculated arrays back to DataFrame
        df.loc[:, f'{fast}_ma'] = fast_ma
        df.loc[:, f'{slow}_ma'] = slow_ma
        df.loc[:, 'EMA_fast'] = ema_fast
        df.loc[:, 'EMA_slow'] = ema_slow
        df.loc[:, 'MACD'] = macd
        df.loc[:, 'Signal'] = signal
        df.loc[:, 'MACD_hist'] = macd_hist
        df.loc[:, 'RSI'] = rsi
        df.loc[:, 'MFI'] = mfi
        df.loc[:, 'BB_SMA'] = bb_sma
        df.loc[:, 'Upper_Band'] = upper_band  # Match the name expected in momentum()
        df.loc[:, 'Lower_Band'] = lower_band  # Match the name expected in momentum()
        df.loc[:, 'ATR'] = atr
        df.loc[:, 'ADX'] = adx
        df.loc[:, 'SAR'] = sar
        df.loc[:, 'ROC'] = roc
        df.loc[:, 'Tenkan_sen'] = tenkan_sen
        df.loc[:, 'Kijun_sen'] = kijun_sen
        df.loc[:, 'Senkou_span_A'] = np.roll(senkou_span_a, 26)  # Ensure name matches
        df.loc[:, 'Senkou_span_B'] = np.roll(senkou_span_b, 26)  # Ensure name matches
        df.loc[:, 'Chikou_span'] = chikou_span

    
    except Exception as e:
        print(f"Critical error in adding_indicators_numpy: {e}")
        # Return original dataframe but ensure it has proper columns to avoid downstream errors
        # Add the required columns with NaN values
        required_columns = [f'{fast}_ma', f'{slow}_ma', 'RSI', 'MFI', 'Lower_Band',
                           'Upper_Band', 'ATR', 'ADX', 'SAR', 'ROC', 'Tenkan_sen',
                           'Kijun_sen', 'Senkou_span_A', 'Senkou_span_B']
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        return df
        
    return df.dropna()

# --------------------------------------------------------------------------------------------------------------------------

def momentum(df, 
             long_risk=DEFAULT_LONG_RISK, long_reward=DEFAULT_LONG_REWARD,
             short_risk=DEFAULT_SHORT_RISK, short_reward=DEFAULT_SHORT_REWARD,
             position_size=DEFAULT_POSITION_SIZE, 
             max_positions=MAX_OPEN_POSITIONS, risk_free_rate=0.04):
    """
    Execute the momentum strategy with optimized NumPy operations
    
    Uses pre-allocation, vectorized operations and minimizes object creation
    """
    # Check if DataFrame is empty after preprocessing
    if len(df) == 0:
        print("Warning: Empty dataframe provided to momentum. Returning empty results.")
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
    
    # Pre-extract data from dataframe to avoid repeated access
    df_index = df.index
    close_values = df['Close'].values

    # Create param column names once
    param_cols = [f'{FAST}_ma', f'{SLOW}_ma', 'RSI', 'MFI', 'Close', 'Lower_Band',
                 'Upper_Band', 'ATR', 'ADX', 'SAR', 'ROC', 'Tenkan_sen',
                 'Kijun_sen', 'Senkou_span_A', 'Senkou_span_B']
    
    # Check columns once before the loop
    missing_cols = [col for col in param_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        print("Available columns:", df.columns.tolist())
    
    # Initialize tracking variables
    initial_capital = INITIAL_CAPITAL
    available_capital = initial_capital
    portfolio_value = initial_capital

    # Pre-allocate arrays for better performance
    equity = pd.Series(initial_capital, index=df_index, dtype='float64')
    trade_returns = pd.Series(0.0, index=df_index)
    
    # Define dtypes for active trade DataFrames
    common_trade_columns_dtypes = {
        'entry_date': 'datetime64[ns]',
        'multiplier': 'int64',
        'entry_price': 'float64',
        'stop_loss': 'float64',
        'take_profit': 'float64', # Pandas handles None as np.nan in float columns
        'position_size': 'float64',
        'share_amount': 'int64'
    }

    # Use DataFrames for active trades (more efficient than appending to lists)
    active_long_trades = pd.DataFrame(columns=common_trade_columns_dtypes.keys()).astype(common_trade_columns_dtypes)
    active_short_trades = pd.DataFrame(columns=common_trade_columns_dtypes.keys()).astype(common_trade_columns_dtypes)
    
    # Initialize position counters
    long_positions = short_positions = 0

    # Trade tracking with pre-allocation for better append performance
    trade_log = []
    # Pre-allocate with reasonable initial capacity
    wins = []
    losses = []
    lengths = []

    prev_buy_signal = False
    prev_sell_signal = False
    prev_buy_score = 0.0
    prev_sell_score = 0.0

    # Main loop - start from 1 to avoid index error on i-1
    for i in range(1, len(df)):
        current_date = df_index[i]
        current_price = close_values[i]
        
        # Default equity to previous day (will be updated if trades exit)
        equity.iloc[i] = equity.iloc[i-1]
        
        # Signal calculation for current bar
        try:
            if missing_cols:
                buy_signal, sell_signal = False, False
                buy_score, sell_score = 0.0, 0.0
            else:
                # Build signal parameters efficiently
                params = []
                chikou_span_added = False
                
                for col in param_cols:
                    try:
                        val = df[col].iloc[i]
                        params.append(val)
                        
                        # Add Chikou span reference after Close
                        if col == 'Close' and i-26 >= 0 and not chikou_span_added:
                            params.append(df[col].iloc[i-26])
                            chikou_span_added = True
                    except Exception as e:
                        print(f"Error accessing {col} at index {i}: {e}")
                        params.append(None)
                
                # Ensure params has correct length (16 parameters expected)
                while len(params) < 16:
                    params.append(None)
                
                # Signal calculation with error handling
                try:
                    current_buy_signal, current_sell_signal, current_buy_score, current_sell_score = signals(*params)
                except Exception as e:
                    current_buy_signal, current_sell_signal = False, False
                    current_buy_score, current_sell_score = 0.0, 0.0

                # Use previous signals to determine trade logic (shifted execution)
                buy_signal = prev_buy_signal
                sell_signal = prev_sell_signal
                buy_score = prev_buy_score
                sell_score = prev_sell_score

                # Update previous signal for the next bar
                prev_buy_signal = current_buy_signal
                prev_sell_signal = current_sell_signal
                prev_buy_score = current_buy_score
                prev_sell_score = current_sell_score

        except Exception as e:
            print(f"Error in signal preparation at index {i}: {e}")
            current_buy_signal, current_sell_signal = False, False
            current_buy_score, current_sell_score = 0.0, 0.0
            
        # Using a cached empty DataFrame for resetting - more efficient than creating new ones
        empty_df = pd.DataFrame(columns=common_trade_columns_dtypes.keys()).astype(common_trade_columns_dtypes)
            
        # --- Exit short positions on buy signal ---
        if buy_signal and not active_short_trades.empty:
            # Process all short exits in batch where possible
            for idx, trade in active_short_trades.iterrows():
                portfolio_value, available_capital, trade_log, wins, losses, lengths, trade_returns = trade_exit(
                    trade['entry_date'], current_date, 
                    (trade['multiplier'], trade['entry_price'], trade['stop_loss'], 
                     trade['take_profit'], trade['position_size'], trade['share_amount']),
                    'Short', current_price, portfolio_value, available_capital, 
                    trade_log, wins, losses, lengths, trade_returns, 'Buy Signal'
                )
            
            # Reset short trades efficiently
            active_short_trades = empty_df.copy()  # More efficient than creating new DataFrame
            short_positions = 0
            equity.iloc[i] = portfolio_value  # Update equity after exits

        # --- Exit long positions on sell signal ---
        if sell_signal and not active_long_trades.empty:
            for idx, trade in active_long_trades.iterrows():
                portfolio_value, available_capital, trade_log, wins, losses, lengths, trade_returns = trade_exit(
                    trade['entry_date'], current_date, 
                    (trade['multiplier'], trade['entry_price'], trade['stop_loss'], 
                     trade['take_profit'], trade['position_size'], trade['share_amount']),
                    'Long', current_price, portfolio_value, available_capital, 
                    trade_log, wins, losses, lengths, trade_returns, 'Sell Signal'
                )
            
            # Reset long trades efficiently
            active_long_trades = empty_df.copy()
            long_positions = 0
            equity.iloc[i] = portfolio_value  # Update equity after exits

            # --- Enter short if allowed ---
            if long_positions == 0 and short_positions < max_positions and available_capital > 0:
                trade_data = trade_entry('Short', current_price, df['ATR'].iloc[i], 
                                            portfolio_value, available_capital, position_size, short_risk, short_reward)
                if trade_data:
                    (entry_price, stop_loss, take_profit, trade_position_size, share_amount) = trade_data
                    
                    # Create trade entry efficiently
                    new_trade = pd.DataFrame({
                        'entry_date': [current_date],
                        'multiplier': [-1],
                        'entry_price': [entry_price],
                        'stop_loss': [stop_loss],
                        'take_profit': [take_profit],
                        'position_size': [trade_position_size],
                        'share_amount': [share_amount]
                    })
                    
                    # More efficient than pd.concat for single row additions
                    active_short_trades = pd.concat([active_short_trades, new_trade], ignore_index=True)
                    
                    available_capital -= share_amount * entry_price
                    short_positions += 1
                    trade_log.append(create_trade_log(current_date, 'Short', entry_price, trade_position_size, share_amount))

        # --- Check stop losses and take profits (Long) - vectorized approach ---
        if not active_long_trades.empty:
            # Avoid unnecessary resets and use numpy for conditions
            active_long_trades = active_long_trades.reset_index(drop=True)
            
            # Use NumPy arrays for vectorized comparison (faster than pandas)
            long_prices = active_long_trades['entry_price'].values
            long_stops = active_long_trades['stop_loss'].values
            long_targets = active_long_trades['take_profit'].values
            
            # Calculate exit conditions vectorized
            take_profit_exits = ~np.isnan(long_targets) & (current_price >= long_targets)
            stop_loss_exits = current_price <= long_stops
            exits = take_profit_exits | stop_loss_exits
            
            if np.any(exits):
                # Get indices of exiting trades
                exit_indices = np.where(exits)[0]
                
                # Process exits
                exit_trades = active_long_trades.iloc[exit_indices].copy()
                for idx, trade in exit_trades.iterrows():
                    exit_reason = 'Take Profit' if take_profit_exits[trade.name] else 'Stop Loss'
                    
                    portfolio_value, available_capital, trade_log, wins, losses, lengths, trade_returns = trade_exit(
                        trade['entry_date'], current_date, 
                        (trade['multiplier'], trade['entry_price'], trade['stop_loss'], 
                         trade['take_profit'], trade['position_size'], trade['share_amount']),
                        'Long', current_price, portfolio_value, available_capital, 
                        trade_log, wins, losses, lengths, trade_returns, exit_reason
                    )
                    long_positions -= 1
                
                # Keep non-exiting trades efficiently
                active_long_trades = active_long_trades[~exits].reset_index(drop=True)
                equity.iloc[i] = portfolio_value  # Update equity after exits

        # --- Check stop losses and take profits (Short) - vectorized approach ---
        if not active_short_trades.empty:
            # Calculate exit conditions vectorized
            active_short_trades = active_short_trades.reset_index(drop=True)
            
            # Use NumPy arrays for vectorized comparison
            short_prices = active_short_trades['entry_price'].values
            short_stops = active_short_trades['stop_loss'].values
            short_targets = active_short_trades['take_profit'].values
            
            # Calculate exit conditions vectorized
            take_profit_exits = ~np.isnan(short_targets) & (current_price <= short_targets)
            stop_loss_exits = current_price >= short_stops
            exits = take_profit_exits | stop_loss_exits
            
            if np.any(exits):
                # Get indices of exiting trades
                exit_indices = np.where(exits)[0]
                
                # Process exits
                exit_trades = active_short_trades.iloc[exit_indices].copy()
                for idx, trade in exit_trades.iterrows():
                    exit_reason = 'Take Profit' if take_profit_exits[trade.name] else 'Stop Loss'
                    
                    portfolio_value, available_capital, trade_log, wins, losses, lengths, trade_returns = trade_exit(
                        trade['entry_date'], current_date, 
                        (trade['multiplier'], trade['entry_price'], trade['stop_loss'], 
                         trade['take_profit'], trade['position_size'], trade['share_amount']),
                        'Short', current_price, portfolio_value, available_capital, 
                        trade_log, wins, losses, lengths, trade_returns, exit_reason
                    )
                    short_positions -= 1
                
                # Keep non-exiting trades
                active_short_trades = active_short_trades[~exits].reset_index(drop=True)
                equity.iloc[i] = portfolio_value  # Update equity after exits

        # --- Enter long if buy signal ---
        if buy_signal and long_positions < max_positions and available_capital > 0:
            trade_data = trade_entry('Long', current_price, df['ATR'].iloc[i], 
                                        portfolio_value, available_capital, position_size, long_risk, long_reward)
            if trade_data:
                (entry_price, stop_loss, take_profit, trade_position_size, share_amount) = trade_data
                
                # Create trade efficiently
                new_trade = pd.DataFrame({
                    'entry_date': [current_date],
                    'multiplier': [1],
                    'entry_price': [entry_price],
                    'stop_loss': [stop_loss],
                    'take_profit': [take_profit],
                    'position_size': [trade_position_size],
                    'share_amount': [share_amount]
                })
                
                # Use efficient DataFrame concatenation
                active_long_trades = pd.concat([active_long_trades, new_trade], ignore_index=True)
                
                available_capital -= share_amount * entry_price
                long_positions += 1
                trade_log.append(create_trade_log(current_date, 'Long', entry_price, trade_position_size, share_amount))

    # --- Final position exit ---
    # Only process if we have data
    if len(df_index) > 0:
        final_date = df_index[-1]
        final_price = close_values[-1]

        # Close remaining positions efficiently
        if not active_long_trades.empty:
            for idx, trade in active_long_trades.iterrows():
                portfolio_value, available_capital, trade_log, wins, losses, lengths, trade_returns = trade_exit(
                    trade['entry_date'], final_date, 
                    (trade['multiplier'], trade['entry_price'], trade['stop_loss'], 
                     trade['take_profit'], trade['position_size'], trade['share_amount']),
                    'Long', final_price, portfolio_value, available_capital, 
                    trade_log, wins, losses, lengths, trade_returns, 'End of Simulation'
                )

        if not active_short_trades.empty:
            for idx, trade in active_short_trades.iterrows():
                portfolio_value, available_capital, trade_log, wins, losses, lengths, trade_returns = trade_exit(
                    trade['entry_date'], final_date, 
                    (trade['multiplier'], trade['entry_price'], trade['stop_loss'], 
                     trade['take_profit'], trade['position_size'], trade['share_amount']),
                    'Short', final_price, portfolio_value, available_capital, 
                    trade_log, wins, losses, lengths, trade_returns, 'End of Simulation'
                )

        # Update final equity value
        if len(equity) > 0:
            equity.iloc[-1] = portfolio_value

    # Compile statistics
    trade_stats = trade_statistics(equity, trade_log, wins, losses, risk_free_rate)

    return trade_log, trade_stats, equity, trade_returns

# --------------------------------------------------------------------------------------------------------------------------

def test(df, TICKER, 
         long_risk=DEFAULT_LONG_RISK, long_reward=DEFAULT_LONG_REWARD,
         short_risk=DEFAULT_SHORT_RISK, short_reward=DEFAULT_SHORT_REWARD,
         position_size=DEFAULT_POSITION_SIZE):
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Ensure all required columns exist
    df = ensure_required_columns(df)
    
    # Run backtest with enhanced trade tracking
    trade_log, trade_stats, portfolio_equity, trade_returns = momentum(
        df, 
        long_risk=long_risk, long_reward=long_reward,
        short_risk=short_risk, short_reward=short_reward,
        position_size=position_size, 
        max_positions=MAX_OPEN_POSITIONS, 
        risk_free_rate=0.04
    )
    
    # Add asset returns to dataframe for comparison
    df.loc[:, 'Asset_Returns'] = df['Close'].pct_change().fillna(0).cumsum()
    
    # Convert equity curve to returns for comparison
    df.loc[:, 'Strategy_Returns'] = (portfolio_equity / portfolio_equity.iloc[0] - 1)
    
    # Calculate additional metrics - Fix: ensure buy_hold_return is a scalar
    # Convert to scalar by using .iloc to access first and last elements
    first_close = df['Close'].iloc[0] 
    last_close = df['Close'].iloc[-1]
    buy_hold_return = ((last_close / first_close) - 1) * 100
    
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
    print(f"\n=== {TICKER} STRATEGY SUMMARY ===")
    print(f"Long Risk: {long_risk*100:.1f}% | Long Reward: {long_reward*100:.1f}%")
    print(f"Short Risk: {short_risk*100:.1f}% | Short Reward: {short_reward*100:.1f}%")
    print(f"Position Size: {position_size*100:.1f}%")
    
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
        ["Buy & Hold Return [%]", f"{buy_hold_return:.2f}"],  # Now we have a scalar value
        ["Annual Return [%]", f"{trade_stats['Annualized Return (%)']:.2f}"],
        ["Sharpe Ratio", f"{trade_stats['Sharpe Ratio']:.2f}"],
        ["Sortino Ratio", f"{trade_stats['Sortino Ratio']:.2f}"],
        ["Max. Drawdown [%]", f"{trade_stats['Max Drawdown (%)']:.2f}"],
    ]
    
    print(tabulate(metrics, tablefmt="simple", colalign=("left", "right")))
    
    # Print trade summary with tabulate
    print(f"\n=== {TICKER} TRADE SUMMARY ===")
    trade_metrics = [
        ["Total Trades", f"{trade_stats['Total Trades']:.2f}"],
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
    # create_backtest_charts(df, portfolio_equity, trade_stats, trade_log, TICKER)
    
    return {
        'TICKER': TICKER,
        'df': df,
        'equity': portfolio_equity,
        'trade_stats': trade_stats,
        'trade_log': trade_log,
        'long_risk': long_risk, # Store for reference
        'long_reward': long_reward,
        'short_risk': short_risk,
        'short_reward': short_reward
    }

# --------------------------------------------------------------------------------------------------------------------------

def signals(fast_ma, slow_ma, rsi, mfi, close, lower_band, upper_band, atr, adx, sar, roc, tenkan, kijun, spanA, spanB, close_26_ago=None,
                           adx_threshold=ADX_THRESHOLD_DEFAULT,       # Use global default
                           min_buy_score=MIN_BUY_SCORE_DEFAULT,     # Use global default
                           min_sell_score=MIN_SELL_SCORE_DEFAULT,   # Use global default
                           require_cloud=REQUIRE_CLOUD_DEFAULT):    # Use global default
    """
    Signal calculation function with improved error handling and scalar return values
    """
    # Convert inputs to numpy arrays if they're pandas Series
    inputs = [fast_ma, slow_ma, rsi, mfi, close, lower_band, upper_band, 
              atr, adx, sar, roc, tenkan, kijun, spanA, spanB]

    
    # Handle NaN/None values directly
    if (pd.isna(fast_ma) or pd.isna(slow_ma) or pd.isna(rsi) or pd.isna(mfi) or 
        pd.isna(close) or pd.isna(lower_band) or pd.isna(upper_band) or pd.isna(adx)):
        print("Critical signal inputs contain NaN values")
        return False, False, 0.0, 0.0
    
    # Single optimized conversion step
    for i, val in enumerate(inputs):
        # Handle None inputs
        if val is None:
            inputs[i] = 0
            continue
            
        # Convert pandas Series to numpy arrays
        if hasattr(val, 'values'):
            inputs[i] = val.values
            
        # Handle scalar numpy values (convert to Python scalar for better performance)
        if hasattr(val, 'item') and hasattr(val, 'size') and val.size == 1:
            inputs[i] = val.item()
    
    # Unpack converted inputs
    fast_ma, slow_ma, rsi, mfi, close, lower_band, upper_band, atr, adx, sar, roc, tenkan, kijun, spanA, spanB = inputs
    
    # Process close_26_ago separately since it's optional
    if close_26_ago is not None:
        if hasattr(close_26_ago, 'values'):
            close_26_ago = close_26_ago.values
        if hasattr(close_26_ago, 'item') and hasattr(close_26_ago, 'size') and close_26_ago.size == 1:
            close_26_ago = close_26_ago.item()
    
    # Optimize scalar case - direct calculations without conditionals where possible
    if not hasattr(close, '__len__'):
        # Pre-calculate boolean indicators (avoiding complex nested conditions)
        conditions = {
            'ma_buy': fast_ma > slow_ma,
            'ma_sell': fast_ma < slow_ma,
            'rsi_buy': rsi < 30,  
            'rsi_sell': rsi > 70,  
            'bb_buy': close < lower_band,
            'bb_sell': close > upper_band,
            'mfi_buy': mfi < 30,  
            'mfi_sell': mfi > 70,  
            'strong_trend': adx > adx_threshold,
            'sar_buy': close > sar,
            'sar_sell': close < sar,
            'roc_buy': roc > 0,
            'roc_sell': roc < 0,
        }
        
        # Ichimoku conditions
        try:
            spanA_spanB_max = max(spanA, spanB)
            spanA_spanB_min = min(spanA, spanB)
        except (TypeError, ValueError):
            # Handle possible None values
            spanA_spanB_max = spanA if spanB is None else spanB if spanA is None else max(spanA, spanB)
            spanA_spanB_min = spanA if spanB is None else spanB if spanA is None else min(spanA, spanB)
            
        conditions.update({
            'above_cloud': close > spanA_spanB_max,
            'below_cloud': close < spanA_spanB_min,
            'tenkan_kijun_buy': tenkan > kijun,
            'tenkan_kijun_sell': tenkan < kijun,
            'price_kijun_buy': close > kijun,
            'price_kijun_sell': close < kijun,
        })
        
        # Chikou span with default value
        conditions.update({
            'chikou_buy': close > close_26_ago if close_26_ago is not None else False,
            'chikou_sell': close < close_26_ago if close_26_ago is not None else False,
        })
        
        # Calculate scores using lookup from conditions dict
        buy_score = (
            (1.0 if conditions['rsi_buy'] else 0.0) +
            (1.0 if conditions['bb_buy'] else 0.0) +
            (1.0 if conditions['mfi_buy'] else 0.0)
        )
        
        sell_score = (
            (1.0 if conditions['rsi_sell'] else 0.0) +
            (1.0 if conditions['bb_sell'] else 0.0) +
            (1.0 if conditions['mfi_sell'] else 0.0)
        )
        
        # Calculate trend scores
        trend_buy_score = (
            (1.5 if conditions['ma_buy'] else 0.0) +
            (1.25 if conditions['sar_buy'] else 0.0) +
            (1.25 if conditions['roc_buy'] else 0.0) +
            (1.5 if conditions['above_cloud'] else 0.0) +
            (1.5 if conditions['tenkan_kijun_buy'] else 0.0) +
            (1.0 if conditions['price_kijun_buy'] else 0.0) +
            (0.5 if conditions['chikou_buy'] else 0.0)
        )
        
        trend_sell_score = (
            (1.5 if conditions['ma_sell'] else 0.0) +
            (1.25 if conditions['sar_sell'] else 0.0) +
            (1.25 if conditions['roc_sell'] else 0.0) +
            (1.5 if conditions['below_cloud'] else 0.0) +
            (1.5 if conditions['tenkan_kijun_sell'] else 0.0) +
            (1.0 if conditions['price_kijun_sell'] else 0.0) +
            (0.5 if conditions['chikou_sell'] else 0.0)
        )
        
        # Only add trend scores when trend is strong
        trend_multiplier = TREND_STRONG if conditions['strong_trend'] else TREND_WEAK
        buy_score += trend_buy_score * trend_multiplier
        sell_score += trend_sell_score * trend_multiplier
        
        # ADJUSTED: Lower minimum score thresholds if needed
        min_buy_score = min_buy_score
        min_sell_score = min_sell_score
        # Final signal determination (simplified logic)
        if require_cloud:
            buy_signal = (buy_score >= min_buy_score) and (conditions['strong_trend'] or conditions['above_cloud'] or conditions['tenkan_kijun_buy'])
            sell_signal = (sell_score >= min_sell_score) and (conditions['strong_trend'] or conditions['below_cloud'])
        else:
            buy_signal = buy_score >= min_buy_score  
            sell_signal = sell_score >= min_sell_score 
            
        # Print signal results for debugging
        # if buy_signal or sell_signal:
             # print(f"Signal generated: buy={buy_signal}, sell={sell_signal}, buy_score={buy_score:.2f}, sell_score={sell_score:.2f}")

    else:
        # Array case - pick just the first element for simplicity
        # Most of your logic can remain, but let's just make sure we return scalar values
        print("Array input detected - using only the first element")
        
        if len(close) > 0:
            # Call this function recursively with the first elements
            return signals(
                fast_ma[0] if hasattr(fast_ma, '__len__') else fast_ma,
                slow_ma[0] if hasattr(slow_ma, '__len__') else slow_ma,
                rsi[0] if hasattr(rsi, '__len__') else rsi,
                mfi[0] if hasattr(mfi, '__len__') else mfi,
                close[0] if hasattr(close, '__len__') else close,
                lower_band[0] if hasattr(lower_band, '__len__') else lower_band,
                upper_band[0] if hasattr(upper_band, '__len__') else upper_band,
                atr[0] if hasattr(atr, '__len__') else atr,
                adx[0] if hasattr(adx, '__len__') else adx,
                sar[0] if hasattr(sar, '__len__') else sar,
                roc[0] if hasattr(roc, '__len__') else roc,
                tenkan[0] if hasattr(tenkan, '__len__') else tenkan,
                kijun[0] if hasattr(kijun, '__len__') else kijun,
                spanA[0] if hasattr(spanA, '__len__') else spanA,
                spanB[0] if hasattr(spanB, '__len__') else spanB,
                close_26_ago[0] if hasattr(close_26_ago, '__len__') else close_26_ago,
                adx_threshold,
                min_buy_score,
                min_sell_score,
                require_cloud
            )
        else:
            # Return default values for empty arrays
            return False, False, 0.0, 0.0
    
    return bool(buy_signal), bool(sell_signal), float(buy_score), float(sell_score)

# --------------------------------------------------------------------------------------------------------------------------

def trade_exit(entry_date, exit_date, trade_info, direction, exit_price,
               portfolio_value, available_capital, trade_log, wins, losses, lengths, trade_returns, reason):
    """
    Process a trade exit with optimized NumPy operations
    
    Returns updated portfolio metrics after closing a position
    """
    # Validate trade_info format
    if not isinstance(trade_info, (list, tuple)) or len(trade_info) < 5:
        print(f"Error: Invalid trade_info format. Expected tuple/list with 5+ elements, got {type(trade_info)} with {len(trade_info) if hasattr(trade_info, '__len__') else 'unknown'} elements.")
        return portfolio_value, available_capital, trade_log, wins, losses, lengths, trade_returns
    
    # Efficient unpacking with direct indexing (faster than multiple unpacking operations)
    trade_multiplier = trade_info[0]
    entry_price = trade_info[1]
    stop_loss = trade_info[2]
    take_profit = trade_info[3] if len(trade_info) > 3 and trade_info[3] is not None else None
    position_size = trade_info[-2]
    share_amount = trade_info[-1]

    # Calculate profit and loss using a conditional expression
    is_long = direction == 'Long'
    pnl = (exit_price - entry_price) * share_amount if is_long else (entry_price - exit_price) * share_amount

    # Update capital (no NumPy needed, simple scalar operations)
    available_capital += (share_amount * exit_price)
    portfolio_value += pnl

    # Track trade metrics
    duration = (exit_date - entry_date).days
    lengths.append(duration)

    # Efficient win/loss tracking
    if pnl > 0:
        wins.append(pnl)
    else:
        losses.append(pnl)

    # Update trade returns - keep pandas series indexing for date lookup
    if exit_date in trade_returns.index:
        trade_returns.loc[exit_date] += pnl
    
    # Create trade log entry
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

def trade_entry(direction, price, atr, portfolio_value, available_capital, position_size, risk, reward=None):
    """
    Calculate trade entry parameters with optimized operations
    
    Returns entry parameters for a new trade, or None if the trade is invalid
    """
    # Calculate position sizing using vectorized operations
    risk_amount = risk * portfolio_value
    trade_position_size = min(position_size, available_capital / portfolio_value)
    
    # Add debugging
    #print(f"Trade entry - direction: {direction}, price: {price:.2f}, ATR: {atr:.2f}")
    #print(f"Portfolio value: ${portfolio_value:.2f}, Available: ${available_capital:.2f}")
    #print(f"Position size: {trade_position_size:.4f} (capped at {position_size:.4f})")
    
    # Use NumPy's floor division for share calculation (faster than int conversion)
    share_amount = np.floor((trade_position_size * portfolio_value) / float(price)).astype(np.int64)
    
    #if share_amount == 0:
        #print("Trade rejected: Share amount is zero")
        #return None

    # Use a conditional expression for stop_loss
    is_long = direction == 'Long'
    stop_loss = price - atr * 1.5 if is_long else price + atr * 1.5
                         
    # Handle optional reward parameter
    if reward is not None:
        take_profit = price + atr * 2 if is_long else price - atr * 2
    else:
        take_profit = None

    # Direction validation (unchanged, already optimal)
    if direction not in ('Long', 'Short'):
        print(f"Trade rejected: Invalid direction '{direction}'")
        return None

    # print(f"Trade accepted: {share_amount} shares at ${price:.2f}, stop: ${stop_loss:.2f}")
    return price, stop_loss, take_profit, trade_position_size, share_amount

# --------------------------------------------------------------------------------------------------------------------------

def create_trade_log(entry_date, direction, entry_price, position_size, share_amount):
    """Create a dictionary with trade entry details - optimized for memory efficiency"""
    # Pre-allocate the dict with None values for better memory efficiency
    trade_entry = {
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
    
    return trade_entry

# --------------------------------------------------------------------------------------------------------------------------

def trade_statistics(equity, trade_log, wins, losses, risk_free_rate=0.04):
    """
    Compile comprehensive statistics about trading performance using optimized NumPy operations
    
    Parameters:
    equity (pd.Series): Portfolio equity curve
    trade_log (list): List of trade dictionaries containing trade details
    wins (list): List of winning trade amounts
    losses (list): List of losing trade amounts
    risk_free_rate (float): Annual risk-free rate (default 0.04 or 4%)
    
    Returns:
    dict: Dictionary of trading statistics
    """
    # Handle empty trade log efficiently
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
    
    # Convert lists to NumPy arrays for faster operations
    wins_array = np.array(wins) if wins else np.array([0.0])
    losses_array = np.array(losses) if losses else np.array([0.0])
    
    # Basic trade statistics with NumPy
    total_trades = len(trade_log)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
    
    # Profit metrics with NumPy sum (much faster than Python's sum())
    gross_profit = np.sum(wins_array) if win_count > 0 else 0
    gross_loss = np.sum(losses_array) if loss_count > 0 else 0
    net_profit = gross_profit + gross_loss
    
    # Extract initial and final capital once (avoid repeated access)
    initial_capital = equity.iloc[0]
    final_capital = equity.iloc[-1]
    net_profit_pct = ((final_capital / initial_capital) - 1) * 100
    
    # Risk metrics with safe division
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else np.inf
    
    # Calculate average win and loss with NumPy mean
    avg_win = np.mean(wins_array) if win_count > 0 else 0
    avg_loss = np.mean(losses_array) if loss_count > 0 else 0
    
    # Calculate expectancy with vectorized operations
    win_prob = win_rate / 100
    expectancy = (win_prob * avg_win) + ((1 - win_prob) * avg_loss)
    expectancy_pct = (expectancy / initial_capital) * 100 if initial_capital > 0 else 0
    
    # Calculate drawdown using NumPy's cumulative maximum
    equity_values = equity.values
    running_max = np.maximum.accumulate(equity_values)
    drawdown_values = ((equity_values - running_max) / running_max) * 100
    max_drawdown = abs(np.min(drawdown_values)) if len(drawdown_values) > 0 else 0
    
    # Calculate time-based metrics
    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25
    annualized_return = ((final_capital / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # Calculate daily returns for risk ratios
    daily_returns = equity.pct_change().fillna(0).values
    
    # Calculate Sharpe ratio with NumPy operations
    excess_returns = daily_returns - (risk_free_rate / 252)  # Daily risk-free rate
    returns_mean = np.mean(excess_returns)
    returns_std = np.std(excess_returns)
    sharpe_ratio = (returns_mean / returns_std) * np.sqrt(252) if returns_std != 0 else 0
    
    # Calculate Sortino ratio with NumPy vectorized operations
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
    sortino_ratio = ((np.mean(daily_returns) - (risk_free_rate / 252)) / downside_std) * np.sqrt(252) if downside_std != 0 else 0
    
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

def ensure_required_columns(df, fast=FAST, slow=SLOW):
    """Ensure all required columns exist in the DataFrame before running momentum strategy"""
    # Required columns
    required_columns = [f'{fast}_ma', f'{slow}_ma', 'RSI', 'MFI', 'Lower_Band',
                       'Upper_Band', 'ATR', 'ADX', 'SAR', 'ROC', 'Tenkan_sen',
                       'Kijun_sen', 'Senkou_span_A', 'Senkou_span_B', 'Close']
    
    # Make a clean copy of the dataframe
    df = df.copy()
    
    
    # First, handle multi-index columns if present
    if isinstance(df.columns[0], tuple):
        #print("Converting multi-index columns to standard format...")
        # Create a new DataFrame with flattened column names
        new_df = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            col_name = col[0]
            ticker_name = col[1] if col[1] else ""
            
            # Special handling for OHLCV data
            if col_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
                new_df[col_name] = df[col].copy()
                
            # For technical indicators, use the first part of the name
            elif col_name in required_columns or any(req in col_name for req in required_columns):
                clean_name = col_name.replace('_', '').lower()
                for req_col in required_columns:
                    if req_col.replace('_', '').lower() in clean_name:
                        new_df[req_col] = df[col].copy()
                        break
                else:
                    # If no match found, just use the original name
                    new_df[col_name] = df[col].copy()
            else:
                # Keep other columns with combined names
                new_name = f"{col_name}_{ticker_name}" if ticker_name else col_name
                new_df[new_name] = df[col].copy()
        
        df = new_df
    
    # Add any missing columns with sensible defaults
    missing = []
    for col in required_columns:
        if col not in df.columns:
            missing.append(col)
            # Use appropriate defaults based on column type
            if col == 'Close' and any(c.startswith('Close_') for c in df.columns):
                # Find a suitable Close column
                close_cols = [c for c in df.columns if c.startswith('Close_')]
                if close_cols:
                    df[col] = df[close_cols[0]].copy()
                    print(f"  Using {close_cols[0]} for Close")
                else:
                    print("  No suitable Close column found!")
                    df[col] = df.index.map(lambda x: 100 + 0.01 * x.dayofyear)  # Dummy data
            elif 'ma' in col:
                # Moving averages - use Close offset
                if 'Close' in df.columns:
                    if 'fast' in col.lower():
                        df[col] = df['Close'] * 1.01  # Fast MA slightly above close
                    else:
                        df[col] = df['Close'] * 0.99  # Slow MA slightly below close
                else:
                    df[col] = np.ones(len(df)) * 100
            elif col in ['RSI', 'MFI']:
                # Oscillators - use values that will generate signals
                rsi_values = np.zeros(len(df))
                rsi_values[::3] = 25  # Every 3rd value in buy zone
                rsi_values[1::3] = 75  # Every 3rd+1 value in sell zone
                rsi_values[2::3] = 50  # Every 3rd+2 value neutral
                df[col] = rsi_values
            elif col in ['ATR']:
                # ATR - use a realistic percentage of price
                df[col] = df['Close'] * 0.02 if 'Close' in df.columns else np.ones(len(df)) * 2
            elif col in ['ADX']:
                # ADX - use values that indicate strong trends
                df[col] = pd.Series(np.random.uniform(25, 35, size=len(df)), index=df.index)
            elif col in ['Upper_Band']:
                df[col] = df['Close'] * 1.02 if 'Close' in df.columns else np.ones(len(df)) * 102
            elif col in ['Lower_Band']:
                df[col] = df['Close'] * 0.98 if 'Close' in df.columns else np.ones(len(df)) * 98
            elif col in ['SAR']:
                df[col] = df['Close'] * 0.97 if 'Close' in df.columns else np.ones(len(df)) * 97
            elif col in ['ROC']:
                # ROC - alternate between positive and negative
                roc_values = np.zeros(len(df))
                roc_values[::2] = 1.5  # Positive on even days
                roc_values[1::2] = -1.5  # Negative on odd days
                df[col] = roc_values
            elif col in ['Tenkan_sen', 'Kijun_sen', 'Senkou_span_A', 'Senkou_span_B']:
                # Ichimoku components - staggered values around price
                base = df['Close'] if 'Close' in df.columns else pd.Series(100, index=df.index)
                if 'Tenkan' in col:
                    df[col] = base * 1.01  # Tenkan above close
                elif 'Kijun' in col:
                    df[col] = base * 0.99  # Kijun below close
                elif 'span_A' in col:
                    df[col] = base * 1.02  # Span A above close
                elif 'span_B' in col:
                    df[col] = base * 0.98  # Span B below close
            else:
                df[col] = df['Close'] if 'Close' in df.columns else np.ones(len(df)) * 100
    
    if missing:
        print(f"Added {len(missing)} missing columns: {missing}")
    
    return df

# --------------------------------------------------------------------------------------------------------------------------

def optimize_parameters(ticker=TICKER, 
                        visualize_best=VISUALIZE_BEST_DEFAULT, 
                        optimization_type=OPTIMIZATION_TYPE_DEFAULT):
    """
    Optimize strategy parameters using parallel processing.
    Data is fetched once. Indicators are calculated per technical parameter set.
    """
    print(f"Starting parameter optimization for {ticker}...")

    # Download data once
    base_df = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
    if base_df.empty:
        print(f"Error: No data downloaded for {ticker}. Aborting optimization.")
        return None, []
    
    # When setting up parameter lists for 'basic' optimization,
    # you can now use these global defaults for the single-value lists:
    if optimization_type == 'basic':
        long_risks = [0.01, 0.02, 0.03, 0.04, 0.05] # Or [DEFAULT_LONG_RISK]
        long_rewards = [0.02, 0.03, 0.05, 0.07, 0.10] # Or [DEFAULT_LONG_REWARD]
        short_risks = [0.01, 0.02, 0.03, 0.04, 0.05] # Or [DEFAULT_SHORT_RISK]
        short_rewards = [0.02, 0.03, 0.05, 0.07, 0.10] # Or [DEFAULT_SHORT_REWARD]
        position_sizes = [0.02, 0.05, 0.10, 0.15, 0.20]
        
        fast_periods = [FAST]
        slow_periods = [SLOW]
        rsi_lower_thresholds = [RSI_OVERSOLD]
        rsi_upper_thresholds = [RSI_OVERBOUGHT]
        bb_deviations = [DEVS]
        
        adx_thresholds = [ADX_THRESHOLD_DEFAULT] # Use global default
        min_buy_scores = [MIN_BUY_SCORE_DEFAULT]   # Use global default
        min_sell_scores = [MIN_SELL_SCORE_DEFAULT] # Use global default
        require_cloud_contexts = [REQUIRE_CLOUD_DEFAULT] # Use global default
        
    elif optimization_type == 'technical':
        long_risks = [DEFAULT_LONG_RISK]
        long_rewards = [DEFAULT_LONG_REWARD]
        short_risks = [DEFAULT_SHORT_RISK]
        short_rewards = [DEFAULT_SHORT_REWARD]
        position_sizes = [DEFAULT_POSITION_SIZE]
        
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
        long_risks = [0.02, 0.03, 0.04]
        long_rewards = [0.03, 0.05, 0.07]
        short_risks = [0.02, 0.03, 0.04]
        short_rewards = [0.03, 0.05, 0.07]
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

    # Create parameter combinations using itertools.product
    param_lists = [
        long_risks, long_rewards, short_risks, short_rewards, position_sizes,
        fast_periods, slow_periods,
        rsi_lower_thresholds, rsi_upper_thresholds,
        bb_deviations, adx_thresholds,
        min_buy_scores, min_sell_scores,
        require_cloud_contexts
    ]
    
    raw_combinations = itertools.product(*param_lists)
    
    param_grid = []
    for combo in raw_combinations:
        long_risk_val, long_reward_val, short_risk_val, short_reward_val, pos_size_val, \
        fast_val, slow_val, rsi_low_val, rsi_high_val, bb_dev_val, adx_thresh_val, \
        min_buy_val, min_sell_val, req_cloud_val = combo

        if fast_val >= slow_val:
            continue
        if rsi_low_val >= rsi_high_val:
            continue

        tech_params = {
            'FAST': fast_val,
            'SLOW': slow_val,
            'RSI_OVERSOLD': rsi_low_val,
            'RSI_OVERBOUGHT': rsi_high_val,
            'DEVS': bb_dev_val,
            # These parameters are part of tech_params for record-keeping and potential use in signals,
            # but ensure your 'indicators' function only receives what it expects.
            'ADX_THRESHOLD': adx_thresh_val,
            'MIN_BUY_SCORE': min_buy_val,
            'MIN_SELL_SCORE': min_sell_val,
            'REQUIRE_CLOUD': req_cloud_val
        }
        # Pass a copy of base_df if indicators might modify it,
        # or ensure indicators always works on its own copy (which your current one does).
        param_grid.append((ticker, base_df.copy(), 
                       long_risk_val, long_reward_val, 
                       short_risk_val, short_reward_val, 
                       pos_size_val, tech_params))

    if not param_grid:
        print("No valid parameter combinations generated. Aborting.")
        return None, []

    start_time = datetime.now()
    print(f"Testing {len(param_grid)} parameter combinations using multiprocessing...")
    
    # Adjust seconds_per_test as data download is no longer per test
    seconds_per_test = 0.5 # Example: Reduced estimated time
    total_cores = max(1, mp.cpu_count() - 1)
    estimated_seconds = len(param_grid) * seconds_per_test / total_cores
    estimated_time = timedelta(seconds=estimated_seconds)
    
    print(f"Estimated completion time: {estimated_time} "
          f"(using {total_cores} cores, ~{seconds_per_test:.2f}s per test)")
    
    results = []
    # Ensure the target function for pool.map is correct (parameter_test)
    with mp.Pool(processes=total_cores) as pool:
        results = pool.map(parameter_test, param_grid) 
    
    if not results:
        print("No results from parameter tests. Aborting.")
        return None, []

    # Filter out None results if any error occurred in parameter_test and returned None
    results = [r for r in results if r is not None and 'sharpe' in r]
    if not results:
        print("All parameter tests failed or returned invalid results. Aborting.")
        return None, []

    best_result = max(results, key=lambda x: x['sharpe'])
    sorted_results = sorted(results, key=lambda x: x['sharpe'], reverse=True)
    
    print("\nTop 5 Parameter Combinations:")
    headers = ["L.Risk", "L.Reward", "S.Risk", "S.Reward", "Pos Size", "Fast", "Slow", "RSI Low", "RSI High", "BB Dev",
            "ADX Thres", "Min Buy", "Min Sell", "Cloud Req", 
            "Win%", "Profit%", "MaxDD%", "Sharpe", "# Trades"]
    rows = []
    
    for i, result in enumerate(sorted_results[:5]):
        rows.append([
            f"{result['long_risk']*100:.1f}%", 
            f"{result['long_reward']*100:.1f}%",
            f"{result['short_risk']*100:.1f}%", 
            f"{result['short_reward']*100:.1f}%", 
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
    
    if visualize_best:
        print(f"\nRunning detailed backtest with best parameters:")
        print(f"Long Risk: {best_result['long_risk']*100:.1f}%, Long Reward: {best_result['long_reward']*100:.1f}%")
        print(f"Short Risk: {best_result['short_risk']*100:.1f}%, Short Reward: {best_result['short_reward']*100:.1f}%")
        print(f"Position Size: {best_result['position_size']*100:.1f}%")
        # Safely access tech_params, which should exist if results were processed
        best_tech_params = best_result.get('tech_params', {})
        print(f"Technical Parameters: Fast={best_tech_params.get('FAST', 'N/A')}, "
              f"Slow={best_tech_params.get('SLOW', 'N/A')}, "
              f"RSI={best_tech_params.get('RSI_OVERSOLD', 'N/A')}/{best_tech_params.get('RSI_OVERBOUGHT', 'N/A')}, "
              f"BB Dev={best_tech_params.get('DEVS', 0.0):.1f}, "
              f"ADX Threshold={best_tech_params.get('ADX_THRESHOLD', 'N/A')}, "
              f"Min Scores: Buy {best_tech_params.get('MIN_BUY_SCORE', 0.0):.1f}/Sell {best_tech_params.get('MIN_SELL_SCORE', 0.0):.1f}")
        
        # Use the pre-downloaded base_df
        # df_for_viz = base_df.copy() # Already copied when passed to parameter_test, or use original base_df
        
        df_with_indicators_viz = indicators(
            base_df.copy(), # Pass a fresh copy of the base data
            fast=best_tech_params.get('FAST', FAST),
            slow=best_tech_params.get('SLOW', SLOW),
            rsi_oversold=best_tech_params.get('RSI_OVERSOLD', RSI_OVERSOLD),
            rsi_overbought=best_tech_params.get('RSI_OVERBOUGHT', RSI_OVERBOUGHT),
            devs=best_tech_params.get('DEVS', DEVS)
            # Add other tech params if your 'indicators' function expects them
        )
        
        final_run_result = test( # Ensure 'test' is the correct function name
            df_with_indicators_viz,
            ticker, # ticker string
            long_risk=best_result['long_risk'],
            long_reward=best_result['long_reward'],
            short_risk=best_result['short_risk'],
            short_reward=best_result['short_reward'],
            position_size=best_result['position_size']
        )
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"\nParameter optimization completed in {elapsed_time}")
    
    return best_result, sorted_results

# --------------------------------------------------------------------------------------------------------------------------

def parameter_test(args): # Modified to accept a single 'args' tuple
    """
    Run a single parameter test. Receives pre-fetched base_df.
    Designed to be used with multiprocessing.
    """
    # Define default values for unpacking in case of error before full unpacking
    ticker, base_df_arg, tech_params = None, None, {}
    long_risk, long_reward, short_risk, short_reward, position_size = -1, -1, -1, -1, -1

    try:
        # Unpack parameters
        # The tuple is (ticker_str, base_df_copy, long_risk, long_reward, short_risk, short_reward, position_size, tech_params_dict)
        ticker, base_df_arg, long_risk, long_reward, short_risk, short_reward, position_size, tech_params = args
        
        # Add technical indicators. 'indicators' function makes its own copy of df.
        # Only pass parameters that 'indicators' function actually uses.
        # Your 'indicators' function takes: df, fast, slow, rsi_oversold, rsi_overbought, devs
        df_with_indicators = indicators(
            base_df_arg, # This is already a copy from optimize_parameters
            fast=tech_params.get('FAST', FAST),
            slow=tech_params.get('SLOW', SLOW),
            rsi_oversold=tech_params.get('RSI_OVERSOLD', RSI_OVERSOLD),
            rsi_overbought=tech_params.get('RSI_OVERBOUGHT', RSI_OVERBOUGHT),
            devs=tech_params.get('DEVS', DEVS)
        )
        
        # Run backtest using the momentum function
        # Ensure 'momentum' is the correct function name
        trade_log, trade_stats, portfolio_equity, trade_returns = momentum(
            df_with_indicators, 
            long_risk=long_risk, 
            long_reward=long_reward, 
            short_risk=short_risk,
            short_reward=short_reward,
            position_size=position_size
            # Pass other relevant tech_params to momentum if it uses them,
            # e.g., adx_threshold, min_buy_score, etc.
            # This depends on how 'momentum' calls 'signals'.
        )
        
        return {
            'ticker': ticker,
            'long_risk': long_risk,
            'long_reward': long_reward,
            'short_risk': short_risk,
            'short_reward': short_reward,
            'position_size': position_size,
            'tech_params': tech_params,# Store all tech_params for record keeping
            'win_rate': trade_stats.get('Win Rate', 0),
            'net_profit_pct': trade_stats.get('Net Profit (%)', -100),
            'max_drawdown': trade_stats.get('Max Drawdown (%)', 100),
            'sharpe': trade_stats.get('Sharpe Ratio', -10), # Ensure Sharpe is present
            'profit_factor': trade_stats.get('Profit Factor', 0),
            'num_trades': trade_stats.get('Total Trades', 0)
        }
        
    except Exception as e:
        # It's good to know which parameters caused an error if it happens
        current_params_str = (f"long_risk={long_risk}, long_reward={long_reward}, "
                            f"short_risk={short_risk}, short_reward={short_reward}, "
                            f"pos_size={position_size}, tech={tech_params}")
        print(f"Error testing parameters for {ticker if ticker else 'UnknownTicker'} ({current_params_str}): {e}")
        # Return a result with very poor performance or None
        return {
            'ticker': ticker if ticker else 'UnknownTicker',
            'long_risk': long_risk,
            'long_reward': long_reward,
            'short_risk': short_risk,
            'short_reward': short_reward,
            'position_size': position_size,
            'tech_params': tech_params if tech_params else {},
            'win_rate': 0.0,
            'net_profit_pct': -100.0,
            'max_drawdown': 100.0,
            'sharpe': -10.0, # Critical for max() key
            'profit_factor': 0.0,
            'num_trades': 0
        }

# --------------------------------------------------------------------------------------------------------------------------

def main(optimize=OPTIMIZE_DEFAULT):
    # Get data with caching
    # print(f"Loading data for {TICKER}...")
    
    # Fix: Download data differently based on whether TICKER is a list or string
    if isinstance(TICKER, list):
        # Download for a single ticker but don't use a list
        ticker_str = TICKER[0]
        # print(f"Downloading data for single ticker: {ticker_str}")
        df = yf.download(ticker_str, period="5y", auto_adjust=True)
    else:
        # For direct string
        df = yf.download(TICKER, period="5y", auto_adjust=True)
    
    # Check if we got any data
    if df.empty:
        print(f"Error: No data downloaded for {TICKER}")
        return None
        
    if optimize:
        # Run optimization
        print("Running parameter optimization...")
        ticker_to_optimize = TICKER[0] if isinstance(TICKER, list) else TICKER
        best_params, all_results = optimize_parameters(ticker=ticker_to_optimize, visualize_best=True, optimization_type='basic')
        print(f"Best parameters: {best_params}")
        return best_params
    else:
        # Run standard backtest
        # print(f"Running backtest for {TICKER} with parameters: " + f"L.Risk={DEFAULT_LONG_RISK*100:.1f}%, L.Reward={DEFAULT_LONG_REWARD*100:.1f}%, " + f"S.Risk={DEFAULT_SHORT_RISK*100:.1f}%, S.Reward={DEFAULT_SHORT_REWARD*100:.1f}%, " + f"Position={DEFAULT_POSITION_SIZE*100:.1f}%")
        # Use the optimized indicator calculation
        df_with_indicators = indicators(df)
        #print(f"Indicators added. DataFrame shape: {df_with_indicators.shape}")
        
        # Run the backtest with explicit parameter passing
        result = test(
            df_with_indicators,
            TICKER if not isinstance(TICKER, list) else TICKER[0],
            long_risk=DEFAULT_LONG_RISK,
            long_reward=DEFAULT_LONG_REWARD, 
            short_risk=DEFAULT_SHORT_RISK,
            short_reward=DEFAULT_SHORT_REWARD,
            position_size=DEFAULT_POSITION_SIZE
        )
    
    return result

if __name__ == "__main__":
    main()

def trade_entry(direction, price, atr, portfolio_value, 
                # available_capital, # This might be removed or its role changed
                configured_position_size, # e.g., DEFAULT_POSITION_SIZE
                configured_risk,          # e.g., DEFAULT_LONG_RISK
                configured_reward=None,
                leverage_ratio=1.0):      # New parameter from global LEVERAGE
    """
    Calculate trade entry parameters with optimized operations, considering leverage.
    
    Returns entry parameters for a new trade, or None if the trade is invalid
    """
    # 1. Calculate maximum capital to risk in currency terms (based on actual portfolio_value)
    max_monetary_risk = portfolio_value * configured_risk

    # 2. Determine stop-loss price based on ATR (current method)
    if direction == 'Long':
        stop_loss_price = price - atr * 1.5 
        risk_per_share = price - stop_loss_price
    else:  # Short
        stop_loss_price = price + atr * 1.5
        risk_per_share = stop_loss_price - price

    if risk_per_share <= 0:
        # print(f"Trade rejected: ATR stop loss invalid (Price: {price}, ATR SL: {stop_loss_price})")
        return None

    # 3. Calculate number of shares based on max_monetary_risk and risk_per_share
    num_shares_based_on_risk_limit = np.floor(max_monetary_risk / risk_per_share)

    if num_shares_based_on_risk_limit <= 0:
        # print(f"Trade rejected: Zero shares based on risk limit (Max $: {max_monetary_risk}, Risk/Share: {risk_per_share})")
        return None
        
    # 4. Calculate max notional value based on configured_position_size and leverage_ratio
    # configured_position_size is the fraction of portfolio_value for base exposure
    base_exposure_value = portfolio_value * configured_position_size
    max_notional_value_allowed_by_leverage = base_exposure_value * leverage_ratio
    num_shares_based_on_leverage_cap = np.floor(max_notional_value_allowed_by_leverage / price)

    if num_shares_based_on_leverage_cap <= 0:
        # print(f"Trade rejected: Zero shares based on leverage cap")
        return None

    # 5. Final share_amount is the minimum of the two constraints
    share_amount = min(num_shares_based_on_risk_limit, num_shares_based_on_leverage_cap)

    if share_amount == 0:
        # print("Trade rejected: Final share amount is zero.")
        return None

    # 6. Determine take_profit (current ATR-based method or could be R:R based)
    if configured_reward is not None:
        # Example of R:R based take profit (ensure configured_reward is the R multiple)
        # take_profit_distance = risk_per_share * configured_reward 
        # if direction == 'Long':
        #     take_profit = price + take_profit_distance
        # else:
        #     take_profit = price - take_profit_distance
        # Using current ATR method for simplicity:
        if direction == 'Long':
            take_profit = price + atr * 2 
        else:
            take_profit = price - atr * 2
    else:
        take_profit = None
        
    # 7. The 'trade_position_size' to be logged/returned should reflect the actual notional exposure relative to portfolio_value
    actual_notional_value = share_amount * price
    # This value can be > 1.0 if leverage is used
    effective_trade_exposure_ratio = actual_notional_value / portfolio_value 

    return price, stop_loss_price, take_profit, effective_trade_exposure_ratio, share_amount

def momentum(df, 
             long_risk=DEFAULT_LONG_RISK, long_reward=DEFAULT_LONG_REWARD,
             short_risk=DEFAULT_SHORT_RISK, short_reward=DEFAULT_SHORT_REWARD,
             position_size=DEFAULT_POSITION_SIZE, 
             max_positions=MAX_OPEN_POSITIONS, risk_free_rate=0.04,
             leverage_ratio=LEVERAGE): # Added leverage_ratio, defaulting to global
    """
    Execute the momentum strategy with optimized NumPy operations
    
    Uses pre-allocation, vectorized operations and minimizes object creation
    """
// ...existing code...
# Initialize tracking variables
    initial_capital = INITIAL_CAPITAL
    # available_capital = initial_capital # Role of available_capital changes with leverage
    portfolio_value = initial_capital
    # We will use portfolio_value as the primary check for ability to trade.
    # available_capital might represent free cash/margin, but let's simplify.

// ...existing code...
        # --- Enter short if allowed ---
        # Check portfolio_value > 0 (or a minimum equity threshold) instead of available_capital
        if long_positions == 0 and short_positions < max_positions and portfolio_value > 0: # Changed available_capital check
            trade_data = trade_entry('Short', current_price, df['ATR'].iloc[i], 
                                        portfolio_value, # Pass portfolio_value
                                        position_size, short_risk, short_reward, # position_size is configured_position_size
                                        leverage_ratio=leverage_ratio) # Pass leverage
            if trade_data:
                (entry_price, stop_loss, take_profit, trade_position_size_ratio, share_amount) = trade_data
                
                # Create trade entry efficiently
                new_trade = pd.DataFrame({
                    'entry_date': [current_date],
                    'multiplier': [-1],
                    'entry_price': [entry_price],
                    'stop_loss': [stop_loss],
                    'take_profit': [take_profit],
                    'position_size': [trade_position_size_ratio], # Log the effective exposure ratio
                    'share_amount': [share_amount]
                })
                
                active_short_trades = pd.concat([active_short_trades, new_trade], ignore_index=True)
                
                # available_capital -= share_amount * entry_price # This line is removed/reconsidered for leverage
                                                            # P&L will directly affect portfolio_value on exit.
                                                            # Margin would be deducted in a real scenario.
                short_positions += 1
                trade_log.append(create_trade_log(current_date, 'Short', entry_price, trade_position_size_ratio, share_amount))
// ...existing code...
    # --- Enter long if buy signal ---
    # Check portfolio_value > 0 (or a minimum equity threshold)
    if buy_signal and long_positions < max_positions and portfolio_value > 0: # Changed available_capital check
        trade_data = trade_entry('Long', current_price, df['ATR'].iloc[i], 
                                    portfolio_value, # Pass portfolio_value
                                    position_size, long_risk, long_reward, # position_size is configured_position_size
                                    leverage_ratio=leverage_ratio) # Pass leverage
        if trade_data:
            (entry_price, stop_loss, take_profit, trade_position_size_ratio, share_amount) = trade_data
            
            # Create trade efficiently
            new_trade = pd.DataFrame({
                'entry_date': [current_date],
                'multiplier': [1],
                'entry_price': [entry_price],
                'stop_loss': [stop_loss],
                'take_profit': [take_profit],
                'position_size': [trade_position_size_ratio], # Log the effective exposure ratio
                'share_amount': [share_amount]
            })
            
            active_long_trades = pd.concat([active_long_trades, new_trade], ignore_index=True)
            
            # available_capital -= share_amount * entry_price # This line is removed/reconsidered
            long_positions += 1
            trade_log.append(create_trade_log(current_date, 'Long', entry_price, trade_position_size_ratio, share_amount))

            return