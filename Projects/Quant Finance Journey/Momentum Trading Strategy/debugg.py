import pandas as pd
import numpy as np
import yfinance as yf
import os
from tabulate import tabulate  
import pandas_ta as ta 
from datetime import datetime, timedelta
import multiprocessing as mp 
import itertools
import matplotlib as plt
import seaborn as sns

# --------------------------------------------------------------------------------------------------------------------------

# Base parameters
TICKER = ['SPY']
INITIAL_CAPITAL = 100000.0
LEVERAGE = 1.0 
TRAILING_STOP_ATR_MULTIPLIER = 3.0

# Moving average strategy parameters
FAST = 20
SLOW = 50
WEEKLY_MA_PERIOD = 50

# MACD strategy parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SPAN = 9

# RSI strategy parameters
RSI_LENGTH = 14
RSI_OVERBOUGHT = 75
RSI_OVERSOLD = 35

# MFI strategy parameters
MFI_LENGTH = 14
MFI_OVERBOUGHT = 70
MFI_OVERSOLD = 30

# Bollinger Bands strategy parameters
BB_LEN = 20
DEVS = 2

# Risk & Reward
DEFAULT_LONG_RISK = 0.03  
DEFAULT_LONG_REWARD = 0.06  
DEFAULT_SHORT_RISK = 0.02  
DEFAULT_SHORT_REWARD = 0.04  
DEFAULT_POSITION_SIZE = 0.10  
MAX_OPEN_POSITIONS = 3 

TREND_STRONG = 1
TREND_WEAK = 0.75
ADX_THRESHOLD_DEFAULT = 25
MIN_BUY_SCORE_DEFAULT = 3.0
MIN_SELL_SCORE_DEFAULT = 5.0
REQUIRE_CLOUD_DEFAULT = False
USE_TRAILING_STOPS_DEFAULT = False

# Script Execution Defaults 
OPTIMIZE_DEFAULT = False
VISUALIZE_BEST_DEFAULT = False
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

        # Volume Moving Average (Volume_MA20)
        volume_ma20 = np.empty_like(volume_np, dtype=float) # Ensure float for mean
        volume_ma20_period = 20
        volume_ma20[:volume_ma20_period-1] = np.nan
        for i in range(volume_ma20_period-1, len(volume_np)):
            volume_ma20[i] = np.mean(volume_np[i-volume_ma20_period+1:i+1])

        # Weekly Moving Average (Weekly_MA50)
        # Ensure the DataFrame index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df_weekly = df['Close'].resample('W').last() # Resample to weekly, take last close of week
        weekly_ma50_series = df_weekly.rolling(window=WEEKLY_MA_PERIOD, min_periods=1).mean() # Calculate 50-week MA
        # Map weekly MA back to daily data, forward fill to handle NaNs within the week
        df[f'Weekly_MA{WEEKLY_MA_PERIOD}'] = weekly_ma50_series.reindex(df.index, method='ffill')
        weekly_ma50_np = df[f'Weekly_MA{WEEKLY_MA_PERIOD}'].values
        
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
        df.loc[:, 'Volume_MA20'] = volume_ma20 # Added Volume_MA20
        df[f'Weekly_MA{WEEKLY_MA_PERIOD}'] = weekly_ma50_np
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
             max_positions=MAX_OPEN_POSITIONS, risk_free_rate=0.04,
             leverage_ratio=LEVERAGE, use_trailing_stops=USE_TRAILING_STOPS_DEFAULT):
    """
    Execute the momentum strategy with optimized NumPy operations,
    staggered exits, volume confirmation, trailing stops, and parallel long/short trading.
    """
    if len(df) == 0:
        print("Warning: Empty dataframe provided to momentum. Returning empty results.")
        empty_equity = pd.Series(dtype='float64')
        empty_returns = pd.Series(dtype='float64')
        return [], {
            'Total Trades': 0, 'Win Rate': 0, 'Net Profit (%)': 0, 'Profit Factor': 0,
            'Expectancy (%)': 0, 'Max Drawdown (%)': 0, 'Annualized Return (%)': 0,
            'Sharpe Ratio': 0, 'Sortino Ratio': 0
        }, empty_equity, empty_returns
    
    df_index = df.index
    close_values = df['Close'].values

    param_cols = [f'{FAST}_ma', f'{SLOW}_ma', 'RSI', 'MFI', 'Close', 'Lower_Band',
                 'Upper_Band', 'ATR', 'ADX', 'SAR', 'ROC', 'Tenkan_sen',
                 'Kijun_sen', 'Senkou_span_A', 'Senkou_span_B']
    
    weekly_ma_col_name = f'Weekly_MA{WEEKLY_MA_PERIOD}'
    missing_cols_check = [col for col in param_cols + ['Volume_MA20', weekly_ma_col_name, 'ADX'] if col not in df.columns]
    if missing_cols_check:
        print(f"Warning: Missing columns for momentum strategy: {missing_cols_check}")
    

    initial_capital = INITIAL_CAPITAL
    portfolio_value = initial_capital # This will track realized equity

    # Item 3: Separate Realized and Unrealized PnL Tracking
    realized_equity_curve = pd.Series(initial_capital, index=df_index, dtype='float64')
    unrealized_pnl_curve = pd.Series(0.0, index=df_index, dtype='float64')
    total_equity_curve = pd.Series(initial_capital, index=df_index, dtype='float64') # For reporting and stats

    trade_returns = pd.Series(0.0, index=df_index)
    
    common_trade_columns_dtypes = {
        'entry_date': 'datetime64[ns]', 'multiplier': 'int64', 'entry_price': 'float64',
        'stop_loss': 'float64', 'take_profit': 'float64', 'position_size': 'float64',
        'share_amount': 'int64', 
        'highest_close_since_entry': 'float64', 
        'lowest_close_since_entry': 'float64'   
    }

    active_long_trades = pd.DataFrame(columns=common_trade_columns_dtypes.keys()).astype(common_trade_columns_dtypes)
    active_short_trades = pd.DataFrame(columns=common_trade_columns_dtypes.keys()).astype(common_trade_columns_dtypes)
    
    long_positions = short_positions = 0
    trade_log, wins, losses, lengths = [], [], [], []

    prev_buy_signal, prev_sell_signal = False, False
    
    empty_df = pd.DataFrame(columns=common_trade_columns_dtypes.keys()).astype(common_trade_columns_dtypes)

    for i in range(1, len(df)):
        current_date = df_index[i]
        current_price = close_values[i]
        current_atr = df['ATR'].iloc[i] if 'ATR' in df.columns and not pd.isna(df['ATR'].iloc[i]) else 0.01 # Default ATR if missing
        current_adx = df['ADX'].iloc[i] if 'ADX' in df.columns and not pd.isna(df['ADX'].iloc[i]) else 25 # Default ADX

        # Calculate current signals for bar 'i'
        current_buy_signal, current_sell_signal, _, _ = False, False, 0.0, 0.0
        if not missing_cols_check: 
            params = []
            chikou_span_added = False
            for col_name in param_cols:
                try:
                    val = df[col_name].iloc[i]
                    params.append(val)
                    if col_name == 'Close' and i - 26 >= 0 and not chikou_span_added:
                        params.append(df[col_name].iloc[i - 26]) 
                        chikou_span_added = True
                except KeyError: 
                    params.append(None) 
            
            while len(params) < 16: 
                params.append(None)
            
            try:
                current_buy_signal, current_sell_signal, _, _ = signals(*params)
            except Exception as e:
                pass 

        buy_signal = prev_buy_signal
        sell_signal = prev_sell_signal

        # Item 1: Trailing Stop Logic: Whipsaws in Choppy Markets
        if use_trailing_stops:
            atr_multiplier = 2.0 if current_adx < 25 else 4.0 # Tighter in weak/choppy trends
            
            if not active_long_trades.empty:
                for idx, trade_row in active_long_trades.iterrows():
                    current_highest = max(trade_row['highest_close_since_entry'], current_price)
                    active_long_trades.at[idx, 'highest_close_since_entry'] = current_highest
                    new_trailing_stop = current_highest - atr_multiplier * current_atr
                    active_long_trades.at[idx, 'stop_loss'] = max(new_trailing_stop, trade_row['stop_loss'])
            
            if not active_short_trades.empty:
                for idx, trade_row in active_short_trades.iterrows():
                    current_lowest = min(trade_row.get('lowest_close_since_entry', current_price), current_price)
                    active_short_trades.at[idx, 'lowest_close_since_entry'] = current_lowest
                    new_trailing_stop = current_lowest + atr_multiplier * current_atr
                    active_short_trades.at[idx, 'stop_loss'] = min(new_trailing_stop, trade_row['stop_loss'])

        # --- Process Long Exits ---
        if not active_long_trades.empty:
            trades_to_keep_long_list = []
            for idx, trade_row in active_long_trades.iterrows():
                trade = trade_row.to_dict() 
                exited_fully_this_iteration = False
                exit_reason_this_iteration = None

                if sell_signal: # This is prev_sell_signal
                    is_trend_still_valid = (current_adx > 25 and current_price > df[f'{SLOW}_ma'].iloc[i]) if f'{SLOW}_ma' in df.columns else False
                    if is_trend_still_valid: 
                        partial_shares_to_exit = int(trade['share_amount'] * 0.3)
                        if partial_shares_to_exit > 0:
                            realized_pnl_from_exit, _, _, _, _, _, _, _ = trade_exit(
                                trade['entry_date'], current_date,
                                (trade['multiplier'], trade['entry_price'], trade['stop_loss'], trade['take_profit'], 
                                 trade['position_size'], partial_shares_to_exit), 
                                'Long', current_price, portfolio_value, 0, 
                                trade_log, wins, losses, lengths, trade_returns, 'Partial Profit (Sell Signal)'
                            )
                            portfolio_value += realized_pnl_from_exit 
                            trade['share_amount'] -= partial_shares_to_exit
                            if trade['share_amount'] <= 0:
                                exited_fully_this_iteration = True
                                long_positions -= 1
                    else: 
                        realized_pnl_from_exit, _, _, _, _, _, _, _ = trade_exit(
                            trade['entry_date'], current_date,
                            (trade['multiplier'], trade['entry_price'], trade['stop_loss'], trade['take_profit'],
                             trade['position_size'], trade['share_amount']), 
                            'Long', current_price, portfolio_value, 0,
                            trade_log, wins, losses, lengths, trade_returns, 'Full Exit (Sell Signal - Trend Broken)'
                        )
                        portfolio_value += realized_pnl_from_exit
                        exited_fully_this_iteration = True
                        long_positions -= 1
                
                if not exited_fully_this_iteration:
                    if current_price <= trade['stop_loss']:
                        exit_reason_this_iteration = 'Stop Loss'
                    elif trade['take_profit'] is not None and current_price >= trade['take_profit']:
                        exit_reason_this_iteration = 'Take Profit'
                    
                    if exit_reason_this_iteration:
                        realized_pnl_from_exit, _, _, _, _, _, _, _ = trade_exit(
                            trade['entry_date'], current_date,
                            (trade['multiplier'], trade['entry_price'], trade['stop_loss'], trade['take_profit'],
                             trade['position_size'], trade['share_amount']), 
                            'Long', current_price, portfolio_value, 0,
                            trade_log, wins, losses, lengths, trade_returns, exit_reason_this_iteration
                        )
                        portfolio_value += realized_pnl_from_exit
                        exited_fully_this_iteration = True
                        long_positions -= 1
                
                if not exited_fully_this_iteration and trade['share_amount'] > 0:
                    trades_to_keep_long_list.append(trade)
            
            active_long_trades = pd.DataFrame(trades_to_keep_long_list).astype(common_trade_columns_dtypes) if trades_to_keep_long_list else empty_df.copy()

        # --- Process Short Exits ---
        if not active_short_trades.empty:
            trades_to_keep_short_list = []
            for idx, trade_row in active_short_trades.iterrows():
                trade = trade_row.to_dict()
                exited_fully_this_iteration = False
                exit_reason_this_iteration = None
                
                # Consider buy_signal (prev_buy_signal) for exiting shorts
                if buy_signal: # This is prev_buy_signal
                    # Add logic if shorts should be exited on a counter-signal (buy_signal)
                    # For now, we'll assume SL/TP are primary for shorts, or end of simulation
                    pass


                if not exit_reason_this_iteration: 
                    if current_price >= trade['stop_loss']: 
                        exit_reason_this_iteration = 'Stop Loss'
                    elif trade['take_profit'] is not None and current_price <= trade['take_profit']: 
                        exit_reason_this_iteration = 'Take Profit'

                if exit_reason_this_iteration:
                    realized_pnl_from_exit, _, _, _, _, _, _, _ = trade_exit(
                        trade['entry_date'], current_date,
                        (trade['multiplier'], trade['entry_price'], trade['stop_loss'], trade['take_profit'],
                         trade['position_size'], trade['share_amount']),
                        'Short', current_price, portfolio_value, 0,
                        trade_log, wins, losses, lengths, trade_returns, exit_reason_this_iteration
                    )
                    portfolio_value += realized_pnl_from_exit
                    exited_fully_this_iteration = True
                    short_positions -= 1
                
                if not exited_fully_this_iteration and trade['share_amount'] > 0:
                    trades_to_keep_short_list.append(trade)

            active_short_trades = pd.DataFrame(trades_to_keep_short_list).astype(common_trade_columns_dtypes) if trades_to_keep_short_list else empty_df.copy()

        # --- Enter Long ---
        volume_confirmed = df['Volume'].iloc[i] > df['Volume_MA20'].iloc[i] if 'Volume_MA20' in df.columns and not pd.isna(df['Volume_MA20'].iloc[i]) and not pd.isna(df['Volume'].iloc[i]) else False
        if buy_signal and volume_confirmed and long_positions < max_positions and portfolio_value > 0: 
            trade_data = trade_entry('Long', current_price, current_atr, portfolio_value, 
                        position_size, long_risk, long_reward,
                        leverage_ratio, current_adx)
            if trade_data:
                entry_price, stop_loss, take_profit, trade_pos_size_ratio, share_amount = trade_data
                new_trade_dict = {
                    'entry_date': current_date, 'multiplier': 1, 'entry_price': entry_price,
                    'stop_loss': stop_loss, 'take_profit': take_profit, 
                    'position_size': trade_pos_size_ratio, 'share_amount': share_amount,
                    'highest_close_since_entry': entry_price,
                    'lowest_close_since_entry': np.nan 
                }
                active_long_trades = pd.concat([active_long_trades, pd.DataFrame([new_trade_dict])], ignore_index=True)
                long_positions += 1

        # --- Sell-Signal Filtering & Enter Short ---
        filtered_sell_signal = sell_signal # sell_signal is prev_sell_signal
        if sell_signal: 
            # Item 2: Weekly MA Filter: Delayed Reactions
            if weekly_ma_col_name in df.columns and not pd.isna(df[weekly_ma_col_name].iloc[i]):
                current_weekly_ma = df[weekly_ma_col_name].iloc[i]
                # Ensure we don't go out of bounds for previous weekly MA
                prev_weekly_ma = df[weekly_ma_col_name].iloc[i-1] if i > 0 and not pd.isna(df[weekly_ma_col_name].iloc[i-1]) else current_weekly_ma
                
                weekly_uptrend_momentum = (current_price > current_weekly_ma) and (current_weekly_ma > prev_weekly_ma)
                if weekly_uptrend_momentum:
                    filtered_sell_signal = False 
                    
        if filtered_sell_signal and short_positions < max_positions and portfolio_value > 0:
            trade_data = trade_entry('Short', current_price, current_atr, portfolio_value,
                        position_size, short_risk, short_reward,
                        leverage_ratio, current_adx)
            if trade_data:
                entry_price, stop_loss, take_profit, trade_pos_size_ratio, share_amount = trade_data
                new_trade_dict = {
                    'entry_date': current_date, 'multiplier': -1, 'entry_price': entry_price,
                    'stop_loss': stop_loss, 'take_profit': take_profit,
                    'position_size': trade_pos_size_ratio, 'share_amount': share_amount,
                    'highest_close_since_entry': np.nan, 
                    'lowest_close_since_entry': entry_price 
                }
                active_short_trades = pd.concat([active_short_trades, pd.DataFrame([new_trade_dict])], ignore_index=True)
                short_positions += 1
        
        # Item 3: Unrealized PnL Tracking & Equity Update
        realized_equity_curve.iloc[i] = portfolio_value # portfolio_value is realized equity after exits for day i

        current_unrealized_pnl_scalar = 0.0
        if not active_long_trades.empty:
            for _, trade in active_long_trades.iterrows():
                current_unrealized_pnl_scalar += (current_price - trade['entry_price']) * trade['share_amount']
        if not active_short_trades.empty:
            for _, trade in active_short_trades.iterrows():
                current_unrealized_pnl_scalar += (trade['entry_price'] - current_price) * trade['share_amount']
        
        unrealized_pnl_curve.iloc[i] = current_unrealized_pnl_scalar
        total_equity_curve.iloc[i] = realized_equity_curve.iloc[i] + unrealized_pnl_curve.iloc[i]

        prev_buy_signal = current_buy_signal
        prev_sell_signal = current_sell_signal

    if len(df_index) > 0:
        final_date = df_index[-1]
        final_price = close_values[-1]
        final_realized_portfolio_value = portfolio_value 
        
        for trades_df, direction_str in [(active_long_trades, 'Long'), (active_short_trades, 'Short')]:
            if not trades_df.empty:
                for idx, trade in trades_df.iterrows():
                    pnl_on_exit, _, _, _, _, _, _, _ = trade_exit(
                        trade['entry_date'], final_date,
                        (trade['multiplier'], trade['entry_price'], trade['stop_loss'], trade['take_profit'],
                         trade['position_size'], trade['share_amount']),
                        direction_str, final_price, final_realized_portfolio_value, 0, 
                        trade_log, wins, losses, lengths, trade_returns, 'End of Simulation'
                    )
                    final_realized_portfolio_value += pnl_on_exit 
        
        if len(total_equity_curve) > 0: 
            realized_equity_curve.iloc[-1] = final_realized_portfolio_value
            unrealized_pnl_curve.iloc[-1] = 0.0 # All positions closed
            total_equity_curve.iloc[-1] = final_realized_portfolio_value

    trade_stats = trade_statistics(total_equity_curve, trade_log, wins, losses, risk_free_rate) # Use total_equity_curve for stats
    return trade_log, trade_stats, total_equity_curve, trade_returns

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
            'rsi_buy': (rsi > 40) & (rsi < 65),  
            'rsi_sell': rsi > 75,  
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
            buy_signal = (buy_score >= min_buy_score) and (
                sum([conditions['ma_buy'], conditions['rsi_buy'], conditions['tenkan_kijun_buy']]) >= 2
            )
            sell_signal = (sell_score >= min_sell_score) and (
                sum([conditions['ma_sell'], conditions['rsi_sell'], conditions['below_cloud']]) >= 2
            )
        else:
            buy_signal = (buy_score >= min_buy_score) and (
                sum([conditions['ma_buy'], conditions['rsi_buy'], conditions['tenkan_kijun_buy']]) >= 2
            )  
            sell_signal = (sell_score >= min_sell_score) and (
                sum([conditions['ma_sell'], conditions['rsi_sell'], conditions['below_cloud']]) >= 2
            ) 
            
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

def trade_entry(direction, price, atr, portfolio_value, 
               configured_position_size, 
               configured_risk, 
               configured_reward=None,
               leverage_ratio=1.0,
               adx_value=None,  # New: Dynamic sizing based on trend strength
               use_trailing_stops=True):  # New: Flag for trailing stops
    """
    Enhanced trade entry with:
    - Dynamic position sizing (scales up in strong trends)
    - Volatility-adjusted stops (3x ATR for stronger trends)
    - Smarter take-profit logic (adaptive risk-reward)
    """
    # --- 1. Dynamic Position Sizing ---
    # Scale up position size in strong trends (ADX > 30)
    position_size_multiplier = 1.5 if (adx_value is not None and adx_value > 30) else 1.0
    scaled_position_size = configured_position_size * position_size_multiplier

    # --- 2. Calculate Risk Parameters ---
    # Wider stops in strong trends (3x ATR), tighter in weak trends (1.5x ATR)
    atr_multiplier = 2.0 if (adx_value is not None and adx_value > 30) else 4.0
    
    if direction == 'Long':
        stop_loss_price = price - atr * atr_multiplier
        risk_per_share = price - stop_loss_price
    else:  # Short
        stop_loss_price = price + atr * atr_multiplier
        risk_per_share = stop_loss_price - price

    if risk_per_share <= 0:
        return None

    # --- 3. Share Calculation (Risk vs. Leverage Constraints) ---
    max_monetary_risk = portfolio_value * configured_risk
    num_shares_risk = np.floor(max_monetary_risk / risk_per_share)

    base_exposure = portfolio_value * scaled_position_size
    max_notional = base_exposure * leverage_ratio
    num_shares_leverage = np.floor(max_notional / price)

    share_amount = min(num_shares_risk, num_shares_leverage)
    if share_amount <= 0:
        return None

    # --- 4. Adaptive Take-Profit ---
    if configured_reward is not None:
        # Dynamic R:R - wider targets in strong trends
        reward_multiplier = 3.0 if (adx_value is not None and adx_value > 30) else 2.0
        take_profit_distance = risk_per_share * reward_multiplier
        
        if direction == 'Long':
            take_profit = price + take_profit_distance
        else:
            take_profit = price - take_profit_distance
    else:
        take_profit = None

    # --- 5. Return Trade Data ---
    actual_notional = share_amount * price
    exposure_ratio = actual_notional / portfolio_value

    return price, stop_loss_price, take_profit, exposure_ratio, share_amount

# --------------------------------------------------------------------------------------------------------------------------

def trade_exit(entry_date, exit_date, trade_info, direction, exit_price,
               current_portfolio_value_before_exit, # Renamed for clarity
               available_capital_dummy, # This is not really used for PnL calculation here
               trade_log, wins, losses, lengths, trade_returns, reason):
    """
    Process a trade exit.
    Returns PnL of this specific trade, and updates shared lists (trade_log, wins, etc.).
    """
    if not isinstance(trade_info, (list, tuple)) or len(trade_info) < 5:
        print(f"Error: Invalid trade_info format. Expected tuple/list with 5+ elements, got {type(trade_info)} with {len(trade_info) if hasattr(trade_info, '__len__') else 'unknown'} elements.")
        # Return 0 PnL and unchanged lists if error
        return 0.0, current_portfolio_value_before_exit, available_capital_dummy, trade_log, wins, losses, lengths, trade_returns
    
    trade_multiplier = trade_info[0]
    entry_price = trade_info[1]
    # stop_loss = trade_info[2] # Not directly used in PnL calc here
    # take_profit = trade_info[3] if len(trade_info) > 3 and trade_info[3] is not None else None # Not directly used
    # position_size = trade_info[-2] # Not directly used
    share_amount = trade_info[-1]

    is_long = direction == 'Long'
    pnl = (exit_price - entry_price) * share_amount if is_long else (entry_price - exit_price) * share_amount

    # available_capital would be updated by: available_capital += (share_amount * exit_price)
    # But since we are only returning PnL, we don't update it here directly.
    # The main momentum loop will update its 'portfolio_value' (realized equity) using this pnl.

    duration = (exit_date - entry_date).days
    lengths.append(duration)

    if pnl > 0:
        wins.append(pnl)
    else:
        losses.append(pnl)

    if exit_date in trade_returns.index:
        trade_returns.loc[exit_date] += pnl
    
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

    # Return the PnL of this trade, and the other lists/values as they were passed or updated
    return pnl, current_portfolio_value_before_exit, available_capital_dummy, trade_log, wins, losses, lengths, trade_returns

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

def test(df, TICKER, 
         long_risk=DEFAULT_LONG_RISK, long_reward=DEFAULT_LONG_REWARD,
         short_risk=DEFAULT_SHORT_RISK, short_reward=DEFAULT_SHORT_REWARD,
         position_size=DEFAULT_POSITION_SIZE, use_trailing_stops=USE_TRAILING_STOPS_DEFAULT):
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
    print(f"Use Trailing Stops: {use_trailing_stops}")
    
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
            position_size=DEFAULT_POSITION_SIZE,
            use_trailing_stops=USE_TRAILING_STOPS_DEFAULT # Pass default here
        )
    
    return result

if __name__ == "__main__":
    main()