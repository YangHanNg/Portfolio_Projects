import pandas as pd
import numpy as np
try:
    import cupy as cp
    # Test GPU availability
    cp.cuda.runtime.getDeviceCount()
    HAS_GPU = True
    print("GPU acceleration enabled")
except Exception as e:
    HAS_GPU = False
    print(f"GPU acceleration not available: {e}")
    print("Using CPU only")
import yfinance as yf
from tabulate import tabulate  
import pandas_ta as ta 
from datetime import datetime, timedelta
import multiprocessing as mp 
import itertools
import matplotlib as plt
import seaborn as sns
from collections import deque
from numba import jit
from tqdm import tqdm

# --------------------------------------------------------------------------------------------------------------------------


# Script Execution Defaults 
OPTIMIZE_DEFAULT = True
VISUALIZE_BEST_DEFAULT = False
OPTIMIZATION_TYPE_DEFAULT = 'basic'

# Trade Management
ADX_THRESHOLD_DEFAULT = 25
MIN_BUY_SCORE_DEFAULT = 3.0
MIN_SELL_SCORE_DEFAULT = 3.0
REQUIRE_CLOUD_DEFAULT = True
USE_TRAILING_STOPS_DEFAULT = True

# Risk & Reward
DEFAULT_LONG_RISK = 0.03  
DEFAULT_LONG_REWARD = 0.09  
DEFAULT_SHORT_RISK = 0.02  
DEFAULT_SHORT_REWARD = 0.04  
DEFAULT_POSITION_SIZE = 1 
MAX_OPEN_POSITIONS = 10 

TREND_WEIGHTS = {
    'primary_trend': 4.0,
    'trend_strength': 3.5,
    'cloud_confirmation': 2.0,
    'kumo_confirmation': 1.0,
    'ma_alignment': 2.0  
}

CONFIRMATION_WEIGHTS = {
    'rsi': 1.2,
    'mfi': 1.0,
    'volume': 1.5,
    'bollinger': 0.8
}

# Base parameters
TICKER = ['SPY']
INITIAL_CAPITAL = 100000.0
LEVERAGE = 1.0 

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

# --------------------------------------------------------------------------------------------------------------------------

def prepare_data(df_original, fast=FAST, 
                 slow=SLOW, rsi_oversold=RSI_OVERSOLD, rsi_overbought=RSI_OVERBOUGHT, 
                 devs=DEVS, weekly_ma_period_val=WEEKLY_MA_PERIOD):
    """
    Prepares data for momentum strategy with optimized memory usage and parallel calculations where possible.
    """
    # 1. Initial Setup & Memory Optimization
    df = df_original.copy()
    if df.empty:
        print("Warning: Empty dataframe provided to prepare_data.")
        return df
    
    # More efficient dtype conversion
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in ohlcv_cols:
        if col in df.columns:
            df[col] = df[col].astype('float32')
    
    # Replace this with more targeted conversion
    numeric_cols = df.select_dtypes(include=['float64']).columns
    df[numeric_cols] = df[numeric_cols].astype('float32')
    
    if len(df) > 100000:  # Only for large datasets
        text_cols = df.select_dtypes(include=['object']).columns
        df[text_cols] = df[text_cols].astype('category')
    
    # 2. Multi-index Handling Simplification
    if isinstance(df.columns, pd.MultiIndex):
        # Assuming the primary column names are in the first level (level 0)
        # and we want to drop the second level (level 1).
        df.columns = df.columns.droplevel(1)
        # Ensure no duplicate column names after droplevel, take the first occurrence
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # 3. Ensure Base Columns
    core_ohlcv_names = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in core_ohlcv_names:
        if col not in df.columns:
            if col == 'Close':
                df[col] = pd.Series(100.0, index=df.index, dtype='float32')
            elif col == 'Open':
                df[col] = df['Close'].copy() if 'Close' in df else pd.Series(100.0, index=df.index, dtype='float32')
            elif col == 'High':
                df[col] = df['Close'] * 1.001 if 'Close' in df else pd.Series(101.0, index=df.index, dtype='float32')
            elif col == 'Low':
                df[col] = df['Close'] * 0.999 if 'Close' in df else pd.Series(99.0, index=df.index, dtype='float32')
            elif col == 'Volume':
                df[col] = pd.Series(10000.0, index=df.index, dtype='float32')
    
    # 4. Calculate Core Indicators
    try:
        # Vectorized Moving Averages
        df[f'{fast}_ma'] = df['Close'].rolling(window=fast, min_periods=1).mean()
        df[f'{slow}_ma'] = df['Close'].rolling(window=slow, min_periods=1).mean()
        df['Volume_MA20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        
        # Weekly Moving Average
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df_weekly = df['Close'].resample('W').last()
        df[f'Weekly_MA{weekly_ma_period_val}'] = (
            df_weekly.rolling(window=weekly_ma_period_val, min_periods=1)
            .mean().reindex(df.index, method='ffill')
        )
        
        # RSI using pandas_ta (faster than manual)
        df['RSI'] = ta.rsi(df['Close'], length=RSI_LENGTH).fillna(50.0)
        
        # MFI using pandas_ta
        df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], 
                          length=MFI_LENGTH).fillna(50.0)
        
        # Bollinger Bands using pandas_ta
        bb = ta.bbands(df['Close'], length=BB_LEN, std=devs)
        if bb is not None and not bb.empty:
            df['Upper_Band'] = bb[f'BBU_{BB_LEN}_{float(devs)}'].fillna(df['Close'] * 1.02)
            df['Lower_Band'] = bb[f'BBL_{BB_LEN}_{float(devs)}'].fillna(df['Close'] * 0.98)
        else:
            df['Upper_Band'] = df['Close'] * 1.02
            df['Lower_Band'] = df['Close'] * 0.98

        # Calculate Secondary Indicators
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14).fillna(df['Close'] * 0.02)

        psar_result = ta.psar(df['High'], df['Low'])
        if isinstance(psar_result, pd.DataFrame) and not psar_result.empty:
            # Try to find a column that contains 'PSAR'
            psar_col = next((col for col in psar_result.columns if 'PSAR' in col), None)
            if psar_col:
                df['SAR'] = psar_result[psar_col].fillna(df['Close'] * 0.97)
            else:
                df['SAR'] = df['Close'] * 0.97 # fallback
        else:
            df['SAR'] = df['Close'] * 0.97  # fallback value

        df['ROC'] = ta.roc(df['Close'], length=12).fillna(0)
        df['Close_26_ago'] = df['Close'].shift(26).ffill().fillna(df['Close'])

        adx_result = ta.adx(df['High'], df['Low'], df['Close'])
        if isinstance(adx_result, pd.DataFrame) and 'ADX_14' in adx_result.columns:
            df['ADX'] = adx_result['ADX_14'].fillna(25.0)
        else:
            df['ADX'] = pd.Series(25.0, index=df.index)

        # Volume and Weekly Trend Confirmations
        df['volume_confirmed'] = df['Volume'] > df['Volume_MA20']
        df['weekly_uptrend'] = (df['Close'] > df[f'Weekly_MA{weekly_ma_period_val}']) & \
                       (df[f'Weekly_MA{weekly_ma_period_val}'].shift(1).ffill() < \
                        df[f'Weekly_MA{weekly_ma_period_val}'])
        
        # Ichimoku Cloud with vectorized calculations
        try:
            high, low, close = df['High'], df['Low'], df['Close']
            
            # Pre-calculate all periods needed
            periods = {
                'conv': (9, (high.rolling(9, min_periods=1).max() + low.rolling(9, min_periods=1).min()) / 2),
                'base': (26, (high.rolling(26, min_periods=1).max() + low.rolling(26, min_periods=1).min()) / 2),
                'span_b_calc': (52, (high.rolling(52, min_periods=1).max() + low.rolling(52, min_periods=1).min()) / 2) # Renamed to avoid clash
            }
            
            # Then build all components from these
            df['Tenkan_sen'] = periods['conv'][1]
            df['Kijun_sen'] = periods['base'][1]
            df['Senkou_span_A'] = ((periods['conv'][1] + periods['base'][1]) / 2).shift(26)
            df['Senkou_span_B'] = periods['span_b_calc'][1].shift(26)
            df['Chikou_span'] = close.shift(-26)

            # Fill NaN values for Ichimoku components
            cloud_defaults = {
                'Tenkan_sen': df['Close'] * 1.01,
                'Kijun_sen': df['Close'] * 0.99,
                'Senkou_span_A': df['Close'] * 1.02, # Default if all else fails after shift
                'Senkou_span_B': df['Close'] * 0.98, # Default if all else fails after shift
                'Chikou_span': df['Close'] # Default if all else fails after shift
            }
            for col, default_val_series in cloud_defaults.items():
                df[col] = df[col].fillna(default_val_series)

        except Exception as e_cloud:
            print(f"Error calculating Ichimoku Cloud: {e_cloud}. Using defaults.")
            df['Tenkan_sen'] = df['Close'] * 1.01
            df['Kijun_sen'] = df['Close'] * 0.99
            df['Senkou_span_A'] = df['Close'] * 1.02
            df['Senkou_span_B'] = df['Close'] * 0.98
            df['Chikou_span'] = df['Close']
        
    except Exception as e:
        print(f"Error calculating indicators: {e}. Applying defaults.")
        # This block will be superseded by the new default handling below if indicators fail broadly

    # 5. Validate and Fill Missing (New Default Value Handling)
    # Ensure 'Close' exists before defining defaults that depend on it.
    if 'Close' not in df.columns: # Should have been created in step 3, but defensive
        df['Close'] = 100.0 

    default_values = {
        f'{fast}_ma': df['Close'],
        f'{slow}_ma': df['Close'],
        'RSI': 50.0,
        'MFI': 50.0,
        'Upper_Band': df['Close'] * 1.02,
        'Lower_Band': df['Close'] * 0.98,
        'ATR': df['Close'] * 0.02,
        'ADX': 25.0,
        'SAR': df['Close'] * 0.97,
        'ROC': 0.0,
        'Tenkan_sen': df['Close'] * 1.01,
        'Kijun_sen': df['Close'] * 0.99,
        'Senkou_span_A': df['Close'] * 1.02,
        'Senkou_span_B': df['Close'] * 0.98,
        'Chikou_span': df['Close'],
        'Volume_MA20': df['Volume'] if 'Volume' in df else 10000.0, # Default if volume itself is missing
        f'Weekly_MA{weekly_ma_period_val}': df['Close'],
        'Close_26_ago': df['Close']
    }
    
    # Add core OHLCV defaults here too, in case they weren't handled or an error occurred before step 3
    # This makes the default handling more robust.
    if 'Open' not in df.columns: df['Open'] = df['Close']
    if 'High' not in df.columns: df['High'] = df['Close'] * 1.001
    if 'Low' not in df.columns: df['Low'] = df['Close'] * 0.999
    if 'Volume' not in df.columns: df['Volume'] = 10000.0


    for col_name, default_series_or_value in default_values.items():
        if col_name not in df.columns or df[col_name].isnull().all(): # If col is missing or all NaN
            df[col_name] = default_series_or_value
        else: # If column exists but has some NaNs, fill them
            df[col_name] = df[col_name].fillna(default_series_or_value)


    # 6. Final Cleanup
    df = df.dropna() # Drop rows where essential data might still be missing after fill attempts
    if df.empty and not df_original.empty:
        print("Warning: DataFrame became empty after processing.")
    
    return df

# --------------------------------------------------------------------------------------------------------------------------

def momentum(
    df_with_indicators,  
    fast_ma_col_name: str, 
    slow_ma_col_name: str, 
    weekly_ma_col_name: str,
    long_risk=DEFAULT_LONG_RISK, long_reward=DEFAULT_LONG_REWARD,
    short_risk=DEFAULT_SHORT_RISK, short_reward=DEFAULT_SHORT_REWARD,
    position_size=DEFAULT_POSITION_SIZE, max_positions=MAX_OPEN_POSITIONS, # Modified to accept max_positions
    risk_free_rate=0.04, leverage_ratio=LEVERAGE,
    use_trailing_stops=USE_TRAILING_STOPS_DEFAULT, # Modified to accept use_trailing_stops
    adx_threshold_sig=ADX_THRESHOLD_DEFAULT,
    min_buy_score_sig=MIN_BUY_SCORE_DEFAULT,
    min_sell_score_sig=MIN_SELL_SCORE_DEFAULT,
    require_cloud_sig=REQUIRE_CLOUD_DEFAULT,
    trend_weights_sig=None, 
    confirmation_weights_sig=None
):
    
    """
    Optimized momentum strategy execution.
    Assumes df_with_indicators has been processed by prepare_data().
    Internally calls vectorized_signals().
    """
    if df_with_indicators.empty:
        print("Warning: Empty dataframe (df_with_indicators) provided to momentum.")
        return [], create_empty_stats(), pd.Series(dtype='float64'), pd.Series(dtype='float64')

    final_trend_weights = trend_weights_sig if trend_weights_sig is not None else TREND_WEIGHTS
    final_confirmation_weights = confirmation_weights_sig if confirmation_weights_sig is not None else CONFIRMATION_WEIGHTS

    # 1. Generate signals using the indicator-laden DataFrame
    signals_df = signals(
        df_with_indicators, 
        adx_threshold_val=adx_threshold_sig,
        min_buy_score_val=min_buy_score_sig,
        min_sell_score_val=min_sell_score_sig,
        require_cloud_val=require_cloud_sig,
        fast_ma_col=fast_ma_col_name,
        slow_ma_col=slow_ma_col_name,
        weekly_ma_col_name=weekly_ma_col_name,
        trend_weights_dict_param=final_trend_weights, 
        confirmation_weights_dict_param=final_confirmation_weights
    )

    # 2. Initialize Trade Managers and Tracking
    # Ensure INITIAL_CAPITAL is accessible or passed if not a global
    long_manager = TradeManager(INITIAL_CAPITAL/2, max_positions, 'Long')
    short_manager = TradeManager(INITIAL_CAPITAL/2, max_positions, 'Short')

    # Use the index from df_with_indicators
    equity_curve = pd.Series(INITIAL_CAPITAL, index=df_with_indicators.index)
    returns_series = pd.Series(0.0, index=df_with_indicators.index)
    if not df_with_indicators.index.empty:
        equity_curve.iloc[0] = INITIAL_CAPITAL
        returns_series.iloc[0] = 0.0
    else: # Should not happen if df_with_indicators is not empty, but defensive
        print("Warning: df_with_indicators index is empty in momentum.")
        return [], create_empty_stats(), pd.Series(dtype='float64'), pd.Series(dtype='float64')


    # 3. Main Processing Loop
    # Iterate using df_with_indicators for data and signals_df for signals
    loop_start_time = datetime.now()
    for i in range(1, len(df_with_indicators)):
        current_trade_date = df_with_indicators.index[i]
        transaction_price = df_with_indicators['Open'].iloc[i]
        
        # Critical data checks for the current row
        # Data for decision making (ATR, ADX, confirmations) is from the PREVIOUS day's close
        previous_day_data_row = df_with_indicators.iloc[i-1]
        previous_day_atr = previous_day_data_row['ATR']
        previous_day_adx = previous_day_data_row['ADX']

        buy_signal_for_today = signals_df['buy_signal'].iloc[i-1]
        sell_signal_for_today = signals_df['sell_signal'].iloc[i-1]
        
        previous_day_volume_confirmed = previous_day_data_row.get('volume_confirmed', False)
        previous_day_weekly_uptrend = previous_day_data_row.get('weekly_uptrend', True) 

        # Use the 'use_trailing_stops' parameter passed to the function
        if use_trailing_stops:
            long_manager.update_trailing_stops(transaction_price, previous_day_atr, previous_day_adx)
            short_manager.update_trailing_stops(transaction_price, previous_day_atr, previous_day_adx)

        long_manager.process_exits(current_trade_date, transaction_price,
                                   signal_exit=sell_signal_for_today, 
                                   trend_valid=(previous_day_adx > adx_threshold_sig)) 
        short_manager.process_exits(current_trade_date, transaction_price,
                                     signal_exit=buy_signal_for_today,
                                     trend_valid=(previous_day_adx > adx_threshold_sig))

        entry_params_base = {
            'price': transaction_price, 
            'atr': previous_day_atr,    
            'adx': previous_day_adx,    
            'position_size': position_size, 
            'leverage': leverage_ratio,
            'sar': previous_day_data_row.get('SAR', transaction_price * 0.97) 
        }

        if buy_signal_for_today and previous_day_volume_confirmed: 
            long_entry_params = {**entry_params_base,
                                 'portfolio_value': long_manager.portfolio_value,
                                 'risk': long_risk, 'reward': long_reward}
            long_manager.process_entry(current_trade_date, long_entry_params)

        if sell_signal_for_today and not previous_day_weekly_uptrend: 
            short_entry_params = {**entry_params_base,
                                  'portfolio_value': short_manager.portfolio_value,
                                  'risk': short_risk, 'reward': short_reward}
            short_manager.process_entry(current_trade_date, short_entry_params)
        
        total_value = (long_manager.portfolio_value + short_manager.portfolio_value +
                       long_manager.calculate_unrealized_pnl(transaction_price) +
                       short_manager.calculate_unrealized_pnl(transaction_price))
        equity_curve.iloc[i] = total_value

        if equity_curve.iloc[i-1] != 0:
            returns_series.iloc[i] = (total_value / equity_curve.iloc[i-1]) - 1
        else:
            returns_series.iloc[i] = 0.0 if total_value == 0 else np.nan

    loop_end_time = datetime.now() # Overall loop end time
    total_loop_duration = (loop_end_time - loop_start_time).total_seconds()

    # 4. Combine trade logs and calculate statistics
    combined_trade_log = deque(list(long_manager.trade_log) + list(short_manager.trade_log))
    combined_wins = deque(list(long_manager.wins) + list(short_manager.wins))
    combined_losses = deque(list(long_manager.losses) + list(short_manager.losses))

    # Ensure trade_statistics function is defined and accessible
    stats = trade_statistics(equity_curve, combined_trade_log, combined_wins, combined_losses, risk_free_rate)

    return combined_trade_log, stats, equity_curve, returns_series

# --------------------------------------------------------------------------------------------------------------------------

def _dynamic_thresholds(atr_ratio_np, base_buy, base_sell):
    """Calculates dynamic thresholds based on volatility."""
    # Ensure atr_ratio_np is a numpy array
    volatility_factor = np.log1p(atr_ratio_np / 0.02) # 0.02 is a reference ATR ratio
    return (
        base_buy * (1 + volatility_factor),
        base_sell * (1 + volatility_factor)
    )

# --------------------------------------------------------------------------------------------------------------------------

def _calculate_dynamic_scores(conditions, rsi_np_arr, mfi_np_arr, trend_weights_dict, 
                            confirmation_weights_dict, direction='long', 
                            weekly_trend=None, atr_ratio_np=None):
    """Enhanced scoring system with volatility-adjusted indicators"""
    # Initialize base score
    if rsi_np_arr is not None and hasattr(rsi_np_arr, 'shape'):
        base_score = np.zeros_like(rsi_np_arr, dtype=float)
    elif mfi_np_arr is not None and hasattr(mfi_np_arr, 'shape'):
        base_score = np.zeros_like(mfi_np_arr, dtype=float)
    else:
        base_score = np.zeros_like(conditions.get('primary_trend_long', np.array([])), dtype=float)

    # Trend components with weekly trend confirmation
    if direction == 'long':
        base_score += conditions.get('primary_trend_long', False).astype(float) * trend_weights_dict.get('primary_trend', 0)
        if weekly_trend is not None:
            base_score += weekly_trend.astype(float) * 1.5  # Additional weight for higher timeframe
    else:
        base_score += conditions.get('primary_trend_short', False).astype(float) * trend_weights_dict.get('primary_trend', 0)
        if weekly_trend is not None:
            base_score += (~weekly_trend).astype(float) * 1.5  # Inverse for shorts

    # Volatility-adjusted momentum scoring
    if atr_ratio_np is not None:
        high_vol_mask = atr_ratio_np > 0.015
        if direction == 'long':
            rsi_strength = np.where(
                high_vol_mask,
                np.clip((rsi_np_arr - 40) / 15, 0, 1),  # More sensitive in high vol
                np.clip((rsi_np_arr - 45) / 10, 0, 1)   # Less sensitive in low vol
            )
        else:
            rsi_strength = np.where(
                high_vol_mask,
                np.clip((60 - rsi_np_arr) / 15, 0, 1),  # More sensitive in high vol
                np.clip((55 - rsi_np_arr) / 10, 0, 1)   # Less sensitive in low vol
            )
    else:
        # Fall back to original RSI calculation if atr_ratio not provided
        if direction == 'long':
            rsi_strength = np.clip((rsi_np_arr - 50) / 20, 0, 1)
        else:
            rsi_strength = np.clip((50 - rsi_np_arr) / 20, 0, 1)

    # Add remaining components
    base_score += rsi_strength * confirmation_weights_dict.get('rsi', 0)
    base_score += conditions.get('volume_ok', False).astype(float) * confirmation_weights_dict.get('volume', 0)
    
    # Additional trend strength confirmatioxns
    base_score += conditions.get('trend_strength_ok', False).astype(float) * trend_weights_dict.get('trend_strength', 0)
    
    return base_score

# --------------------------------------------------------------------------------------------------------------------------

def _generate_final_signals(conditions, scores_dict, min_scores_dict, rsi_np_arr):
    """Improved signal generation with additional filters"""
    # Long Signals
    long_checks = (
        conditions.get('primary_trend_long', False) &
        conditions.get('trend_strength_ok', False) &
        (conditions.get('kumo_breakout', False) | conditions.get('strong_cloud_support', False)) &
        (scores_dict.get('buy', np.array([-np.inf])) >= min_scores_dict.get('buy', np.array([np.inf]))) & # Ensure proper default for comparison
        conditions.get('volume_ok', False)
    )
    
    # Short Signals
    short_checks = (
        conditions.get('primary_trend_short', False) &
        conditions.get('trend_strength_ok', False) &
        (conditions.get('kumo_breakdown', False) | conditions.get('below_cloud', False)) &
        (scores_dict.get('sell', np.array([-np.inf])) >= min_scores_dict.get('sell', np.array([np.inf]))) & # Ensure proper default for comparison
        conditions.get('volume_ok', False)
    )
    
    # Additional filter: Don't trade in extreme RSI zones
    overbought = rsi_np_arr > 70
    oversold = rsi_np_arr < 30
    
    
    final_buy = long_checks & ~overbought
    final_sell = short_checks & ~oversold
    
    return {
        'buy': final_buy,
        'sell': final_sell
    }

# --------------------------------------------------------------------------------------------------------------------------

def signals(df, adx_threshold_val, min_buy_score_val, min_sell_score_val, require_cloud_val,
                       fast_ma_col, slow_ma_col, weekly_ma_col_name, 
                       trend_weights_dict_param=None, confirmation_weights_dict_param=None):
    """
    Generates trading signals for the entire DataFrame using vectorized operations, NumPy,
    primary trend filters, secondary confirmations, and a weighted scoring approach.
    """
    signals_df = pd.DataFrame(index=df.index)

    # Use passed weights if available, otherwise default to global
    current_trend_weights = trend_weights_dict_param if trend_weights_dict_param is not None else TREND_WEIGHTS
    current_confirmation_weights = confirmation_weights_dict_param if confirmation_weights_dict_param is not None else CONFIRMATION_WEIGHTS

    required_indicator_cols = [
        fast_ma_col, slow_ma_col, weekly_ma_col_name, 'RSI', 'MFI', 'Close', 'Lower_Band', 'Upper_Band',
        'ATR', 'ADX', 'Tenkan_sen', 'Kijun_sen', 'Senkou_span_A',
        'Senkou_span_B', 'Volume', 'Volume_MA20', 'Open', 'High', 'Low'
    ]
    missing_cols = [col for col in required_indicator_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for signals: {missing_cols}. Returning empty signals.")
        for col_name in ['buy_signal', 'sell_signal', 'buy_score', 'sell_score', 'signal_changed']:
            signals_df[col_name] = False if 'signal' in col_name or 'changed' in col_name else 0.0
        return signals_df

    # 1. Performance Optimizations: Use NumPy arrays
    close_np = df['Close'].values
    atr_np = df['ATR'].values
    adx_np = df['ADX'].values
    fast_ma_np = df[fast_ma_col].values
    slow_ma_np = df[slow_ma_col].values
    weekly_ma_np = df[weekly_ma_col_name].values
    rsi_np = df['RSI'].values
    mfi_np = df['MFI'].values
    lower_band_np = df['Lower_Band'].values
    upper_band_np = df['Upper_Band'].values
    tenkan_np = df['Tenkan_sen'].values
    kijun_np = df['Kijun_sen'].values
    spanA_np = df['Senkou_span_A'].values
    spanB_np = df['Senkou_span_B'].values
    volume_np = df['Volume'].values
    volume_ma20_np = df['Volume_MA20'].values
    
    atr_ratio_np = np.full_like(close_np, 0.02, dtype=float) # Ensure float
    valid_close_mask = close_np != 0
    # Ensure division by zero is handled if close_np can be zero
    safe_close_np = np.where(valid_close_mask, close_np, 1e-9) # Avoid division by zero
    atr_ratio_np[valid_close_mask] = atr_np[valid_close_mask] / safe_close_np[valid_close_mask]
    atr_ratio_np = np.nan_to_num(atr_ratio_np, nan=0.02, posinf=0.02, neginf=0.02)

    actual_min_buy_np, actual_min_sell_np = _dynamic_thresholds(
        atr_ratio_np, min_buy_score_val, min_sell_score_val
    )

    # Calculate Ichimoku components for conditions
    span_max_np = np.maximum(spanA_np, spanB_np)
    span_min_np = np.minimum(spanA_np, spanB_np)
    min_cloud_thickness_val_np = np.maximum(0.005 * close_np, 0.005)
    min_cloud_thickness_val_np[~valid_close_mask] = 0.005
    cloud_thickness_np = np.abs(spanA_np - spanB_np)

    # 2. Calculate Primary and Secondary Conditions
    conditions = {}
    # Primary Trend Filters
    conditions['primary_trend_long'] = (close_np > fast_ma_np) & (close_np > slow_ma_np) 
    conditions['primary_trend_short'] = (close_np < fast_ma_np) & (close_np < slow_ma_np) 
    conditions['trend_strength_ok'] = adx_np > adx_threshold_val

    # Secondary Confirmation Indicators / Scoring Components
    # Ichimoku
    conditions['strong_cloud_support'] = (close_np > span_max_np) & (cloud_thickness_np > min_cloud_thickness_val_np)
    conditions['below_cloud'] = close_np < span_min_np
    conditions['kumo_breakout'] = ((tenkan_np > kijun_np) & (close_np > span_max_np) & (cloud_thickness_np > min_cloud_thickness_val_np))
    conditions['kumo_breakdown'] = ((tenkan_np < kijun_np) & (close_np < span_min_np) & (cloud_thickness_np > min_cloud_thickness_val_np))
    
    # Momentum (used in _calculate_dynamic_scores, not directly as conditions here)
    # Volume
    conditions['volume_ok'] = volume_np > volume_ma20_np
    # MA Alignment for scoring
    conditions['ma_buy'] = fast_ma_np > slow_ma_np
    conditions['ma_sell'] = fast_ma_np < slow_ma_np
    # Bollinger Bands for scoring
    conditions['bb_buy'] = close_np < lower_band_np
    conditions['bb_sell'] = close_np > upper_band_np
    
    # Ensure all conditions are boolean arrays of the same shape
    # Get a reference shape from a reliable condition
    ref_shape = close_np.shape
    for key in list(conditions.keys()): # Iterate over a copy of keys if modifying dict
        cond_val = conditions[key]
        if not isinstance(cond_val, np.ndarray) or cond_val.shape != ref_shape:
            # For simplicity, if a condition is scalar False/True, broadcast it
            if isinstance(cond_val, bool):
                conditions[key] = np.full(ref_shape, cond_val, dtype=bool)
            else: # If it's an array of wrong shape or other type, default to all False
                conditions[key] = np.zeros(ref_shape, dtype=bool)
        # Fill NaNs in boolean conditions
        if pd.api.types.is_bool_dtype(conditions[key]):
             conditions[key] = np.nan_to_num(conditions[key].astype(float), nan=0.0).astype(bool)
        elif hasattr(conditions[key], 'dtype') and pd.isna(conditions[key]).any(): # Check if it's an array and has nans
             conditions[key] = np.nan_to_num(conditions[key].astype(float), nan=0.0).astype(bool)


    # 3. Calculate Dynamic Scores
    # Ensure TREND_WEIGHTS and CONFIRMATION_WEIGHTS are accessible (e.g., global or passed)
    buy_score_np = _calculate_dynamic_scores(conditions, rsi_np, mfi_np, current_trend_weights, current_confirmation_weights, direction='long')
    sell_score_np = _calculate_dynamic_scores(conditions, rsi_np, mfi_np, current_trend_weights, current_confirmation_weights, direction='short')
    
    signals_df['buy_score'] = buy_score_np
    signals_df['sell_score'] = sell_score_np

    # 4. Generate Final Signals
    final_signals_dict = _generate_final_signals(
        conditions,
        {'buy': buy_score_np, 'sell': sell_score_np},
        {'buy': actual_min_buy_np, 'sell': actual_min_sell_np},
        rsi_np # Pass rsi_np for overbought/oversold checks
    )

    signals_df['buy_signal'] = final_signals_dict['buy']
    signals_df['sell_signal'] = final_signals_dict['sell']

    # 5. Signal Validation (Conflicting signals and changed status)
    conflicting_mask = np.logical_and(signals_df['buy_signal'].values, signals_df['sell_signal'].values)
    signals_df.loc[conflicting_mask, 'buy_signal'] = False
    signals_df.loc[conflicting_mask, 'sell_signal'] = False
    
    last_signal_np = signals_df['buy_signal'].astype(int).values - signals_df['sell_signal'].astype(int).values
    signal_changed_np = np.diff(last_signal_np, prepend=0) != 0
    signals_df['signal_changed'] = signal_changed_np

    return signals_df

# --------------------------------------------------------------------------------------------------------------------------

class TradeManager:
    def __init__(self, initial_capital, max_positions, direction='Long'):
        self.direction = direction
        self.portfolio_value = initial_capital
        self.max_positions = max_positions
        self.position_count = 0
        self.trade_log = deque()  # Use deque instead of list
        self.wins = deque()       # Use deque instead of list
        self.losses = deque()     # Use deque instead of list
        self.lengths = deque()    # Use deque instead of list
        
        # Define column dtypes for better memory usage
        self.dtypes = {
            'entry_date': 'datetime64[ns]',
            'multiplier': 'int8',
            'entry_price': 'float32',
            'stop_loss': 'float32',
            'take_profit': 'float32',
            'position_size': 'float32',
            'share_amount': 'int32',
            'highest_close_since_entry': 'float32',
            'lowest_close_since_entry': 'float32'
        }
        
        # Initialize active trades DataFrame with proper dtypes
        self.active_trades = pd.DataFrame(columns=self.dtypes.keys()).astype(self.dtypes)

    # ----------------------------------------------------------------------------------------------------------

    def _calculate_entry_parameters(self, entry_params_dict):
        """
        Calculates entry parameters for a new trade with dynamic risk/reward based on volatility.
        """
        price = entry_params_dict['price']
        atr = entry_params_dict['atr']
        portfolio_value = entry_params_dict['portfolio_value']
        risk_pct = entry_params_dict['risk']
        position_size_pct = entry_params_dict['position_size']
        leverage = entry_params_dict['leverage']
        adx = entry_params_dict.get('adx', 25)  # Default ADX if not provided

        # Dynamic multipliers based on volatility regime
        atr_ratio = atr / price
        if atr_ratio > 0.015:  # High volatility
            atr_multiplier = 3.0 if self.direction == 'Long' else 2.5
            reward_multiplier = 2.5  # Smaller targets in high vol
        else:  # Low volatility
            atr_multiplier = 2.0 if self.direction == 'Long' else 1.5  
            reward_multiplier = 3.0  # Larger targets in low vol
        
        # Scale position size based on ADX strength
        adx_strength = min(max((adx - 20) / 30, 0), 1)  # Normalized 0-1
        position_size_multiplier = 1.0 + adx_strength  # 1-2x
        scaled_position_size = position_size_pct * position_size_multiplier

        if self.direction == 'Long':
            stop_loss_price = price - atr * atr_multiplier
            risk_per_share = price - stop_loss_price
        else:  # Short
            stop_loss_price = price + atr * atr_multiplier
            risk_per_share = stop_loss_price - price

        # Ensure risk_per_share is positive and not too small
        valid_risk_per_share = np.maximum(risk_per_share, 1e-9)

        max_monetary_risk = portfolio_value * risk_pct
        num_shares_risk = np.floor(max_monetary_risk / valid_risk_per_share)

        base_exposure = portfolio_value * scaled_position_size
        max_notional_allowed = base_exposure * leverage
        num_shares_leverage = np.floor(max_notional_allowed / price)

        share_amount = int(np.minimum(num_shares_risk, num_shares_leverage))
        
        if share_amount <= 0:
            return None

        take_profit = None
        if 'reward' in entry_params_dict:
            take_profit_distance = valid_risk_per_share * reward_multiplier
            if self.direction == 'Long':
                take_profit = price + take_profit_distance
            else:
                take_profit = price - take_profit_distance
        
        actual_notional = share_amount * price
        exposure_ratio = actual_notional / portfolio_value if portfolio_value > 0 else 0.0

        return price, stop_loss_price, take_profit, exposure_ratio, share_amount

    # ----------------------------------------------------------------------------------------------------------

    @staticmethod
    @jit(nopython=True)
    def _calculate_position_size(price, atr, portfolio_value, risk_pct, position_size_pct, 
                               leverage, adx, direction):
        """Numba-optimized position size calculation"""
        position_size_multiplier = 1.5 if adx > 30 else 1.0
        scaled_position_size = position_size_pct * position_size_multiplier
        atr_multiplier = 2.0 if adx > 30 else 4.0
        
        stop_loss_price = price - atr * atr_multiplier if direction == 1 else price + atr * atr_multiplier
        risk_per_share = abs(price - stop_loss_price)
        
        max_risk = portfolio_value * risk_pct
        shares_risk = np.floor(max_risk / max(risk_per_share, 1e-9))
        
        max_exposure = portfolio_value * scaled_position_size * leverage
        shares_leverage = np.floor(max_exposure / price)
        
        return min(shares_risk, shares_leverage), stop_loss_price
    
    # ----------------------------------------------------------------------------------------------------------

    def process_entry(self, current_date, entry_params):
        """Process new trade entry with enhanced stop calculation"""
        if self.position_count >= self.max_positions or entry_params['portfolio_value'] <= 0:
            return False
            
        direction_mult = 1 if self.direction == 'Long' else -1
        
        # Calculate initial stop using the enhanced method
        initial_stop = self.calculate_initial_stop(
            price=entry_params['price'],
            atr=entry_params['atr'],
            adx=entry_params['adx'],
            current_sar=entry_params.get('sar', entry_params['price'] * 0.97)  # Use SAR if provided, else fallback
        )
        
        # Use the calculated stop for position sizing
        risk_per_share = abs(entry_params['price'] - initial_stop)
        max_risk = entry_params['portfolio_value'] * entry_params['risk']
        # Ensure risk_per_share is not zero to avoid division by zero
        shares_risk = np.floor(max_risk / max(risk_per_share, 1e-9)) if max(risk_per_share, 1e-9) > 0 else 0
        
        # Calculate position size based on risk and leverage
        max_exposure = entry_params['portfolio_value'] * entry_params['position_size'] * entry_params['leverage']
        shares_leverage = np.floor(max_exposure / entry_params['price']) if entry_params['price'] > 0 else 0
        
        share_amount = int(min(shares_risk, shares_leverage))
        
        if share_amount <= 0:
            return False
            
        # Calculate take profit using risk-reward ratio
        take_profit = None
        if 'reward' in entry_params:
            reward_mult = 3.0 if entry_params['adx'] > 30 else 2.0
            take_profit_distance = risk_per_share * reward_mult
            take_profit = entry_params['price'] + (take_profit_distance * direction_mult)
        
        # Create new trade with proper dtypes
        new_trade = pd.DataFrame([{
            'entry_date': current_date,
            'multiplier': direction_mult,
            'entry_price': entry_params['price'],
            'stop_loss': initial_stop,  # Use the calculated initial stop
            'take_profit': take_profit,
            'position_size': (share_amount * entry_params['price']) / entry_params['portfolio_value'] if entry_params['portfolio_value'] > 0 else 0,
            'share_amount': share_amount,
            'highest_close_since_entry': entry_params['price'] if self.direction == 'Long' else np.nan,
            'lowest_close_since_entry': entry_params['price'] if self.direction == 'Short' else np.nan
        }]).astype(self.dtypes)
        
        concatenated_trades = pd.concat([self.active_trades, new_trade], ignore_index=True)
        self.active_trades = concatenated_trades.astype(self.dtypes)

        self.position_count += 1
        return True
    
    # ----------------------------------------------------------------------------------------------------------

    def _calculate_exit_pnl(self, trade_series, exit_date, exit_price, shares_to_exit, reason):
        """
        Calculates PnL for a single trade exit and logs the trade.
        `trade_series` is a pandas Series representing the active trade row.
        `shares_to_exit` is the number of shares being exited for this specific event.
        """
        entry_price = trade_series['entry_price']
        entry_date = trade_series['entry_date']

        pnl = 0
        if self.direction == 'Long':
            pnl = (exit_price - entry_price) * shares_to_exit
        else:  # Short
            pnl = (entry_price - exit_price) * shares_to_exit

        duration = 0
        # Ensure entry_date is a Timestamp if it's not already
        if not isinstance(entry_date, pd.Timestamp):
            entry_date = pd.Timestamp(entry_date)
        if pd.notnull(entry_date) and pd.notnull(exit_date):
            if not isinstance(exit_date, pd.Timestamp): # Ensure exit_date is also Timestamp
                exit_date = pd.Timestamp(exit_date)
            duration = (exit_date - entry_date).days
        
        self.lengths.append(duration)

        if pnl > 0:
            self.wins.append(pnl)
        else:
            self.losses.append(pnl)
        
        self.trade_log.append({
            'Entry Date': entry_date,
            'Exit Date': exit_date,
            'Direction': self.direction,
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Shares': shares_to_exit, # Log the shares actually exited
            'PnL': pnl,
            'Duration': duration,
            'Exit Reason': reason
        })
        return pnl

    # ----------------------------------------------------------------------------------------------------------

    def process_exits(self, current_date, current_price, signal_exit, trend_valid):
        if self.active_trades.empty:
            return 0.0

        total_pnl_for_iteration = 0.0
        indices_to_remove_from_active_trades = []

        # Iterate over a copy of indices as we might modify share_amount or mark for removal
        for idx in self.active_trades.index.tolist():
            # Skip if already marked for full removal in this iteration (e.g. by a prior signal exit)
            if idx in indices_to_remove_from_active_trades:
                continue

            trade_row = self.active_trades.loc[idx]
            current_shares_in_trade = trade_row['share_amount']

            if current_shares_in_trade <= 0: # Should not happen if trades are removed properly
                if idx not in indices_to_remove_from_active_trades:
                    indices_to_remove_from_active_trades.append(idx)
                continue

            exited_this_step = False

            # 1. Process Signal Exits
            if signal_exit:
                if trend_valid:  # Partial Exit (30%)
                    shares_to_exit_partial = int(current_shares_in_trade * 0.3)
                    if shares_to_exit_partial > 0:
                        pnl = self._calculate_exit_pnl(trade_row, current_date, current_price, shares_to_exit_partial, 'Partial Signal Exit (Trend Valid)')
                        total_pnl_for_iteration += pnl
                        self.portfolio_value += pnl
                        self.active_trades.loc[idx, 'share_amount'] -= shares_to_exit_partial
                        current_shares_in_trade = self.active_trades.loc[idx, 'share_amount'] # Update current shares
                        if current_shares_in_trade <= 0:
                            indices_to_remove_from_active_trades.append(idx)
                            self.position_count -= 1
                            exited_this_step = True
                else:  # Full Exit (Trend Invalid)
                    pnl = self._calculate_exit_pnl(trade_row, current_date, current_price, current_shares_in_trade, 'Full Signal Exit (Trend Invalid)')
                    total_pnl_for_iteration += pnl
                    self.portfolio_value += pnl
                    self.active_trades.loc[idx, 'share_amount'] = 0
                    indices_to_remove_from_active_trades.append(idx)
                    self.position_count -= 1
                    exited_this_step = True
            
            if exited_this_step:
                continue # Move to next trade if this one was fully closed by signal

            # 2. Process Stop Loss (if not already exited)
            sl_hit = False
            if self.direction == 'Long' and current_price <= trade_row['stop_loss']:
                sl_hit = True
            elif self.direction == 'Short' and current_price >= trade_row['stop_loss']:
                sl_hit = True
            
            if sl_hit:
                pnl = self._calculate_exit_pnl(trade_row, current_date, current_price, current_shares_in_trade, 'Stop Loss')
                total_pnl_for_iteration += pnl
                self.portfolio_value += pnl
                self.active_trades.loc[idx, 'share_amount'] = 0
                indices_to_remove_from_active_trades.append(idx)
                self.position_count -= 1
                continue # Exited by SL, move to next trade

            # 3. Process Take Profit (if not already exited by SL or signal)
            tp_hit = False
            if pd.notna(trade_row['take_profit']):
                if self.direction == 'Long' and current_price >= trade_row['take_profit']:
                    tp_hit = True
                elif self.direction == 'Short' and current_price <= trade_row['take_profit']:
                    tp_hit = True
            
            if tp_hit:
                pnl = self._calculate_exit_pnl(trade_row, current_date, current_price, current_shares_in_trade, 'Take Profit')
                total_pnl_for_iteration += pnl
                self.portfolio_value += pnl
                self.active_trades.loc[idx, 'share_amount'] = 0
                indices_to_remove_from_active_trades.append(idx)
                self.position_count -= 1
                # continue will happen implicitly at end of loop

        # Cleanup: Remove trades that were fully closed
        if indices_to_remove_from_active_trades:
            # Ensure unique indices before dropping
            unique_indices = sorted(list(set(indices_to_remove_from_active_trades)))
            self.active_trades = self.active_trades.drop(index=unique_indices).reset_index(drop=True)
        
        return total_pnl_for_iteration
    
    # ----------------------------------------------------------------------------------------------------------

    def update_trailing_stops(self, current_price, current_atr, adx_value):
        """
        Enhanced trailing stop management with dynamic ATR multiplier based on market conditions.
        """
        if self.active_trades.empty: 
            return
        
        current_atr_f32 = np.float32(current_atr)
        safe_current_price = np.float32(current_price) if current_price != 0 else np.float32(1e-9)
        atr_ratio = current_atr_f32 / safe_current_price
        
        if adx_value > 30 and atr_ratio < 0.01:  # Strong trend, low vol
            atr_multiplier = np.float32(4.0)  # Give more room
        elif adx_value > 25:  # Moderate trend
            atr_multiplier = np.float32(3.0)
        else:  # Weak trend or high vol
            atr_multiplier = np.float32(2.0)
        
        if self.direction == 'Long':
            # Update highest prices seen
            self.active_trades.loc[:, 'highest_close_since_entry'] = np.maximum(
                self.active_trades['highest_close_since_entry'].values, 
                np.float32(current_price) # Ensure current_price is also float32 for comparison
            ).astype(np.float32) # Explicitly cast the result
            # Calculate new stops with dynamic multiplier
            new_stops = self.active_trades['highest_close_since_entry'].values - (atr_multiplier * current_atr_f32)
            
            # Never move stops down for longs
            self.active_trades.loc[:, 'stop_loss'] = np.maximum(
                new_stops, 
                self.active_trades['stop_loss'].values
            ).astype(np.float32) # Explicitly cast the final result to float32
        else:  # Short
            # Update lowest prices seen
            self.active_trades.loc[:, 'lowest_close_since_entry'] = np.minimum(
                self.active_trades['lowest_close_since_entry'].values, 
                np.float32(current_price) # Ensure current_price is also float32 for comparison
            ).astype(np.float32) # Explicitly cast the result

            # Calculate new stops with dynamic multiplier
            new_stops = self.active_trades['lowest_close_since_entry'].values + (atr_multiplier * current_atr_f32)
            
            # Never move stops up for shorts
            self.active_trades.loc[:, 'stop_loss'] = np.minimum(
                new_stops, 
                self.active_trades['stop_loss'].values
            ).astype(np.float32) # Explicitly cast the final result to float32

    # ----------------------------------------------------------------------------------------------------------

    def calculate_initial_stop(self, price, atr, adx, current_sar):
        """
        Calculate initial stop loss using multiple technical factors.
        
        Args:
            price (float): Current price
            atr (float): Average True Range
            adx (float): ADX value
            current_sar (float): Current Parabolic SAR value
        """
        # Use parabolic SAR as stop baseline
        sar_distance = abs(price - current_sar)
        
        # ATR-based stop distance varies with trend strength
        atr_multiplier = 3.0 if adx > 25 else 2.0
        atr_distance = atr * atr_multiplier
        
        # Use the smaller of SAR and ATR distances to set stop
        # This helps prevent stops that are too wide
        stop_distance = min(sar_distance, atr_distance)
        
        # Add minimum stop distance based on price
        min_stop_distance = price * 0.005  # 0.5% minimum
        stop_distance = max(stop_distance, min_stop_distance)
        
        if self.direction == 'Long':
            return price - stop_distance
        else:
            return price + stop_distance

    # ----------------------------------------------------------------------------------------------------------

    def calculate_unrealized_pnl(self, current_price):
        if self.active_trades.empty: return 0.0
        if self.direction == 'Long':
            return ((current_price - self.active_trades['entry_price']) * self.active_trades['share_amount']).sum()
        else:
            return ((self.active_trades['entry_price'] - current_price) * self.active_trades['share_amount']).sum()
        
# --------------------------------------------------------------------------------------------------------------------------

def create_empty_stats(risk_free_rate=0.04):
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

# --------------------------------------------------------------------------------------------------------------------------

@jit(nopython=True)
def _calculate_risk_metrics(returns_array, risk_free_daily):
    """Numba-optimized calculation of risk metrics"""
    excess_returns = returns_array - risk_free_daily
    returns_mean = np.mean(excess_returns)
    returns_std = np.std(excess_returns)
    
    # Sharpe Ratio
    sharpe = (returns_mean / returns_std) * np.sqrt(252) if returns_std != 0 else 0
    
    # Sortino Ratio
    downside_returns = returns_array[returns_array < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
    sortino = ((np.mean(returns_array) - risk_free_daily) / downside_std) * np.sqrt(252) if downside_std != 0 else 0
    
    return sharpe, sortino

# --------------------------------------------------------------------------------------------------------------------------

def trade_statistics(equity, trade_log, wins, losses, risk_free_rate=0.04):
    """Vectorized trade statistics calculation using NumPy and Numba"""
    if not trade_log:
        return create_empty_stats()
    
    # Convert lists to NumPy arrays
    wins_array = np.array(wins) if wins else np.array([0.0])
    losses_array = np.array(losses) if losses else np.array([0.0])
    
    # Basic trade statistics (vectorized)
    total_trades = len(trade_log)
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    
    # Profit metrics (vectorized)
    gross_profit = np.sum(wins_array)
    gross_loss = np.sum(losses_array)
    net_profit = gross_profit + gross_loss
    
    # Portfolio metrics (vectorized)
    initial_capital = equity.iloc[0]
    final_capital = equity.iloc[-1]
    net_profit_pct = ((final_capital / initial_capital) - 1) * 100
    
    # Risk metrics (vectorized)
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else np.inf
    
    # Expectancy calculation (vectorized)
    avg_win = np.mean(wins_array) if len(wins) > 0 else 0
    avg_loss = np.mean(losses_array) if len(losses) > 0 else 0
    win_prob = win_rate / 100
    expectancy = (win_prob * avg_win) + ((1 - win_prob) * avg_loss)
    expectancy_pct = (expectancy / initial_capital) * 100 if initial_capital > 0 else 0
    
    # Drawdown calculation (vectorized)
    equity_values = equity.values
    running_max = np.maximum.accumulate(equity_values)
    drawdowns = ((equity_values - running_max) / running_max) * 100
    max_drawdown = abs(np.min(drawdowns))
    
    # Time-based metrics
    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25
    annualized_return = ((final_capital / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # Calculate risk ratios using Numba-optimized function
    daily_returns = np.diff(np.log(equity_values))
    daily_rf_rate = risk_free_rate / 252
    sharpe_ratio, sortino_ratio = _calculate_risk_metrics(daily_returns, daily_rf_rate)
    
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

@jit(nopython=True, parallel=True)
def _evaluate_parameters_batch_cpu(param_arrays, price_data, fast_ma, slow_ma):
    """
    Vectorized and JIT-compiled evaluation of multiple parameter sets simultaneously.
    
    Args:
        param_arrays: numpy array of shape (n_params, n_features) containing parameter combinations
        price_data: numpy array of price data
        fast_ma: numpy array of fast moving average values
        slow_ma: numpy array of slow moving average values
    
    Returns:
        numpy array of shape (n_params,) containing evaluation metrics
    """
    n_params = param_arrays.shape[0]
    results = np.zeros(n_params, dtype=np.float32)
    
    for i in range(n_params):
        # Extract parameters
        long_risk = param_arrays[i, 0]
        long_reward = param_arrays[i, 1]
        short_risk = param_arrays[i, 2]
        short_reward = param_arrays[i, 3]
        position_size = param_arrays[i, 4]
        
        # Quick validation
        if long_risk >= long_reward or short_risk >= short_reward:
            results[i] = -np.inf
            continue
            
        # Calculate signals and returns
        signals = np.zeros(len(price_data))
        for j in range(1, len(price_data)):
            # Simplified signal generation for performance
            if fast_ma[j] > slow_ma[j]:
                signals[j] = 1
            elif fast_ma[j] < slow_ma[j]:
                signals[j] = -1
                
        # Calculate returns
        position_values = signals * position_size
        returns = np.diff(price_data) / price_data[:-1]
        strategy_returns = position_values[:-1] * returns
        
        # Calculate metrics
        if len(strategy_returns) > 0:
            sharpe = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else -np.inf
            results[i] = sharpe
        else:
            results[i] = -np.inf
            
    return results

# --------------------------------------------------------------------------------------------------------------------------

def _evaluate_parameters_batch_gpu(param_arrays, price_data, fast_ma, slow_ma):
    """GPU version of parameter evaluation using CuPy"""
    try:
        # Convert input arrays to GPU arrays
        param_arrays_gpu = cp.asarray(param_arrays, dtype=cp.float32)
        price_data_gpu = cp.asarray(price_data, dtype=cp.float32)
        fast_ma_gpu = cp.asarray(fast_ma, dtype=cp.float32)
        slow_ma_gpu = cp.asarray(slow_ma, dtype=cp.float32)
        
        n_params = param_arrays_gpu.shape[0]
        results = cp.zeros(n_params, dtype=cp.float32)
        
        # Vectorized operations on GPU
        signals = cp.zeros((n_params, len(price_data_gpu)), dtype=cp.float32)
        
        # Calculate signals for all parameter sets at once
        for i in range(1, len(price_data_gpu)):
            signals[:, i] = cp.where(fast_ma_gpu[i] > slow_ma_gpu[i], 1, 
                                   cp.where(fast_ma_gpu[i] < slow_ma_gpu[i], -1, 0))
        
        # Broadcast position sizes
        position_values = signals * param_arrays_gpu[:, 4:5]
        
        # Calculate returns
        returns = cp.diff(price_data_gpu) / price_data_gpu[:-1]
        strategy_returns = position_values[:, :-1] * returns
        
        # Calculate metrics
        valid_returns = cp.any(strategy_returns != 0, axis=1)
        means = cp.mean(strategy_returns, axis=1)
        stds = cp.std(strategy_returns, axis=1)
        
        # Calculate Sharpe ratios
        results = cp.where(
            cp.logical_and(valid_returns, stds > 0),
            cp.sqrt(252) * means / cp.maximum(stds, 1e-8),
            -cp.inf
        )
        
        # Risk-reward validation
        invalid_params = cp.logical_or(
            param_arrays_gpu[:, 0] >= param_arrays_gpu[:, 1],  # long_risk >= long_reward
            param_arrays_gpu[:, 2] >= param_arrays_gpu[:, 3]   # short_risk >= short_reward
        )
        results = cp.where(invalid_params, -cp.inf, results)
        
        # Free GPU memory explicitly
        del param_arrays_gpu, price_data_gpu, fast_ma_gpu, slow_ma_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return cp.asnumpy(results)
        
    except Exception as e:
        print(f"GPU processing error: {e}")
        raise  # Re-raise the exception to trigger CPU fallback

# --------------------------------------------------------------------------------------------------------------------------

def _random_parameter_sampling(param_lists, sample_size, num_params_before_weights, optimization_type, ticker, base_df):
    """
    Smart random sampling of parameter combinations.
    Args:
        param_lists: List of parameter value lists
        sample_size: Number of combinations to generate
        num_params_before_weights: Number of parameters in combo before scoring weights (specific to 'technical')
        optimization_type: String, 'basic' or 'technical'
        ticker: The ticker symbol string
        base_df: The base DataFrame for the ticker (passed by reference, copied in parameter_test)
    Returns:
        List of parameter combinations
    """
    param_grid = deque()
    seen_combinations = set()
    total_attempts = 0
    max_attempts = sample_size * 5  # Increased attempts for potentially sparse valid space

    while len(param_grid) < sample_size and total_attempts < max_attempts:
        indices = [np.random.randint(len(p_list)) for p_list in param_lists]
        param_key = tuple(indices)

        if param_key in seen_combinations:
            total_attempts += 1
            continue
        seen_combinations.add(param_key)

        combo = [param_lists[j][idx] for j, idx in enumerate(indices)]
        total_attempts += 1

        # Initialize tech_params with all global defaults
        tech_params = {
            'FAST': FAST, 'SLOW': SLOW,
            'RSI_OVERSOLD': RSI_OVERSOLD, 'RSI_OVERBOUGHT': RSI_OVERBOUGHT,
            'DEVS': DEVS, 'ADX_THRESHOLD': ADX_THRESHOLD_DEFAULT,
            'MIN_BUY_SCORE': MIN_BUY_SCORE_DEFAULT, 'MIN_SELL_SCORE': MIN_SELL_SCORE_DEFAULT,
            'REQUIRE_CLOUD': REQUIRE_CLOUD_DEFAULT,
            'WEEKLY_MA_PERIOD': WEEKLY_MA_PERIOD,
            'TREND_WEIGHTS': TREND_WEIGHTS.copy(),
            'CONFIRMATION_WEIGHTS': CONFIRMATION_WEIGHTS.copy(),
            # USE_TRAILING_STOPS and MAX_OPEN_POSITIONS will be set from combo
        }

        if optimization_type == 'basic':
            long_risk_val = combo[0]
            long_reward_val = combo[1]
            short_risk_val = combo[2]
            short_reward_val = combo[3]
            pos_size_val = combo[4]
            tech_params['USE_TRAILING_STOPS'] = combo[5]
            tech_params['MAX_OPEN_POSITIONS'] = combo[6]
            # Technical indicators and scoring weights remain default

        elif optimization_type == 'technical':
            long_risk_val = DEFAULT_LONG_RISK
            long_reward_val = DEFAULT_LONG_REWARD
            short_risk_val = DEFAULT_SHORT_RISK
            short_reward_val = DEFAULT_SHORT_REWARD
            pos_size_val = DEFAULT_POSITION_SIZE

            tech_params['USE_TRAILING_STOPS'] = combo[0]
            tech_params['MAX_OPEN_POSITIONS'] = combo[1]
            
            # Scoring weights start after USE_TRAILING_STOPS and MAX_OPEN_POSITIONS
            # num_params_before_weights is 2 for 'technical'
            weight_values = combo[num_params_before_weights:]
            
            trend_weight_keys = list(TREND_WEIGHTS.keys())
            conf_weight_keys = list(CONFIRMATION_WEIGHTS.keys())
            
            current_trend_weights_dict = tech_params['TREND_WEIGHTS']
            idx_offset = 0
            for i, key in enumerate(trend_weight_keys):
                if (idx_offset + i) < len(weight_values):
                    current_trend_weights_dict[key] = weight_values[idx_offset + i]
            
            idx_offset += len(trend_weight_keys)
            current_confirmation_weights_dict = tech_params['CONFIRMATION_WEIGHTS']
            for i, key in enumerate(conf_weight_keys):
                if (idx_offset + i) < len(weight_values):
                    current_confirmation_weights_dict[key] = weight_values[idx_offset + i]
        else:
            # Should not happen if optimize_parameters is structured correctly
            continue

        # Validation for risk/reward
        if long_reward_val > 0 and long_risk_val >= long_reward_val: continue
        if short_reward_val > 0 and short_risk_val >= short_reward_val: continue
        
        param_grid.append((
            ticker, base_df, # Pass base_df by reference; copy will be made in parameter_test
            long_risk_val, long_reward_val,
            short_risk_val, short_reward_val,
            pos_size_val, tech_params
        ))
        
    return list(param_grid)

# --------------------------------------------------------------------------------------------------------------------------

def _generate_parameter_grid(param_lists, num_params_before_weights, optimization_type, ticker, base_df):
    """
    Generates a grid of all possible parameter combinations.
    Args:
        param_lists: List of parameter value lists.
        num_params_before_weights: Number of parameters in combo before scoring weights (specific to 'technical')
        optimization_type: String, 'basic' or 'technical'
        ticker: The ticker symbol string.
        base_df: The base DataFrame for the ticker (passed by reference, copied in parameter_test)
    Returns:
        List of parameter combination tuples.
    """
    generated_params = deque()
    
    current_total_combos = 1
    for p_list_item in param_lists: # Renamed p_list to p_list_item to avoid conflict
        current_total_combos *= len(p_list_item) if p_list_item else 1

    all_combinations = itertools.product(*param_lists)

    print(f"Generating and filtering all {current_total_combos} parameter combinations for {optimization_type}...")
    for combo in tqdm(all_combinations, total=current_total_combos, desc=f"Generating {optimization_type} Combinations", unit="combo"):
        tech_params = {
            'FAST': FAST, 'SLOW': SLOW,
            'RSI_OVERSOLD': RSI_OVERSOLD, 'RSI_OVERBOUGHT': RSI_OVERBOUGHT,
            'DEVS': DEVS, 'ADX_THRESHOLD': ADX_THRESHOLD_DEFAULT,
            'MIN_BUY_SCORE': MIN_BUY_SCORE_DEFAULT, 'MIN_SELL_SCORE': MIN_SELL_SCORE_DEFAULT,
            'REQUIRE_CLOUD': REQUIRE_CLOUD_DEFAULT,
            'WEEKLY_MA_PERIOD': WEEKLY_MA_PERIOD,
            'TREND_WEIGHTS': TREND_WEIGHTS.copy(),
            'CONFIRMATION_WEIGHTS': CONFIRMATION_WEIGHTS.copy(),
        }

        if optimization_type == 'basic':
            long_risk_val = combo[0]
            long_reward_val = combo[1]
            short_risk_val = combo[2]
            short_reward_val = combo[3]
            pos_size_val = combo[4]
            tech_params['USE_TRAILING_STOPS'] = combo[5]
            tech_params['MAX_OPEN_POSITIONS'] = combo[6]

        elif optimization_type == 'technical':
            long_risk_val = DEFAULT_LONG_RISK
            long_reward_val = DEFAULT_LONG_REWARD
            short_risk_val = DEFAULT_SHORT_RISK
            short_reward_val = DEFAULT_SHORT_REWARD
            pos_size_val = DEFAULT_POSITION_SIZE

            tech_params['USE_TRAILING_STOPS'] = combo[0]
            tech_params['MAX_OPEN_POSITIONS'] = combo[1]
            
            weight_values = combo[num_params_before_weights:] # num_params_before_weights is 2
            
            trend_weight_keys = list(TREND_WEIGHTS.keys())
            conf_weight_keys = list(CONFIRMATION_WEIGHTS.keys())
            
            current_trend_weights_dict = tech_params['TREND_WEIGHTS']
            idx_offset = 0
            for i, key in enumerate(trend_weight_keys):
                if (idx_offset + i) < len(weight_values):
                    current_trend_weights_dict[key] = weight_values[idx_offset + i]
            
            idx_offset += len(trend_weight_keys)
            current_confirmation_weights_dict = tech_params['CONFIRMATION_WEIGHTS']
            for i, key in enumerate(conf_weight_keys):
                if (idx_offset + i) < len(weight_values):
                    current_confirmation_weights_dict[key] = weight_values[idx_offset + i]
        else:
            continue

        if long_reward_val > 0 and long_risk_val >= long_reward_val: continue
        if short_reward_val > 0 and short_risk_val >= short_reward_val: continue
        
        generated_params.append((
            ticker, base_df, # Pass base_df by reference; copy will be made in parameter_test
            long_risk_val, long_reward_val,
            short_risk_val, short_reward_val,
            pos_size_val, tech_params
        ))
        
    return list(generated_params)

# --------------------------------------------------------------------------------------------------------------------------

def optimize_parameters(ticker=TICKER, 
                        visualize_best=VISUALIZE_BEST_DEFAULT, 
                        optimization_type=OPTIMIZATION_TYPE_DEFAULT):
    """
    Optimize strategy parameters using smart sampling and parallel processing.
    """
    print(f"Starting parameter optimization for {ticker} of type: {optimization_type}...")
    
    # Ensure ticker is a string for yf.download
    current_ticker_str = ticker[0] if isinstance(ticker, list) else ticker

    base_df = yf.download(current_ticker_str, period="5y", progress=False, auto_adjust=True)
    if base_df.empty:
        print(f"Error: No data downloaded for {current_ticker_str}. Aborting optimization.")
        return None, []
    
    use_trailing_stops_options_range = [True, False] 

    if optimization_type == 'basic':
        long_risks = [0.01, 0.02, 0.03, 0.04, 0.05]
        long_rewards = [0.02, 0.03, 0.05, 0.07, 0.10]
        short_risks = [0.01, 0.02, 0.03, 0.04, 0.05]
        short_rewards = [0.02, 0.03, 0.05, 0.07, 0.10]
        position_sizes = [0.02, 0.05, 0.10, 0.15, 0.20] # Percentage of portfolio for one position
        max_open_positions_options = [3, 5, 10, 15]
        
        param_lists = [
            long_risks, long_rewards, short_risks, short_rewards, position_sizes,
            use_trailing_stops_options_range, 
            max_open_positions_options  
        ]
        
        num_params_before_weights = len(param_lists) 

    elif optimization_type == 'technical':
        
        # Optimize execution rules and scoring weights
        max_open_positions_options = [5, 10, 15] 

        trend_primary_weights = [2.0, 3.0, 4.0, 5.0]
        trend_strength_weights = [1.5, 2.5, 3.5, 4.5]
        trend_cloud_confirmation_weights = [1.0, 1.5, 2.0, 2.5] 
        trend_kumo_confirmation_weights = [0.5, 1.0, 1.5, 2.0]
        trend_ma_alignment_weights = [1.0, 1.5, 2.0, 2.5]
        
        conf_rsi_weights = [0.8, 1.0, 1.2, 1.5]
        conf_mfi_weights = [0.5, 0.8, 1.0, 1.2]
        conf_volume_weights = [0.5, 1.0, 1.5, 2.0]
        conf_bollinger_weights = [0.3, 0.5, 0.8, 1.0]
        
        param_lists = [
            use_trailing_stops_options_range, 
            max_open_positions_options, 
            # Scoring Weights (9 lists)
            trend_primary_weights, trend_strength_weights, trend_cloud_confirmation_weights,
            trend_kumo_confirmation_weights, trend_ma_alignment_weights,
            conf_rsi_weights, conf_mfi_weights, conf_volume_weights, conf_bollinger_weights
        ]
        # Parameters before scoring weights are: use_trailing_stops, max_open_positions
        num_params_before_weights = 2 
    else:
        print(f"Unknown optimization_type: {optimization_type}. Supported types: 'basic', 'technical'. Aborting.")
        return None, []
        
    # Calculate total combinations
    total_combos = 1
    for p_list in param_lists:
        total_combos *= len(p_list) if p_list else 1
    
    print(f"Generating all {total_combos} parameter combinations...")
    param_grid = _generate_parameter_grid(param_lists, num_params_before_weights, optimization_type, current_ticker_str, base_df)

    if not param_grid:
        print("No valid parameter combinations generated. Aborting.")
        return None, []

    starttime = datetime.now()
    print(f"Testing {len(param_grid)} parameter combinations using multiprocessing...")
    
    # Create progress bar for parameter testing
    with tqdm(total=len(param_grid), desc="Parameter Testing Progress", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        results_list = []
        with mp.Pool(processes=max(1, mp.cpu_count() - 2)) as pool:
            for result in pool.imap_unordered(parameter_test, param_grid):
                results_list.append(result)
                pbar.update(1)
                
                # Display current best result
                if result and 'sharpe' in result and result['sharpe'] is not None:
                    current_best = max(results_list, key=lambda x: x['sharpe'])
                    pbar.set_postfix({
                        'Best Sharpe': f"{current_best['sharpe']:.2f}",
                        'Win Rate': f"{current_best['win_rate']:.1f}%"
                    })
        
    best_result = max(results_list, key=lambda x: x['sharpe'])
    sorted_results = sorted(results_list, key=lambda x: x['sharpe'], reverse=True)
    
    print("\nTop 5 Parameter Combinations:")
    
    # Define base headers common to all types
    base_headers = ["L.Risk", "L.Reward", "S.Risk", "S.Reward", "Pos Size", "MaxPos"]
    perf_headers = ["Win%", "Profit%", "MaxDD%", "Sharpe", "# Trades"]
    
    headers = list(base_headers) # Start with a copy

    if optimization_type == 'technical':
        # For 'technical', Risk/Reward/Pos.Size are fixed but still good to show.
        # Technical indicator parameters are fixed (not shown as optimized).
        # Scoring weights are optimized.
        headers.extend([
            "T.PrimW", "T.StrW", "T.CloudW", "T.KumoW", "T.MAW",
            "C.RSIW", "C.MFIW", "C.VolW", "C.BBW"
        ])
    # For 'basic', only base_headers and perf_headers are needed.
    # Technical indicators and weights are fixed to default and not shown as optimized.

    headers.extend(perf_headers)
    
    rows = []
    for i, result_item in enumerate(sorted_results[:5]): # Renamed result to result_item
        tech_p_res = result_item.get('tech_params', {}) # Ensure tech_params exists
        
        row_data = [
            f"{result_item['long_risk']*100:.1f}%", f"{result_item['long_reward']*100:.1f}%",
            f"{result_item['short_risk']*100:.1f}%", f"{result_item['short_reward']*100:.1f}%",
            f"{result_item['position_size']*100:.1f}%", # This is % of portfolio per trade
            f"{tech_p_res.get('USE_TRAILING_STOPS', USE_TRAILING_STOPS_DEFAULT)}",
            f"{tech_p_res.get('MAX_OPEN_POSITIONS', MAX_OPEN_POSITIONS)}",
        ]

        if optimization_type == 'technical':
            tw = tech_p_res.get('TREND_WEIGHTS', TREND_WEIGHTS) 
            cw = tech_p_res.get('CONFIRMATION_WEIGHTS', CONFIRMATION_WEIGHTS)
            row_data.extend([
                f"{tw.get('primary_trend', TREND_WEIGHTS['primary_trend']):.1f}", 
                f"{tw.get('trend_strength', TREND_WEIGHTS['trend_strength']):.1f}",
                f"{tw.get('cloud_confirmation', TREND_WEIGHTS['cloud_confirmation']):.1f}", 
                f"{tw.get('kumo_confirmation', TREND_WEIGHTS['kumo_confirmation']):.1f}",
                f"{tw.get('ma_alignment', TREND_WEIGHTS['ma_alignment']):.1f}",
                f"{cw.get('rsi', CONFIRMATION_WEIGHTS['rsi']):.1f}", 
                f"{cw.get('mfi', CONFIRMATION_WEIGHTS['mfi']):.1f}",
                f"{cw.get('volume', CONFIRMATION_WEIGHTS['volume']):.1f}", 
                f"{cw.get('bollinger', CONFIRMATION_WEIGHTS['bollinger']):.1f}",
            ])
            
        row_data.extend([
            f"{result_item.get('win_rate', 0):.1f}", 
            f"{result_item.get('net_profit_pct', -100):.1f}",
            f"{result_item.get('max_drawdown', 100):.1f}", 
            f"{result_item.get('sharpe', -10):.2f}",
            f"{result_item.get('num_trades', 0)}"
        ])
        rows.append(row_data)
        
    if rows:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print("No results to display in table.")
    
    if visualize_best and best_result: # Check if best_result is not None
        print(f"\nRunning detailed backtest with best parameters for {current_ticker_str}:")
        best_tech_params = best_result.get('tech_params', {})
        
        # For visualization, always use the default technical indicator periods,
        # as they are not optimized in 'basic' or 'technical' types.
        # The optimized parts are risk/reward or scoring weights.
        df_with_indicators_viz = prepare_data(
            base_df.copy(), 
            fast=FAST, # Use global default
            slow=SLOW, # Use global default
            rsi_oversold=RSI_OVERSOLD, # Use global default
            rsi_overbought=RSI_OVERBOUGHT, # Use global default
            devs=DEVS, # Use global default
            weekly_ma_period_val=WEEKLY_MA_PERIOD # Use global default
        )
        
        final_run_result = test( 
            df_with_indicators_viz,
            current_ticker_str, 
            long_risk=best_result['long_risk'],
            long_reward=best_result['long_reward'],
            short_risk=best_result['short_risk'],
            short_reward=best_result['short_reward'],
            position_size=best_result['position_size'],
            # Get optimized execution rules from best_tech_params
            use_trailing_stops=best_tech_params.get('USE_TRAILING_STOPS', USE_TRAILING_STOPS_DEFAULT),
            max_positions_param=best_tech_params.get('MAX_OPEN_POSITIONS', MAX_OPEN_POSITIONS),
            # Technical indicators are fixed to defaults
            fast_period=FAST,
            slow_period=SLOW,
            weekly_ma_p=WEEKLY_MA_PERIOD,
            adx_thresh=ADX_THRESHOLD_DEFAULT,
            min_buy_s=MIN_BUY_SCORE_DEFAULT,
            min_sell_s=MIN_SELL_SCORE_DEFAULT,
            req_cloud=REQUIRE_CLOUD_DEFAULT,
            # Get optimized scoring weights from best_tech_params
            trend_weights_param=best_tech_params.get('TREND_WEIGHTS', TREND_WEIGHTS), 
            confirmation_weights_param=best_tech_params.get('CONFIRMATION_WEIGHTS', CONFIRMATION_WEIGHTS)
        )
    
    end_time = datetime.now()
    elapsed_time = end_time - starttime
    print(f"\nParameter optimization for {current_ticker_str} completed in {elapsed_time}")
    
    return best_result, sorted_results

# --------------------------------------------------------------------------------------------------------------------------

def parameter_test(args):
    """
    Parameter testing function with GPU acceleration when available.
    """
    global HAS_GPU
    
    ticker, base_df_original, long_risk, long_reward, short_risk, short_reward, position_size, tech_params = args
    
    try:
        # Extract parameters and prepare data
        current_fast_period = tech_params.get('FAST', FAST)
        current_slow_period = tech_params.get('SLOW', SLOW)
        current_weekly_ma_period = tech_params.get('WEEKLY_MA_PERIOD', WEEKLY_MA_PERIOD)
        current_trend_weights = tech_params.get('TREND_WEIGHTS', TREND_WEIGHTS)
        current_confirmation_weights = tech_params.get('CONFIRMATION_WEIGHTS', CONFIRMATION_WEIGHTS)
        current_use_trailing_stops = tech_params.get('USE_TRAILING_STOPS', USE_TRAILING_STOPS_DEFAULT)
        current_max_open_positions = tech_params.get('MAX_OPEN_POSITIONS', MAX_OPEN_POSITIONS)

        param_array = np.array([[long_risk, long_reward, short_risk, short_reward, position_size]], dtype=np.float32)
        
        df_prepared = prepare_data(
            base_df_original.copy(),
            fast=current_fast_period,
            slow=current_slow_period,
            rsi_oversold=tech_params.get('RSI_OVERSOLD', RSI_OVERSOLD),
            rsi_overbought=tech_params.get('RSI_OVERBOUGHT', RSI_OVERBOUGHT),
            devs=tech_params.get('DEVS', DEVS),
            weekly_ma_period_val=current_weekly_ma_period
        )

        if HAS_GPU:
            try:
                # GPU path
                results = _evaluate_parameters_batch_gpu(
                    param_array,
                    df_prepared['Close'].values,
                    df_prepared[f"{current_fast_period}_ma"].values,
                    df_prepared[f"{current_slow_period}_ma"].values
                )
                sharpe = float(results[0])
                
            except Exception as gpu_error:
                print(f"GPU processing failed, falling back to CPU: {gpu_error}")
                HAS_GPU = False
                # Fall back to CPU evaluation
                results = _evaluate_parameters_batch_cpu(
                    param_array,
                    df_prepared['Close'].values,
                    df_prepared[f"{current_fast_period}_ma"].values,
                    df_prepared[f"{current_slow_period}_ma"].values
                )
                sharpe = float(results[0])
        else:
            # CPU path
            results = _evaluate_parameters_batch_cpu(
                param_array,
                df_prepared['Close'].values,
                df_prepared[f"{current_fast_period}_ma"].values,
                df_prepared[f"{current_slow_period}_ma"].values
            )
            sharpe = float(results[0])
        
        # Set up column names for moving averages
        fast_ma_col = f"{current_fast_period}_ma"
        slow_ma_col = f"{current_slow_period}_ma"
        weekly_ma_col = f"Weekly_MA{current_weekly_ma_period}"
        
        # Run momentum strategy
        trade_log, trade_stats, portfolio_equity, trade_returns = momentum(
            df_prepared,
            fast_ma_col_name=fast_ma_col,
            slow_ma_col_name=slow_ma_col,
            weekly_ma_col_name=weekly_ma_col,
            long_risk=long_risk,
            long_reward=long_reward,
            short_risk=short_risk,
            short_reward=short_reward,
            position_size=position_size,
            max_positions=current_max_open_positions,
            use_trailing_stops=current_use_trailing_stops,
            adx_threshold_sig=tech_params.get('ADX_THRESHOLD', ADX_THRESHOLD_DEFAULT),
            min_buy_score_sig=tech_params.get('MIN_BUY_SCORE', MIN_BUY_SCORE_DEFAULT),
            min_sell_score_sig=tech_params.get('MIN_SELL_SCORE', MIN_SELL_SCORE_DEFAULT),
            require_cloud_sig=tech_params.get('REQUIRE_CLOUD', REQUIRE_CLOUD_DEFAULT),
            trend_weights_sig=current_trend_weights,
            confirmation_weights_sig=current_confirmation_weights
        )
        
        return {
            'ticker': ticker,
            'long_risk': long_risk,
            'long_reward': long_reward,
            'short_risk': short_risk,
            'short_reward': short_reward,
            'position_size': position_size,
            'tech_params': tech_params,
            'win_rate': trade_stats.get('Win Rate', 0),
            'net_profit_pct': trade_stats.get('Net Profit (%)', -100),
            'max_drawdown': trade_stats.get('Max Drawdown (%)', 100),
            'sharpe': trade_stats.get('Sharpe Ratio', -10),
            'profit_factor': trade_stats.get('Profit Factor', 0),
            'num_trades': trade_stats.get('Total Trades', 0)
        }
        
    except Exception as e:
        current_params_str = (
            f"long_risk={long_risk}, long_reward={long_reward}, "
            f"short_risk={short_risk}, short_reward={short_reward}, "
            f"pos_size={position_size}, tech_params_keys={list(tech_params.keys()) if tech_params else 'None'}"
        )
        print(f"Error testing parameters for {ticker if ticker else 'UnknownTicker'} ({current_params_str}): {e}")
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
            'sharpe': -10.0,
            'profit_factor': 0.0,
            'num_trades': 0
        }

# --------------------------------------------------------------------------------------------------------------------------

def test(df_input,
         TICKER_STR, 
         long_risk=DEFAULT_LONG_RISK, long_reward=DEFAULT_LONG_REWARD,
         short_risk=DEFAULT_SHORT_RISK, short_reward=DEFAULT_SHORT_REWARD,
         position_size=DEFAULT_POSITION_SIZE, 
         use_trailing_stops=USE_TRAILING_STOPS_DEFAULT, # Added to signature
         max_positions_param=MAX_OPEN_POSITIONS, # Added to signature
         fast_period=FAST, slow_period=SLOW, weekly_ma_p=WEEKLY_MA_PERIOD,
         adx_thresh=ADX_THRESHOLD_DEFAULT, min_buy_s=MIN_BUY_SCORE_DEFAULT,
         min_sell_s=MIN_SELL_SCORE_DEFAULT, req_cloud=REQUIRE_CLOUD_DEFAULT,
         trend_weights_param=None, confirmation_weights_param=None): 
    
    df = df_input.copy()

    current_fast_ma_col = f"{fast_period}_ma"
    current_slow_ma_col = f"{slow_period}_ma"
    current_weekly_ma_col = f"Weekly_MA{weekly_ma_p}"

    final_trend_weights = trend_weights_param if trend_weights_param is not None else TREND_WEIGHTS
    final_confirmation_weights = confirmation_weights_param if confirmation_weights_param is not None else CONFIRMATION_WEIGHTS
    
    trade_log, trade_stats, portfolio_equity, trade_returns = momentum(
        df, 
        fast_ma_col_name=current_fast_ma_col,
        slow_ma_col_name=current_slow_ma_col,
        weekly_ma_col_name=current_weekly_ma_col,
        long_risk=long_risk, long_reward=long_reward,
        short_risk=short_risk, short_reward=short_reward,
        position_size=position_size, 
        max_positions=max_positions_param, # Use passed parameter
        risk_free_rate=0.04,
        use_trailing_stops=use_trailing_stops, # Use passed parameter
        adx_threshold_sig=adx_thresh,
        min_buy_score_sig=min_buy_s,
        min_sell_score_sig=min_sell_s,
        require_cloud_sig=req_cloud,
        trend_weights_sig=final_trend_weights, 
        confirmation_weights_sig=final_confirmation_weights
    )
    
    # Add asset returns to dataframe for comparison
    df.loc[:, 'Asset_Returns'] = df['Close'].pct_change().fillna(0).cumsum()
    
    # Convert equity curve to returns for comparison
    df.loc[:, 'Strategy_Returns'] = (portfolio_equity / portfolio_equity.iloc[0] - 1)
    
    first_close = df['Close'].iloc[0] 
    last_close = df['Close'].iloc[-1]
    buy_hold_return = ((last_close / first_close) - 1) * 100
    
    peak_equity = portfolio_equity.max()
    exposure_time = ((df.index[-1] - df.index[0]).days)
    
    if trade_log:
        best_trade_pct = max([t['PnL'] for t in trade_log if t['PnL'] is not None], default=0) / portfolio_equity.iloc[0] * 100 if portfolio_equity.iloc[0] !=0 else 0
        worst_trade_pct = min([t['PnL'] for t in trade_log if t['PnL'] is not None], default=0) / portfolio_equity.iloc[0] * 100 if portfolio_equity.iloc[0] !=0 else 0
        avg_trade_pnl_values = [t['PnL'] for t in trade_log if t['PnL'] is not None]
        avg_trade_pct = (sum(avg_trade_pnl_values) / len(avg_trade_pnl_values) / portfolio_equity.iloc[0] * 100) if avg_trade_pnl_values and portfolio_equity.iloc[0] !=0 else 0
        
        durations = [t['Duration'] for t in trade_log if t['Duration'] is not None]
        max_duration = max(durations) if durations else 0
        avg_duration = sum(durations) / len(durations) if durations else 0
    else:
        best_trade_pct = worst_trade_pct = avg_trade_pct = max_duration = avg_duration = 0

    print(f"\n=== {TICKER_STR} STRATEGY SUMMARY ===") 
    print(f"Long Risk: {long_risk*100:.1f}% | Long Reward: {long_reward*100:.1f}%")
    print(f"Short Risk: {short_risk*100:.1f}% | Short Reward: {short_reward*100:.1f}%")
    print(f"Position Size: {position_size*100:.1f}%")
    print(f"Use Trailing Stops: {use_trailing_stops}")
    print(f"Max Open Positions: {max_positions_param}") # Added print for max positions
    
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
    
    print(f"\n=== {TICKER_STR} TRADE SUMMARY ===") 
    trade_metrics = [
        ["Total Trades", f"{trade_stats['Total Trades']:.0f}"], 
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
    
    # create_backtest_charts(df, portfolio_equity, trade_stats, trade_log, TICKER_STR) 
    
    return {
        'TICKER': TICKER_STR, 
        'df': df, 'equity': portfolio_equity, 'trade_stats': trade_stats, 'trade_log': trade_log,
        'long_risk': long_risk, 'long_reward': long_reward,
        'short_risk': short_risk, 'short_reward': short_reward,
        'use_trailing_stops': use_trailing_stops, # Added to result
        'max_positions': max_positions_param # Added to result
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
    
    if isinstance(TICKER, list):
        # Download for a single ticker but don't use a list
        ticker_str = TICKER[0]
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
        
        # When calling prepare_data directly, use the global defaults or specific values
        current_fast = FAST
        current_slow = SLOW
        current_weekly_ma = WEEKLY_MA_PERIOD
        
        df_prepared = prepare_data(
            df,
            fast=current_fast,
            slow=current_slow,
            weekly_ma_period_val=current_weekly_ma
        )
        
        result = test(
            df_prepared,
            TICKER if not isinstance(TICKER, list) else TICKER[0],
            long_risk=DEFAULT_LONG_RISK,
            long_reward=DEFAULT_LONG_REWARD, 
            short_risk=DEFAULT_SHORT_RISK,
            short_reward=DEFAULT_SHORT_REWARD,
            position_size=DEFAULT_POSITION_SIZE,
            use_trailing_stops=USE_TRAILING_STOPS_DEFAULT,
            # Pass the periods used for prepare_data
            fast_period=current_fast,
            slow_period=current_slow,
            weekly_ma_p=current_weekly_ma,
            # Pass default signal parameters
            adx_thresh=ADX_THRESHOLD_DEFAULT,
            min_buy_s=MIN_BUY_SCORE_DEFAULT,
            min_sell_s=MIN_SELL_SCORE_DEFAULT,
            req_cloud=REQUIRE_CLOUD_DEFAULT
        )
    
    return result

if __name__ == "__main__":
    main()