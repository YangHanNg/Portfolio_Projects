import pandas as pd
import numpy as np
import yfinance as yf
from tabulate import tabulate  
import pandas_ta as ta 
from datetime import datetime, timedelta
import multiprocessing as mp 
import itertools
import matplotlib as plt
import seaborn as sns
import gc

# --------------------------------------------------------------------------------------------------------------------------

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

# Risk & Reward
DEFAULT_LONG_RISK = 0.03  
DEFAULT_LONG_REWARD = 0.06  
DEFAULT_SHORT_RISK = 0.02  
DEFAULT_SHORT_REWARD = 0.04  
DEFAULT_POSITION_SIZE = 0.10  
MAX_OPEN_POSITIONS = 5 

# Trade Management
ADX_THRESHOLD_DEFAULT = 25
MIN_BUY_SCORE_DEFAULT = 5.0
MIN_SELL_SCORE_DEFAULT = 5.0
REQUIRE_CLOUD_DEFAULT = True
USE_TRAILING_STOPS_DEFAULT = True

# Script Execution Defaults 
OPTIMIZE_DEFAULT = False
VISUALIZE_BEST_DEFAULT = False
OPTIMIZATION_TYPE_DEFAULT = 'basic'

TREND_WEIGHTS = {
    'primary_trend': 3.0,
    'trend_strength': 2.5,
    'cloud_confirmation': 2.0,
    'kumo_confirmation': 1.0,
    'ma_alignment': 1.5  
}

CONFIRMATION_WEIGHTS = {
    'rsi': 1.0,
    'mfi': 1.0,
    'volume': 0.8,
    'bollinger': 0.5
}

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
    # Strategy execution parameters
    long_risk=DEFAULT_LONG_RISK, long_reward=DEFAULT_LONG_REWARD,
    short_risk=DEFAULT_SHORT_RISK, short_reward=DEFAULT_SHORT_REWARD,
    position_size=DEFAULT_POSITION_SIZE, max_positions=MAX_OPEN_POSITIONS,
    risk_free_rate=0.04, leverage_ratio=LEVERAGE,
    use_trailing_stops=USE_TRAILING_STOPS_DEFAULT,
    # Signal generation parameters (to be passed to vectorized_signals)
    adx_threshold_sig=ADX_THRESHOLD_DEFAULT,
    min_buy_score_sig=MIN_BUY_SCORE_DEFAULT,
    min_sell_score_sig=MIN_SELL_SCORE_DEFAULT,
    require_cloud_sig=REQUIRE_CLOUD_DEFAULT
):
    
    """
    Optimized momentum strategy execution.
    Assumes df_with_indicators has been processed by prepare_data().
    Internally calls vectorized_signals().
    """
    if df_with_indicators.empty:
        print("Warning: Empty dataframe (df_with_indicators) provided to momentum.")
        return [], create_empty_stats(), pd.Series(dtype='float64'), pd.Series(dtype='float64')

    # 1. Generate signals using the indicator-laden DataFrame
    signals_df = signals(
        df_with_indicators, # Pass the DataFrame that already has indicators
        adx_threshold_val=adx_threshold_sig,
        min_buy_score_val=min_buy_score_sig,
        min_sell_score_val=min_sell_score_sig,
        require_cloud_val=require_cloud_sig,
        fast_ma_col=fast_ma_col_name,
        slow_ma_col=slow_ma_col_name,
        weekly_ma_col_name=weekly_ma_col_name # Pass the actual column name
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
    for i in range(1, len(df_with_indicators)):
        current_trade_date = df_with_indicators.index[i]
        transaction_price = df_with_indicators['Open'].iloc[i]
        
        # Critical data checks for the current row
        # Data for decision making (ATR, ADX, confirmations) is from the PREVIOUS day's close
        previous_day_data_row = df_with_indicators.iloc[i-1]
        previous_day_atr = previous_day_data_row['ATR']
        previous_day_adx = previous_day_data_row['ADX']
        if pd.isna(transaction_price) or pd.isna(previous_day_atr) or pd.isna(previous_day_adx):
            equity_curve.iloc[i] = equity_curve.iloc[i-1]
            returns_series.iloc[i] = 0.0
            continue

        # Get signals from the pre-calculated signals_df
        buy_signal_for_today = signals_df['buy_signal'].iloc[i-1]
        sell_signal_for_today = signals_df['sell_signal'].iloc[i-1]
        
        # Get other confirmations from the main data row
        previous_day_volume_confirmed = previous_day_data_row.get('volume_confirmed', False)
        previous_day_weekly_uptrend = previous_day_data_row.get('weekly_uptrend', True) 

        if use_trailing_stops:
            long_manager.update_trailing_stops(transaction_price, previous_day_atr, previous_day_adx)
            short_manager.update_trailing_stops(transaction_price, previous_day_atr, previous_day_adx)

        long_manager.process_exits(current_trade_date, transaction_price,
                                   signal_exit=sell_signal_for_today, 
                                   trend_valid=(previous_day_adx > adx_threshold_sig)) # Use the same ADX threshold
        short_manager.process_exits(current_trade_date, transaction_price,
                                     signal_exit=buy_signal_for_today,
                                     trend_valid=(previous_day_adx > adx_threshold_sig))

        entry_params_base = {
            'price': transaction_price, # Trade at Open of current day
            'atr': previous_day_atr,    # ATR from previous day
            'adx': previous_day_adx,    # ADX from previous day
            'position_size': position_size, 
            'leverage': leverage_ratio
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


    # 4. Combine trade logs and calculate statistics
    combined_trade_log = long_manager.trade_log + short_manager.trade_log
    combined_wins = long_manager.wins + short_manager.wins
    combined_losses = long_manager.losses + short_manager.losses

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

def _calculate_trend_score(conditions, weights, direction='long'):
    # Determine the shape from a base condition array
    base_condition_for_shape = conditions.get('primary_trend_long', conditions.get('primary_trend_short'))
    if base_condition_for_shape is False or not hasattr(base_condition_for_shape, 'shape'): # Fallback if no primary trend keys
        # Attempt to get shape from another common condition or default to a small array to avoid errors
        # This part might need adjustment based on how 'conditions' can be structured
        fallback_condition = conditions.get('trend_strength_ok', np.array([False])) # Example fallback
        score_shape = fallback_condition.shape if hasattr(fallback_condition, 'shape') else (0,)
    else:
        score_shape = base_condition_for_shape.shape
    
    score = np.zeros(score_shape, dtype=float) # Initialize score as a float array

    if direction == 'long':
        score += conditions.get('primary_trend_long', False).astype(float) * weights.get('primary_trend', 0)
        score += conditions.get('strong_cloud_support', False).astype(float) * weights.get('cloud_confirmation', 0)
        score += conditions.get('ma_buy', False).astype(float) * weights.get('ma_alignment', 0)
    else: # short
        score += conditions.get('primary_trend_short', False).astype(float) * weights.get('primary_trend', 0)
        score += conditions.get('below_cloud', False).astype(float) * weights.get('cloud_confirmation', 0)
        score += conditions.get('ma_sell', False).astype(float) * weights.get('ma_alignment', 0)
    score += conditions.get('trend_strength_ok', False).astype(float) * weights.get('trend_strength', 0) # Common for both
    return score

# --------------------------------------------------------------------------------------------------------------------------

def _calculate_confirmation_score(conditions, weights, rsi_np_arr, mfi_np_arr, direction='long'):
    # Determine the shape from a base condition array
    base_condition_for_shape = conditions.get('volume_ok', False)
    if base_condition_for_shape is False or not hasattr(base_condition_for_shape, 'shape'):
         # Attempt to get shape from another common condition or default
        fallback_condition = rsi_np_arr if rsi_np_arr is not None and hasattr(rsi_np_arr, 'shape') else np.array([False])
        score_shape = fallback_condition.shape
    else:
        score_shape = base_condition_for_shape.shape

    score = np.zeros(score_shape, dtype=float) # Initialize score as a float array

    if direction == 'long':
        score += conditions.get('rsi_confirmation_long', False).astype(float) * weights.get('rsi', 0)
        score += conditions.get('mfi_confirmation_long', False).astype(float) * weights.get('mfi', 0)
        score += conditions.get('bb_buy', False).astype(float) * weights.get('bollinger', 0)
    else: # short
        score += conditions.get('rsi_confirmation_short', False).astype(float) * weights.get('rsi', 0)
        score += conditions.get('mfi_confirmation_short', False).astype(float) * weights.get('mfi', 0)
        score += conditions.get('bb_sell', False).astype(float) * weights.get('bollinger', 0)
    score += conditions.get('volume_ok', False).astype(float) * weights.get('volume', 0) # Common for both
    return score

# --------------------------------------------------------------------------------------------------------------------------

def signals(df, adx_threshold_val, min_buy_score_val, min_sell_score_val, require_cloud_val, # require_cloud_val might become redundant
                       fast_ma_col, slow_ma_col, weekly_ma_col_name):
    """
    Generates trading signals for the entire DataFrame using vectorized operations, NumPy,
    primary trend filters, secondary confirmations, and a weighted scoring approach.
    """
    signals_df = pd.DataFrame(index=df.index)

    required_indicator_cols = [
        fast_ma_col, slow_ma_col, weekly_ma_col_name, 'RSI', 'MFI', 'Close', 'Lower_Band', 'Upper_Band',
        'ATR', 'ADX', 'SAR', 'ROC', 'Tenkan_sen', 'Kijun_sen', 'Senkou_span_A',
        'Senkou_span_B', 'Volume', 'Volume_MA20', 'Open', 'High', 'Low' # Added OHLC for completeness
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
    

    atr_ratio_np = np.full_like(close_np, 0.02)
    valid_close_mask = close_np != 0
    atr_ratio_np[valid_close_mask] = atr_np[valid_close_mask] / close_np[valid_close_mask]
    atr_ratio_np = np.nan_to_num(atr_ratio_np, nan=0.02, posinf=0.02, neginf=0.02)

    actual_min_buy_np, actual_min_sell_np = _dynamic_thresholds(
        atr_ratio_np, min_buy_score_val, min_sell_score_val
    )
    adx_strength_np = np.clip((adx_np - 20.0) / 30.0, 0.0, 1.0)
    trend_multiplier_np = 1.0 + adx_strength_np

    # Calculate Ichimoku components for conditions
    span_max_np = np.maximum(spanA_np, spanB_np)
    span_min_np = np.minimum(spanA_np, spanB_np)
    min_cloud_thickness_val_np = np.maximum(0.005 * close_np, 0.005)
    min_cloud_thickness_val_np[~valid_close_mask] = 0.005
    cloud_thickness_np = np.abs(spanA_np - spanB_np)

    # 2. Calculate Primary and Secondary Conditions
    conditions = {}
    # Primary Trend Filters
    conditions['primary_trend_long'] = (close_np > fast_ma_np) & (close_np > slow_ma_np) & (close_np > weekly_ma_np)
    conditions['primary_trend_short'] = (close_np < fast_ma_np) & (close_np < slow_ma_np) & (close_np < weekly_ma_np)
    conditions['trend_strength_ok'] = adx_np > adx_threshold_val # Using param, example had 25

    # Secondary Confirmation Indicators / Scoring Components
    # Ichimoku
    conditions['strong_cloud_support'] = (close_np > span_max_np) & (cloud_thickness_np > min_cloud_thickness_val_np)
    conditions['below_cloud'] = close_np < span_min_np
    # Kumo Breakout/Breakdown
    conditions['kumo_breakout'] = ((tenkan_np > kijun_np) & (close_np > span_max_np) & (cloud_thickness_np > min_cloud_thickness_val_np))
    conditions['kumo_breakdown'] = ((tenkan_np < kijun_np) & (close_np < span_min_np) & (cloud_thickness_np > min_cloud_thickness_val_np))
    
    # Momentum
    conditions['rsi_confirmation_long'] = (rsi_np > 50) & (rsi_np < 70) 
    conditions['rsi_confirmation_short'] = rsi_np < 50 
    conditions['mfi_confirmation_long'] = mfi_np > 50
    conditions['mfi_confirmation_short'] = mfi_np < 50 
    # Volume
    conditions['volume_ok'] = volume_np > (volume_ma20_np * 1.25)
    # MA Alignment for scoring
    conditions['ma_buy'] = fast_ma_np > slow_ma_np
    conditions['ma_sell'] = fast_ma_np < slow_ma_np
    # Bollinger Bands for scoring
    conditions['bb_buy'] = close_np < lower_band_np
    conditions['bb_sell'] = close_np > upper_band_np
    
    # Fill NaNs in boolean conditions (should ideally be handled in prepare_data)
    for key in conditions:
        if pd.api.types.is_bool_dtype(conditions[key]): # Check if already boolean
            conditions[key] = np.nan_to_num(conditions[key].astype(float), nan=0.0).astype(bool)
        elif pd.isna(conditions[key]).any():
             conditions[key] = np.nan_to_num(conditions[key].astype(float), nan=0.0).astype(bool)


    # 3. Calculate Weighted Scores
    buy_trend_score_np = _calculate_trend_score(conditions, TREND_WEIGHTS, direction='long')
    buy_confirmation_score_np = _calculate_confirmation_score(conditions, CONFIRMATION_WEIGHTS, rsi_np, mfi_np, direction='long')
    buy_score_np = (buy_trend_score_np * trend_multiplier_np) + buy_confirmation_score_np

    sell_trend_score_np = _calculate_trend_score(conditions, TREND_WEIGHTS, direction='short')
    sell_confirmation_score_np = _calculate_confirmation_score(conditions, CONFIRMATION_WEIGHTS, rsi_np, mfi_np, direction='short')
    sell_score_np = (sell_trend_score_np * trend_multiplier_np) + sell_confirmation_score_np
    
    signals_df['buy_score'] = buy_score_np
    signals_df['sell_score'] = sell_score_np

    # 4. Entry Triggers (Primary Filters + Min Confirmations)
    # Long entry confirmations
    long_conf_count = (conditions['strong_cloud_support'].astype(int) + 
                       conditions['kumo_breakout'].astype(int) +
                       conditions['rsi_confirmation_long'].astype(int) +
                       conditions['mfi_confirmation_long'].astype(int) +
                       conditions['volume_ok'].astype(int) 
                      )
    entry_trigger_long = (conditions['primary_trend_long'] &
                          conditions['trend_strength_ok'] &
                          (long_conf_count >= 2) 
                         )

    # Short entry confirmations
    short_conf_count = (conditions['below_cloud'].astype(int) + 
                        conditions['kumo_breakdown'].astype(int) +
                        conditions['rsi_confirmation_short'].astype(int) +
                        conditions['mfi_confirmation_short'].astype(int) +
                        conditions['volume_ok'].astype(int) 
                       )
    entry_trigger_short = (conditions['primary_trend_short'] &
                           conditions['trend_strength_ok'] &
                           (short_conf_count >= 2) 
                          )

    # 5. Generate Final Signals (Entry Trigger + Score Threshold)
    buy_signal_np = entry_trigger_long & (buy_score_np >= actual_min_buy_np)
    sell_signal_np = entry_trigger_short & (sell_score_np >= actual_min_sell_np)
    
    signals_df['buy_signal'] = buy_signal_np
    signals_df['sell_signal'] = sell_signal_np

    # 6. Signal Validation
    conflicting_mask = np.logical_and(signals_df['buy_signal'].values, signals_df['sell_signal'].values)
    signals_df.loc[conflicting_mask, 'buy_signal'] = False
    signals_df.loc[conflicting_mask, 'sell_signal'] = False
    
    last_signal_np = signals_df['buy_signal'].astype(int).values - signals_df['sell_signal'].astype(int).values
    signal_changed_np = np.diff(last_signal_np, prepend=0) != 0
    signals_df['signal_changed'] = signal_changed_np

    
    signals_df['buy_signal'] = (signals_df['buy_signal'] ) # & signals_df['buy_signal'].shift(1))
    signals_df['sell_signal'] = (signals_df['sell_signal'] ) # & signals_df['sell_signal'].shift(1))

    return signals_df

# --------------------------------------------------------------------------------------------------------------------------

class TradeManager:
    def __init__(self, initial_capital, max_positions, direction='Long'):
        self.direction = direction
        self.portfolio_value = initial_capital
        self.max_positions = max_positions
        self.position_count = 0
        self.trade_log = []
        self.wins = []
        self.losses = []
        self.lengths = []
        self.common_trade_columns_dtypes = {
            'entry_date': 'datetime64[ns]', 'multiplier': 'int64',
            'entry_price': 'float64', 'stop_loss': 'float64',
            'take_profit': 'float64', 'position_size': 'float64',
            'share_amount': 'int64', 'highest_close_since_entry': 'float64',
            'lowest_close_since_entry': 'float64'
        }
        self.active_trades = pd.DataFrame(columns=self.common_trade_columns_dtypes.keys())
        self.active_trades = self.active_trades.astype(self.common_trade_columns_dtypes)

    # ----------------------------------------------------------------------------------------------------------

    def _calculate_entry_parameters(self, entry_params_dict):
        """
        Calculates entry parameters for a new trade.
        Logic derived from the original trade_entry function, incorporating optimization ideas.
        """
        price = entry_params_dict['price']
        atr = entry_params_dict['atr']
        portfolio_value = entry_params_dict['portfolio_value']
        risk_pct = entry_params_dict['risk']
        position_size_pct = entry_params_dict['position_size']
        leverage = entry_params_dict['leverage']
        adx = entry_params_dict.get('adx')

        position_size_multiplier = 1.5 if (adx is not None and adx > 30) else 1.0
        scaled_position_size = position_size_pct * position_size_multiplier
        
        atr_multiplier = 2.0 if (adx is not None and adx > 30) else 4.0
        
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
        if entry_params_dict.get('reward') is not None:
            reward_multiplier = 3.0 if (adx is not None and adx > 30) else 2.0
            take_profit_distance = valid_risk_per_share * reward_multiplier # Use valid_risk_per_share
            if self.direction == 'Long':
                take_profit = price + take_profit_distance
            else:
                take_profit = price - take_profit_distance
        
        actual_notional = share_amount * price
        exposure_ratio = actual_notional / portfolio_value if portfolio_value > 0 else 0.0

        return price, stop_loss_price, take_profit, exposure_ratio, share_amount

    # ----------------------------------------------------------------------------------------------------------

    def process_entry(self, current_date, entry_params_dict):
        if self.position_count >= self.max_positions or entry_params_dict['portfolio_value'] <= 0:
            return False
        trade_data = self._calculate_entry_parameters(entry_params_dict)
        if not trade_data: return False
        entry_price, stop_loss, take_profit, trade_pos_size_ratio, share_amount = trade_data
        
        new_trade_dict = {
            'entry_date': current_date,
            'multiplier': 1 if self.direction == 'Long' else -1,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': trade_pos_size_ratio,
            'share_amount': share_amount,
            'highest_close_since_entry': entry_price if self.direction == 'Long' else np.nan,
            'lowest_close_since_entry': entry_price if self.direction == 'Short' else np.nan
        }
        new_trade_df = pd.DataFrame([new_trade_dict]).astype(self.common_trade_columns_dtypes)
        
        self.active_trades = pd.concat([self.active_trades, new_trade_df], ignore_index=True)
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
        if self.active_trades.empty: 
            return
        
        atr_multiplier_val = 2.0 if adx_value < 25 else 4.0
        
        if self.direction == 'Long':
            self.active_trades.loc[:, 'highest_close_since_entry'] = np.maximum(
                self.active_trades['highest_close_since_entry'].values, current_price
            )
            new_stops = self.active_trades['highest_close_since_entry'].values - atr_multiplier_val * current_atr
            self.active_trades.loc[:, 'stop_loss'] = np.maximum(
                new_stops, self.active_trades['stop_loss'].values
            )
        else: # Short
            self.active_trades.loc[:, 'lowest_close_since_entry'] = np.minimum(
                self.active_trades['lowest_close_since_entry'].values, current_price
            )
            new_stops = self.active_trades['lowest_close_since_entry'].values + atr_multiplier_val * current_atr
            self.active_trades.loc[:, 'stop_loss'] = np.minimum(
                new_stops, self.active_trades['stop_loss'].values
            )

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
        # ... (print statements for best_result parameters) ...
        best_tech_params = best_result.get('tech_params', {})
        # ... (print statements for best_tech_params) ...
        
        df_with_indicators_viz = prepare_data(
            base_df.copy(), 
            fast=best_tech_params.get('FAST', FAST),
            slow=best_tech_params.get('SLOW', SLOW),
            rsi_oversold=best_tech_params.get('RSI_OVERSOLD', RSI_OVERSOLD),
            rsi_overbought=best_tech_params.get('RSI_OVERBOUGHT', RSI_OVERBOUGHT),
            devs=best_tech_params.get('DEVS', DEVS),
            weekly_ma_period_val=best_tech_params.get('WEEKLY_MA_PERIOD', WEEKLY_MA_PERIOD) # Ensure this is in tech_params if optimized
        )
        
        final_run_result = test( 
            df_with_indicators_viz,
            ticker, 
            long_risk=best_result['long_risk'],
            long_reward=best_result['long_reward'],
            short_risk=best_result['short_risk'],
            short_reward=best_result['short_reward'],
            position_size=best_result['position_size'],
            use_trailing_stops=best_result.get('use_trailing_stops', USE_TRAILING_STOPS_DEFAULT), # Assuming it might be optimized
            # Pass the actual periods used for prepare_data
            fast_period=best_tech_params.get('FAST', FAST),
            slow_period=best_tech_params.get('SLOW', SLOW),
            weekly_ma_p=best_tech_params.get('WEEKLY_MA_PERIOD', WEEKLY_MA_PERIOD),
            # Pass signal parameters from best_tech_params
            adx_thresh=best_tech_params.get('ADX_THRESHOLD', ADX_THRESHOLD_DEFAULT),
            min_buy_s=best_tech_params.get('MIN_BUY_SCORE', MIN_BUY_SCORE_DEFAULT),
            min_sell_s=best_tech_params.get('MIN_SELL_SCORE', MIN_SELL_SCORE_DEFAULT),
            req_cloud=best_tech_params.get('REQUIRE_CLOUD', REQUIRE_CLOUD_DEFAULT)
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
        
        current_fast_period = tech_params.get('FAST', FAST)
        current_slow_period = tech_params.get('SLOW', SLOW)
        current_weekly_ma_period = tech_params.get('WEEKLY_MA_PERIOD', WEEKLY_MA_PERIOD) # Assuming WEEKLY_MA_PERIOD might be in tech_params

        df_prepared = prepare_data(
            base_df_arg, 
            fast=current_fast_period,
            slow=current_slow_period,
            rsi_oversold=tech_params.get('RSI_OVERSOLD', RSI_OVERSOLD),
            rsi_overbought=tech_params.get('RSI_OVERBOUGHT', RSI_OVERBOUGHT),
            devs=tech_params.get('DEVS', DEVS),
            weekly_ma_period_val=current_weekly_ma_period
        )
        
        # Construct column names based on the parameters used for prepare_data
        fast_ma_col = f"{current_fast_period}_ma"
        slow_ma_col = f"{current_slow_period}_ma"
        weekly_ma_col = f"Weekly_MA{current_weekly_ma_period}"

        trade_log, trade_stats, portfolio_equity, trade_returns = momentum(
            df_prepared,
            long_risk=long_risk,
            long_reward=long_reward,
            short_risk=short_risk,
            short_reward=short_reward,
            position_size=position_size,
            # Pass signal generation parameters from tech_params
            adx_threshold_sig=tech_params.get('ADX_THRESHOLD', ADX_THRESHOLD_DEFAULT),
            min_buy_score_sig=tech_params.get('MIN_BUY_SCORE', MIN_BUY_SCORE_DEFAULT),
            min_sell_score_sig=tech_params.get('MIN_SELL_SCORE', MIN_SELL_SCORE_DEFAULT),
            require_cloud_sig=tech_params.get('REQUIRE_CLOUD', REQUIRE_CLOUD_DEFAULT),
            # Pass the constructed column names
            fast_ma_col_name=fast_ma_col,
            slow_ma_col_name=slow_ma_col,
            weekly_ma_col_name=weekly_ma_col
            # MAX_OPEN_POSITIONS, LEVERAGE, USE_TRAILING_STOPS_DEFAULT are likely using defaults in momentum
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

def test(df_input, # Renamed from df to df_input to avoid confusion with internal df
         TICKER_STR, # Renamed from TICKER to avoid confusion with global
         long_risk=DEFAULT_LONG_RISK, long_reward=DEFAULT_LONG_REWARD,
         short_risk=DEFAULT_SHORT_RISK, short_reward=DEFAULT_SHORT_REWARD,
         position_size=DEFAULT_POSITION_SIZE, 
         use_trailing_stops=USE_TRAILING_STOPS_DEFAULT,
         # Add parameters for prepare_data if they are not fixed to global defaults
         fast_period=FAST, 
         slow_period=SLOW, 
         weekly_ma_p=WEEKLY_MA_PERIOD,
         # Add signal generation parameters expected by momentum
         adx_thresh=ADX_THRESHOLD_DEFAULT,
         min_buy_s=MIN_BUY_SCORE_DEFAULT,
         min_sell_s=MIN_SELL_SCORE_DEFAULT,
         req_cloud=REQUIRE_CLOUD_DEFAULT):
    
    df = df_input.copy() # Work on a copy

    current_fast_ma_col = f"{fast_period}_ma"
    current_slow_ma_col = f"{slow_period}_ma"
    current_weekly_ma_col = f"Weekly_MA{weekly_ma_p}"
    
    trade_log, trade_stats, portfolio_equity, trade_returns = momentum(
        df, 
        long_risk=long_risk, long_reward=long_reward,
        short_risk=short_risk, short_reward=short_reward,
        position_size=position_size, 
        max_positions=MAX_OPEN_POSITIONS, 
        risk_free_rate=0.04, # This is a default in momentum, can be passed if needed
        use_trailing_stops=use_trailing_stops,
        adx_threshold_sig=adx_thresh,
        min_buy_score_sig=min_buy_s,
        min_sell_score_sig=min_sell_s,
        require_cloud_sig=req_cloud,
        fast_ma_col_name=current_fast_ma_col,
        slow_ma_col_name=current_slow_ma_col,
        weekly_ma_col_name=current_weekly_ma_col
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
        'long_risk': long_risk,
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