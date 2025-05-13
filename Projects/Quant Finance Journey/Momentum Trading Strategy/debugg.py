import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import yfinance as yf
from tabulate import tabulate  
import pandas_ta as ta 
from datetime import datetime, timedelta
import multiprocessing as mp 
import itertools
import matplotlib.pyplot as plt  # Fix matplotlib import
import seaborn as sns
from collections import deque
from numba import jit
from tqdm import tqdm
import optuna
from functools import partial  # Add missing import for partial
import cupy as cp

# First, add new optimization direction constants
OPTIMIZATION_DIRECTIONS = {
    'sharpe': 'maximize',
    'profit_factor': 'maximize',
    'expectancy': 'maximize',
    'max_drawdown': 'minimize',
    'combined': 'maximize'  # For weighted combinations
}

# Add weight configuration for multi-objective optimization
OBJECTIVE_WEIGHTS = {
    'sharpe': 0.4,        # 40% weight for Sharpe ratio
    'profit_factor': 0.3, # 30% weight for Profit Factor
    'expectancy': 0.3     # 30% weight for Expectancy
}

# Trade Management
ADX_THRESHOLD_DEFAULT = 15
MIN_BUY_SCORE_DEFAULT = 2.0
MIN_SELL_SCORE_DEFAULT = 5.0
REQUIRE_CLOUD_DEFAULT = True
USE_TRAILING_STOPS_DEFAULT = False

# Risk & Reward
DEFAULT_LONG_RISK = 0.08  
DEFAULT_LONG_REWARD = 0.13 
DEFAULT_SHORT_RISK = 0.02  
DEFAULT_SHORT_REWARD = 0.03  
DEFAULT_POSITION_SIZE = 0.50
MAX_OPEN_POSITIONS = 39 

SIGNAL_WEIGHTS = {
    # Trend components
    'primary_trend': 2.5,     
    'trend_strength': 2.5,    
    'weekly_trend': 1.5,      
    
    # Confirmation components
    'rsi': 1.2,              
    'volume': 1.5,           
    'bollinger': 0.3         
}

# Base parameters
TICKER = ['PEP']
INITIAL_CAPITAL = 100000.0
LEVERAGE = 1.0 

# Moving average strategy parameters
FAST = 20
SLOW = 50
WEEKLY_MA_PERIOD = 50

# RSI strategy parameters
RSI_LENGTH = 14
RSI_OVERBOUGHT = 75
RSI_OVERSOLD = 35

# Bollinger Bands strategy parameters
BB_LEN = 20
ST_DEV = 2

# -------------------------------------------------------------------------------------------------------------
def get_data(ticker):
    data_start_year = 2013
    data = yf.download(ticker, start=f"{data_start_year}-01-01", auto_adjust=True)

    if data.empty:
        print(f"No data available for {ticker}")
        return None, None, None

    # Fix index access - oldest is first, latest is last
    data_oldest = data.index[0]
    data_latest = data.index[-1]
    
    # Calculate data range in years
    data_year_range = (data_latest - data_oldest).days / 365.25
    
    # Determine periods based on data range
    if data_year_range > 8:
        start_date = pd.Timestamp(f"{data_start_year}-01-01")
        in_sample_period = start_date + pd.DateOffset(years=5)
        out_of_sample_period = in_sample_period + pd.DateOffset(years=1)
    else:
        in_sample_period = data_oldest + pd.DateOffset(years=3)
        out_of_sample_period = in_sample_period + pd.DateOffset(years=1)
    
    # Split data into IS and OOS DataFrames
    in_sample_df = data[data_oldest:in_sample_period].copy()
    out_of_sample_df = data[out_of_sample_period:data_latest].copy()

    if in_sample_df.empty or out_of_sample_df.empty:
            raise ValueError(f"No data downloaded for {ticker}")
    
    return in_sample_df, out_of_sample_df
# -------------------------------------------------------------------------------------------------------------
def prepare_data(df_IS):

    # 1. Initial Setup & Memory Optimization
    df = df_IS.copy()
    if df.empty:
        print("Warning: Empty dataframe provided to prepare_data.")
        return df
    
    # Convert known OHLCV columns and any float64 to float32 for memory efficiency
    ohlcv_cols_for_dtype_check = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif col in ohlcv_cols_for_dtype_check:
            if not pd.api.types.is_numeric_dtype(df[col]):
                # If OHLCV cols are present but not numeric (e.g. object), try to convert
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # Ensure it's float32 if it's numeric but not already float32
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype != 'float32':
                 df[col] = df[col].astype('float32')
    
    if len(df) > 100000:  # Only for large datasets
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = df[col].astype('category')
            except TypeError: # Handle cases where conversion to category might fail
                print(f"Warning: Could not convert column '{col}' to category type.")
    
    # 2. Multi-index Handling Simplification
    if isinstance(df.columns, pd.MultiIndex):
        # Assuming the primary column names are in the first level (level 0)
        df.columns = df.columns.droplevel(1)
        # Ensure no duplicate column names after droplevel, take the first occurrence
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # 3. Logic to 
    core_ohlcv_names = ['Open', 'High', 'Low', 'Close', 'Volume']
    # Threshold for dropping rows if NaNs are below this percentage for a given column
    max_nan_percentage_to_drop_rows = 0.05
    for col_name in core_ohlcv_names:
        if col_name not in df.columns:
                print(f"WARNING: Core column '{col_name}' is missing. Calculations requiring it may be affected or use defaults if available in the specific indicator logic.")
        else:
            if df[col_name].isnull().all(): # If the entire column is NaNs
                    print(f"WARNING: Core column '{col_name}' consists entirely of NaN values. This will likely cause issues for indicators relying on it.")
            else:
                num_nans = df[col_name].isnull().sum()
                if num_nans > 0:
                    nan_percentage = num_nans / len(df)
                    if nan_percentage < max_nan_percentage_to_drop_rows:
                        print(f"INFO: Column '{col_name}' has {num_nans} NaN values ({nan_percentage*100:.2f}% of current data). Dropping rows with these NaNs.")
                        df.dropna(subset=[col_name], inplace=True)
                        if df.empty:
                            print(f"WARNING: DataFrame became empty after dropping NaNs for column '{col_name}'.")
                            return df # Return the now empty DataFrame
                    else:
                        print(f"WARNING: Column '{col_name}' has {nan_percentage*100:.2f}% NaNs, which is high. Attempting forward-fill then backward-fill.")
                        df[col_name].ffill(inplace=True)
                        df[col_name].bfill(inplace=True) # Fill any NaNs remaining at the beginning
                        if df[col_name].isnull().any(): # Should only happen if column was all NaNs initially (caught above)
                            print(f"ERROR: Column '{col_name}' still contains NaNs after ffill/bfill. This column may be unusable for some indicators.")


    # Final check: After all initial cleaning, is 'Close' usable?
    if 'Close' not in df.columns or df.empty or df['Close'].isnull().all():
        print("CRITICAL ERROR: 'Close' column is missing or unusable after initial data preparation steps.")
        return df
    
    # 4. Calculate Core Indicators
    try:
        # Main Indicators
        # Vectorized Moving Averages
        df[f'{FAST}_ma'] = df['Close'].rolling(window=FAST, min_periods=1).mean()
        df[f'{SLOW}_ma'] = df['Close'].rolling(window=SLOW, min_periods=1).mean()
        df['Volume_MA20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        
        # Weekly Moving Average
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df_weekly = df['Close'].resample('W').last()
        df[f'Weekly_MA{WEEKLY_MA_PERIOD}'] = (df_weekly.rolling(window=WEEKLY_MA_PERIOD, min_periods=1)
            .mean().reindex(df.index, method='ffill'))
        
        # Relative Strength Index - Measures pct change in price
        df['RSI'] = ta.rsi(df['Close'], length=RSI_LENGTH).fillna(50.0)
        
        # Bollinger Bands - Measures volatility
        bb = ta.bbands(df['Close'], length=BB_LEN, std=ST_DEV)
        if bb is not None and not bb.empty:
            df['Upper_Band'] = bb[f'BBU_{BB_LEN}_{float(ST_DEV)}'].fillna(df['Close'] * 1.02)
            df['Lower_Band'] = bb[f'BBL_{BB_LEN}_{float(ST_DEV)}'].fillna(df['Close'] * 0.98)
        else:
            df['Upper_Band'] = df['Close'] * 1.02
            df['Lower_Band'] = df['Close'] * 0.98

        # Secondary Indicators
        # Average True Range - Measures volatility
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14).fillna(df['Close'] * 0.02)

        # Close price 26 periods ago
        df['Close_26_ago'] = df['Close'].shift(26).ffill().fillna(df['Close'])

        # Average Directional Index - Measures trend strength
        adx_result = ta.adx(df['High'], df['Low'], df['Close'])
        if isinstance(adx_result, pd.DataFrame) and 'ADX_14' in adx_result.columns:
            df['ADX'] = adx_result['ADX_14'].fillna(25.0)
        else:
            df['ADX'] = pd.Series(25.0, index=df.index)

        # Volume and Weekly Trend Confirmations
        df['volume_confirmed'] = df['Volume'] > df['Volume_MA20']
        df['weekly_uptrend'] = (df['Close'] > df[f'Weekly_MA{WEEKLY_MA_PERIOD}']) & \
                       (df[f'Weekly_MA{WEEKLY_MA_PERIOD}'].shift(1).ffill() < \
                        df[f'Weekly_MA{WEEKLY_MA_PERIOD}'])
        
    except Exception as e:
        print(f"Error calculating indicators: {e}. Applying defaults.")

    # 5. Validate and Fill Missing (New Default Value Handling)
    default_values = {
        f'{FAST}_ma': df['Close'],
        f'{SLOW}_ma': df['Close'],
        'RSI': 50.0,
        'Upper_Band': df['Close'] * 1.02,
        'Lower_Band': df['Close'] * 0.98,
        'ATR': df['Close'] * 0.02,
        'ADX': 25.0,
        'Volume_MA20': df['Volume'] if 'Volume' in df else 10000.0, # Default if volume itself is missing
        f'Weekly_MA{WEEKLY_MA_PERIOD}': df['Close'],
        'Close_26_ago': df['Close']
    }
    # Fill missing values with defaults
    for col_name, default_series_or_value in default_values.items():
        if col_name not in df.columns or df[col_name].isnull().all(): # If col is missing or all NaN
            df[col_name] = default_series_or_value
        else: # If column exists but has some NaNs, fill them
            df[col_name] = df[col_name].fillna(default_series_or_value)

    # 6. Final Cleanup
    df = df.dropna() # Drop rows where essential data might still be missing after fill attempts
    if df.empty and not df_IS.empty:
        print("Warning: DataFrame became empty after processing.")
    
    return df
# -------------------------------------------------------------------------------------------------------------
def momentum(df_with_indicators, long_risk=None, long_reward=None, short_risk=None, short_reward=None,
    position_size=None, risk_free_rate=0.04, leverage_ratio=None, max_positions=None, use_trailing_stops=None,
    adx_threshold_sig=None, min_buy_score_sig=None, min_sell_score_sig=None,
    signal_weights_sig=None):
    
    if df_with_indicators.empty:
        print("Warning: Empty dataframe (df_with_indicators) provided to momentum.")
        return [], create_empty_stats(), pd.Series(dtype='float64'), pd.Series(dtype='float64')

    # Ensure parameters are set to defaults if None
    long_risk = long_risk if long_risk is not None else DEFAULT_LONG_RISK
    long_reward = long_reward if long_reward is not None else DEFAULT_LONG_REWARD
    short_risk = short_risk if short_risk is not None else DEFAULT_SHORT_RISK
    short_reward = short_reward if short_reward is not None else DEFAULT_SHORT_REWARD
    position_size = position_size if position_size is not None else DEFAULT_POSITION_SIZE
    leverage_ratio = leverage_ratio if leverage_ratio is not None else LEVERAGE
    max_positions = max_positions if max_positions is not None else MAX_OPEN_POSITIONS
    use_trailing_stops = use_trailing_stops if use_trailing_stops is not None else USE_TRAILING_STOPS_DEFAULT
    adx_threshold_sig = adx_threshold_sig if adx_threshold_sig is not None else ADX_THRESHOLD_DEFAULT
    min_buy_score_sig = min_buy_score_sig if min_buy_score_sig is not None else MIN_BUY_SCORE_DEFAULT
    min_sell_score_sig = min_sell_score_sig if min_sell_score_sig is not None else MIN_SELL_SCORE_DEFAULT
    final_signal_weights = signal_weights_sig if signal_weights_sig is not None else SIGNAL_WEIGHTS


    # 1. Generate signals using the indicator-laden DataFrame
    signals_df = signals(df_with_indicators, adx_threshold=adx_threshold_sig, min_buy_score=min_buy_score_sig, 
                         min_sell_score=min_sell_score_sig, weights=final_signal_weights)

    # 2. Initialize Trade Managers and Tracking
    # Ensure INITIAL_CAPITAL is accessible or passed if not a global
    trade_manager = TradeManager(INITIAL_CAPITAL, max_positions)

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
        current_date = df_with_indicators.index[i]
        transaction_price = df_with_indicators['Open'].iloc[i]
        
        # Critical data checks for the current row
        # Data for decision making (ATR, ADX, confirmations) is from the PREVIOUS day's close
        previous_day_data_row = df_with_indicators.iloc[i-1]
        previous_day_atr = previous_day_data_row['ATR']
        previous_day_adx = previous_day_data_row['ADX']

        buy_signal = signals_df['buy_signal'].iloc[i-1]
        sell_signal = signals_df['sell_signal'].iloc[i-1]
        
        # --- Exit Conditions (Priority Order) ---
        if trade_manager.position_count > 0:
            # 1. Check trailing stop first (highest priority)
            trail_stop_hit = trade_manager.trailing_stops(transaction_price, previous_day_atr, previous_day_adx)
            if trail_stop_hit:
                trade_manager.process_exits(current_date, transaction_price)
                continue

            # 2. Check trend validity  
            trend_valid = previous_day_adx > adx_threshold_sig

            # 3. Hybrid exit rules
            if trade_manager.direction == 'Long':
                # Partial exit on sell signal (if trend is weak)
                if sell_signal and not trail_stop_hit:
                    if trend_valid:
                        trade_manager.process_exits(current_date, transaction_price, Trim = True)  
                    else:
                        trade_manager.process_exits(current_date, transaction_price)
            elif trade_manager.direction == 'Short':
                # Full exit on buy signal (shorts need tighter control)
                if buy_signal and not trail_stop_hit:
                    trade_manager.process_exits(current_date, transaction_price)

        # --- Entry Conditions ---
        entry_params_base = {'price': transaction_price,
                             'atr': previous_day_atr, 'adx': previous_day_adx,
                             'position_size': position_size, 'leverage': leverage_ratio,}

        if buy_signal and trade_manager.position_count < max_positions:
            entry_params = {**entry_params_base,
                            'portfolio_value': trade_manager.portfolio_value,
                            'risk': long_risk, 'reward': long_reward}
            trade_manager.process_entry(current_date, entry_params, direction='Long')

        if sell_signal and trade_manager.position_count < max_positions:
            entry_params = {**entry_params_base,
                            'portfolio_value': trade_manager.portfolio_value,
                            'risk': short_risk, 'reward': short_reward}
            trade_manager.process_entry(current_date, entry_params, direction='Short')

        # --- Portfolio Tracking ---
        total_value = trade_manager.portfolio_value + trade_manager.calculate_unrealized_pnl(transaction_price)
        equity_curve.iloc[i] = total_value
        returns_series.iloc[i] = (total_value / equity_curve.iloc[i-1] - 1) if equity_curve.iloc[i-1] != 0 else 0.0

    # Calculate statistics using the single trade manager's data
    stats = trade_statistics(equity_curve, trade_manager.trade_log, trade_manager.wins, 
                             trade_manager.losses, risk_free_rate)

    return trade_manager.trade_log, stats, equity_curve, returns_series
# --------------------------------------------------------------------------------------------------------------------------
def dynamic_scores(conditions, rsi_np_arr, weights_dict, direction='long', 
                  weekly_trend=None, atr_ratio_np=None):
    
    RSI_HIGH_VOL_THRESHOLD = 0.015  # ATR ratio threshold for high volatility
    RSI_HIGH_VOL_BOUNDS = {'long': (40, 55), 'short': (45, 60)}  # RSI bounds for high vol
    RSI_LOW_VOL_BOUNDS = {'long': (45, 55), 'short': (45, 55)}
    
    # Initialize base score
    if rsi_np_arr is not None and hasattr(rsi_np_arr, 'shape'):
        base_score = np.zeros_like(rsi_np_arr, dtype=float)
    else:
        base_score = np.zeros_like(conditions.get('primary_trend_long', np.array([])), dtype=float)

    # 1. Trend Components
    trend_condition = 'primary_trend_long' if direction == 'long' else 'primary_trend_short'
    base_score += conditions.get(trend_condition, False).astype(float) * weights_dict.get('primary_trend', 0)
    
    # 2. Trend Strength (ADX)
    base_score += conditions.get('trend_strength_ok', False).astype(float) * weights_dict.get('trend_strength', 0)
    
    # 3. Weekly Trend
    if weekly_trend is not None:
        weekly_score = weekly_trend if direction == 'long' else ~weekly_trend
        base_score += weekly_score.astype(float) * weights_dict.get('weekly_trend', 0)

    # 4. Volatility-adjusted RSI
    if atr_ratio_np is not None:
        high_vol_mask = atr_ratio_np > RSI_HIGH_VOL_THRESHOLD
        bounds = RSI_HIGH_VOL_BOUNDS if direction == 'long' else RSI_LOW_VOL_BOUNDS
        
        if direction == 'long':
            rsi_strength = np.where(
                high_vol_mask,
                np.clip((rsi_np_arr - bounds['long'][0]) / 15, 0, 1),
                np.clip((rsi_np_arr - bounds['long'][1]) / 10, 0, 1)
            )
        else:
            rsi_strength = np.where(
                high_vol_mask,
                np.clip((bounds['short'][1] - rsi_np_arr) / 15, 0, 1),
                np.clip((bounds['short'][0] - rsi_np_arr) / 10, 0, 1)
            )
    else:
        # Default RSI calculation without volatility adjustment
        mid_point = 50
        rsi_strength = np.clip((rsi_np_arr - mid_point) / 20, 0, 1) if direction == 'long' \
                      else np.clip((mid_point - rsi_np_arr) / 20, 0, 1)

    # 5. Add confirmation components
    base_score += rsi_strength * weights_dict.get('rsi', 0)
    base_score += conditions.get('volume_ok', False).astype(float) * weights_dict.get('volume', 0)
    
    # 6. Add Bollinger Band component
    bb_condition = 'bb_buy' if direction == 'long' else 'bb_sell'
    base_score += conditions.get(bb_condition, False).astype(float) * weights_dict.get('bollinger', 0)

    return base_score
# --------------------------------------------------------------------------------------------------------------------------
def signals(df, adx_threshold, min_buy_score, min_sell_score, weights=None):
    """
    Generates trading signals for the entire DataFrame using vectorized operations, NumPy,
    primary trend filters, secondary confirmations, and a weighted scoring approach.
    """
    fast_ma_col = f"{FAST}_ma"
    slow_ma_col = f"{SLOW}_ma"
    signals_df = pd.DataFrame(index=df.index)

    # Use passed weights if available, otherwise default to global
    current_weights = weights if weights is not None else SIGNAL_WEIGHTS

    required_indicator_cols = [fast_ma_col, slow_ma_col, 'RSI', 'Close',
                                'Lower_Band', 'Upper_Band', 'ATR', 'ADX', 'Volume', 'Volume_MA20', 'Open', 
                                'High', 'Low','volume_confirmed', 'weekly_uptrend']
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
    rsi_np = df['RSI'].values
    lower_band_np = df['Lower_Band'].values
    upper_band_np = df['Upper_Band'].values
    
    atr_ratio_np = np.full_like(close_np, 0.02, dtype=float) # Ensure float
    valid_close_mask = close_np != 0
    # Ensure division by zero is handled if close_np can be zero
    safe_close_np = np.where(valid_close_mask, close_np, 1e-9) # Avoid division by zero
    atr_ratio_np[valid_close_mask] = atr_np[valid_close_mask] / safe_close_np[valid_close_mask]
    atr_ratio_np = np.nan_to_num(atr_ratio_np, nan=0.02, posinf=0.02, neginf=0.02)

    volatility_factor = np.log1p(atr_ratio_np / 0.02)
    actual_min_buy_np = min_buy_score * (1 + volatility_factor)
    actual_min_sell_np = min_sell_score * (1 + volatility_factor)

    # 2. Calculate Primary and Secondary Conditions
    conditions = {}
    # Primary Trend Filters
    conditions['primary_trend_long'] = (close_np > fast_ma_np) & (close_np > slow_ma_np) 
    conditions['primary_trend_short'] = (close_np < fast_ma_np) & (close_np < slow_ma_np) 
    conditions['trend_strength_ok'] = adx_np > adx_threshold
    conditions['rsi_uptrend'] = rsi_np < 40
    conditions['rsi_weak'] = rsi_np > 60

    # Momentum (used in dynamic_scores, not directly as conditions here)
    # Volume & Uptrend Confirmation
    conditions['volume_ok'] = df['volume_confirmed'].values
    conditions['weekly_uptrend'] = df['weekly_uptrend'].values
    # MA Alignment for scoring
    conditions['ma_buy'] = fast_ma_np > slow_ma_np
    conditions['ma_sell'] = fast_ma_np < slow_ma_np
    # Bollinger Bands for scoring
    conditions['bb_buy'] = close_np < lower_band_np
    conditions['bb_sell'] = close_np > upper_band_np
    
    # Ensure all conditions are boolean arrays of the same shape
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
    buy_score_np = dynamic_scores(conditions, rsi_np, current_weights, direction='long', weekly_trend=conditions['weekly_uptrend'], atr_ratio_np=atr_ratio_np)
    sell_score_np = dynamic_scores(conditions, rsi_np, current_weights, direction='short', weekly_trend=conditions['weekly_uptrend'], atr_ratio_np=atr_ratio_np)

    signals_df['buy_score'] = buy_score_np
    signals_df['sell_score'] = sell_score_np

    signals_df['buy_signal'] = (
        conditions.get('primary_trend_long', False) &
        conditions.get('rsi_uptrend', False) &
        conditions.get('trend_strength_ok', False) &
        conditions.get('volume_ok', False) | (buy_score_np >= actual_min_buy_np))

    signals_df['sell_signal'] = (
        conditions.get('primary_trend_short', False) &
        conditions.get('rsi_weak', False) &
        conditions.get('trend_strength_ok', False) &
        conditions.get('weekly_uptrend', False)  | (sell_score_np >= actual_min_sell_np))

    # 5. Signal Validation (Conflicting signals and changed status)
    conflicting = signals_df['buy_signal'] & signals_df['sell_signal']
    signals_df.loc[conflicting, ['buy_signal', 'sell_signal']] = False
    
    last_signal_np = signals_df['buy_signal'].astype(int).values - signals_df['sell_signal'].astype(int).values
    signal_changed_np = np.diff(last_signal_np, prepend=0) != 0
    signals_df['signal_changed'] = signal_changed_np

    return signals_df
# --------------------------------------------------------------------------------------------------------------------------
class TradeManager:
    def __init__(self, initial_capital, max_positions):
        self.portfolio_value = initial_capital
        self.max_positions = max_positions
        self.position_count = 0
        self.trade_log = deque()
        self.wins = deque()
        self.losses = deque()
        self.lengths = deque()
        self.direction = None

        # Define column dtypes for better memory usage
        self.dtypes = {
            'entry_date': 'datetime64[ns]',
            'direction': 'category',  # Added direction column
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
    def trailing_stops(self, current_price, current_atr, adx_value):
        if self.active_trades.empty:
            return False
        
        # Convert to float32 for consistency
        current_price_f32 = np.float32(current_price)
        current_atr_f32 = np.float32(current_atr)
        
        # Dynamic ATR multiplier based on market conditions
        atr_multiplier = np.float32(2.5 if adx_value > 30 else 1.5)
        
        if self.direction == 'Long':
            # Update highest prices
            self.active_trades['highest_close_since_entry'] = np.maximum(
                self.active_trades['highest_close_since_entry'],
                current_price_f32
            )
            # Calculate new stops
            new_stops = self.active_trades['highest_close_since_entry'] - (atr_multiplier * current_atr_f32)
            self.active_trades['stop_loss'] = np.maximum(new_stops, self.active_trades['stop_loss'])
            # Check for stops hit
            stops_hit = current_price_f32 <= self.active_trades['stop_loss']
            
        else:  # Short
            # Update lowest prices
            self.active_trades['lowest_close_since_entry'] = np.minimum(
                self.active_trades['lowest_close_since_entry'],
                current_price_f32
            )
            # Calculate new stops
            new_stops = self.active_trades['lowest_close_since_entry'] + (atr_multiplier * current_atr_f32)
            self.active_trades['stop_loss'] = np.minimum(new_stops, self.active_trades['stop_loss'])
            # Check for stops hit
            stops_hit = current_price_f32 >= self.active_trades['stop_loss']

        # If any stops were hit, keep only the unaffected trades
        if stops_hit.any():
            # Process exits for stopped out trades
            stopped_trades = self.active_trades[stops_hit]
            for _, trade in stopped_trades.iterrows():
                self.exit_pnl(trade, pd.Timestamp('now'), current_price_f32,
                            trade['share_amount'], 'Trailing Stop')
                self.position_count -= 1
            
            # Keep only trades that didn't hit stops
            self.active_trades = self.active_trades[~stops_hit].copy()
            return True
        
        return False
    # ----------------------------------------------------------------------------------------------------------
    def process_exits(self, current_date, current_price, Trim=False):

        if self.active_trades.empty:
            return 0.0

        total_pnl = 0.0
        indices_to_remove = []
        
        # Iterate over a copy of the DataFrame to avoid index issues
        for idx, trade in self.active_trades.iterrows():
            current_shares = trade['share_amount']
            
            if current_shares <= 0:
                indices_to_remove.append(idx)
                continue

            exited_this_step = False

            # Process Trim or Full Exit
            if Trim:
                shares_to_exit = int(current_shares * 0.3)
                if shares_to_exit > 0:
                    pnl = self.exit_pnl(trade, current_date, current_price, 
                                    shares_to_exit, 'Partial Signal Exit')
                    total_pnl += pnl
                    self.portfolio_value += pnl
                    
                    # Update remaining shares
                    remaining_shares = current_shares - shares_to_exit
                    if remaining_shares > 0:
                        self.active_trades.loc[idx, 'share_amount'] = remaining_shares
                    else:
                        indices_to_remove.append(idx)
                        self.position_count -= 1
                        exited_this_step = True
            else:  # Full Exit 
                pnl = self.exit_pnl(trade, current_date, current_price,
                                current_shares, 'Full Signal Exit')
                total_pnl += pnl
                self.portfolio_value += pnl
                indices_to_remove.append(idx)
                self.position_count -= 1
                exited_this_step = True
            
            if exited_this_step:
                continue

            # Check Stop Loss
            sl_hit = False
            if self.direction == 'Long' and current_price <= trade['stop_loss']:
                sl_hit = True
            elif self.direction == 'Short' and current_price >= trade['stop_loss']:
                sl_hit = True
            
            if sl_hit:
                pnl = self.exit_pnl(trade, current_date, current_price, 
                                current_shares, 'Stop Loss')
                total_pnl += pnl
                self.portfolio_value += pnl
                indices_to_remove.append(idx)
                self.position_count -= 1
                continue

            # Check Take Profit
            tp_hit = False
            if pd.notna(trade['take_profit']):
                if self.direction == 'Long' and current_price >= trade['take_profit']:
                    tp_hit = True
                elif self.direction == 'Short' and current_price <= trade['take_profit']:
                    tp_hit = True
            
            if tp_hit:
                pnl = self.exit_pnl(trade, current_date, current_price, 
                                current_shares, 'Take Profit')
                total_pnl += pnl
                self.portfolio_value += pnl
                indices_to_remove.append(idx)
                self.position_count -= 1
        
        # Remove closed positions
        if indices_to_remove:
            self.active_trades = self.active_trades.drop(index=indices_to_remove)
            self.active_trades = self.active_trades.reset_index(drop=True)

        return total_pnl
    # ----------------------------------------------------------------------------------------------------------
    def process_entry(self, current_date, entry_params, direction=None):
        
        is_long = direction == 'Long'
        direction_mult = 1 if is_long else -1
        self.direction = direction
        
        # 1. Calculate Initial Stop (ATR + ADX adjusted)
        atr_multiplier = 2.5 if entry_params['adx'] > 25 else 1.5
        min_stop = entry_params['price'] * 0.005  # 0.5% minimum
        stop_distance = max(entry_params['atr'] * atr_multiplier, min_stop)
        initial_stop = entry_params['price'] - (stop_distance * direction_mult)
        
        # 2. Position Sizing Based on Risk
        risk_per_share = abs(entry_params['price'] - initial_stop)
        max_risk_amount = entry_params['portfolio_value'] * entry_params['risk']
        shares_risk = int(max_risk_amount / max(risk_per_share, 1e-9))  # Avoid division by zero
    
        # 3. Leverage-Based Maximum Shares
        max_exposure = entry_params['portfolio_value'] * entry_params['position_size'] * entry_params['leverage']
        shares_leverage = int(max_exposure / max(entry_params['price'], 1e-9))
        
        share_amount = min(shares_risk, shares_leverage)
        if share_amount <= 0:
            return False
        
        # 4. Calculate Take-Profit Using Risk/Reward Ratio
        if 'reward' in entry_params:
            # Reward distance = (risk_per_share) * (reward/risk ratio)
            # Example: If risk=0.01 (1%) and reward=0.03 (3%), reward_distance = risk_per_share * 3
            reward_distance = risk_per_share * (entry_params['reward'] / entry_params['risk'])
            take_profit = entry_params['price'] + (reward_distance * direction_mult)
        else:
            take_profit = None
        
        # 5. Create Trade
        new_trade = pd.DataFrame([{
        'entry_date': current_date,
        'direction': direction,  # Add direction explicitly
        'multiplier': direction_mult,
        'entry_price': entry_params['price'],
        'stop_loss': initial_stop,
        'take_profit': take_profit,
        'position_size': (share_amount * entry_params['price']) / entry_params['portfolio_value'],
        'share_amount': share_amount,
        'highest_close_since_entry': entry_params['price'] if is_long else np.nan,
        'lowest_close_since_entry': entry_params['price'] if not is_long else np.nan
        }])
        
        # Apply dtypes after creating the DataFrame
        for col in new_trade.columns:
            if col in self.dtypes:
                new_trade[col] = new_trade[col].astype(self.dtypes[col])
        
        # Concatenate and maintain dtypes
        if self.active_trades.empty:
            self.active_trades = new_trade
        else:
            self.active_trades = pd.concat([self.active_trades, new_trade], ignore_index=True)
        
        self.position_count += 1
        return True
    # ----------------------------------------------------------------------------------------------------------    
    def exit_pnl(self, trade_series, exit_date, exit_price, shares_to_exit, reason):
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
    def calculate_unrealized_pnl(self, current_price):
        if self.active_trades.empty: return 0.0
        if self.direction == 'Long':
            return ((current_price - self.active_trades['entry_price']) * self.active_trades['share_amount']).sum()
        else:
            return ((self.active_trades['entry_price'] - current_price) * self.active_trades['share_amount']).sum()
    # ----------------------------------------------------------------------------------------------------------
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
def risk_metrics(returns_array, risk_free_daily):
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
    sharpe_ratio, sortino_ratio = risk_metrics(daily_returns, daily_rf_rate)
    
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
def main():
    IS_df, OOS_df = get_data(TICKER[0])
    if IS_df is None or OOS_df is None:
        print("Error: Data retrieval failed.")
        return
    prepared_data = prepare_data(IS_df)

    _, stats, _, _ = momentum(prepared_data, long_risk=DEFAULT_LONG_REWARD,
                                            long_reward=DEFAULT_LONG_REWARD, short_risk=DEFAULT_SHORT_RISK,
                                            short_reward=DEFAULT_SHORT_REWARD, position_size=MAX_OPEN_POSITIONS,
                                            risk_free_rate=0.04, leverage_ratio=LEVERAGE,
                                            max_positions=MAX_OPEN_POSITIONS, use_trailing_stops=None,
                                            adx_threshold_sig=ADX_THRESHOLD_DEFAULT, min_buy_score_sig=MIN_BUY_SCORE_DEFAULT, min_sell_score_sig=MIN_SELL_SCORE_DEFAULT,
                                            signal_weights_sig=SIGNAL_WEIGHTS)
    print(stats)

if __name__ == "__main__":
    main()
