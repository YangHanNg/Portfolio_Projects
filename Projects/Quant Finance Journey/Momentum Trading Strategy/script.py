import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import yfinance as yf
from tabulate import tabulate  
import pandas_ta as ta 
from datetime import datetime, timedelta
import multiprocessing as mp 
from collections import deque
from numba import jit
from tqdm import tqdm
import optuna
from functools import partial  # Add missing import for partial
import cupy as cp
import traceback
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from joblib import Parallel, delayed

# First, add new optimization direction constants
OPTIMIZATION_DIRECTIONS = {
    'sharpe': 'maximize',
    'profit_factor': 'maximize',
    'returns': 'maximize',
    'max_drawdown': 'minimize',
}

# Add weight configuration for multi-objective optimization
OBJECTIVE_WEIGHTS = {
    'sharpe': 0.25,        # 40% weight for Sharpe ratio
    'profit_factor': 0.25, # 30% weight for Profit Factor
    'returns': 0.25,     # 30% weight for Returns
    'max_drawdown': 0.25   # 10% weight for Max Drawdown
}

# Trade Management
ADX_THRESHOLD_DEFAULT = 15
MIN_BUY_SCORE_DEFAULT = 2.0
MIN_SELL_SCORE_DEFAULT = 5.0

# Risk & Reward
DEFAULT_LONG_RISK = 0.08  
DEFAULT_LONG_REWARD = 0.13 
DEFAULT_SHORT_RISK = 0.01  
DEFAULT_SHORT_REWARD = 0.03  
DEFAULT_POSITION_SIZE = 0.50
MAX_OPEN_POSITIONS = 39 

SIGNAL_WEIGHTS = {
    'primary_trend': 2.5,     
    'trend_strength': 2.5,    
    'weekly_trend': 1.5,      
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
def momentum(df_with_indicators, long_risk=DEFAULT_LONG_RISK, long_reward=DEFAULT_LONG_REWARD, short_risk=DEFAULT_SHORT_RISK, short_reward=DEFAULT_SHORT_REWARD,
    position_size=DEFAULT_POSITION_SIZE, risk_free_rate=0.04, leverage_ratio=LEVERAGE, max_positions=MAX_OPEN_POSITIONS,
    adx_threshold_sig=ADX_THRESHOLD_DEFAULT, min_buy_score_sig=MIN_BUY_SCORE_DEFAULT, min_sell_score_sig=MIN_SELL_SCORE_DEFAULT,
    signal_weights_sig=SIGNAL_WEIGHTS):

    if df_with_indicators.empty:
        print("Warning: Empty dataframe (df_with_indicators) provided to momentum.")
        return [], create_empty_stats(), pd.Series(dtype='float64'), pd.Series(dtype='float64')

    # 1. Generate signals using the indicator-laden DataFrame
    signals_df = signals(df_with_indicators, adx_threshold=adx_threshold_sig, min_buy_score=min_buy_score_sig,
                         min_sell_score=min_sell_score_sig, weights=signal_weights_sig)

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
            any_trailing_stop_hit = trade_manager.trailing_stops(transaction_price, current_date, previous_day_atr, previous_day_adx)

            # 2. Check trend validity  
            trend_valid = previous_day_adx > adx_threshold_sig

            # 3. Hybrid exit rules
            if not any_trailing_stop_hit:
                # Process exits for LONG positions based on sell signal
                if sell_signal:
                    if trend_valid:
                        # Partial exit for LONG positions if trend is still valid
                        trade_manager.process_exits(current_date, transaction_price, direction_to_exit='Long', Trim=True)  
                    else:
                        # Full exit for LONG positions if trend is no longer valid
                        trade_manager.process_exits(current_date, transaction_price, direction_to_exit='Long', Trim=False)
                
                # Process exits for SHORT positions based on buy signal
                if buy_signal:
                    # Full exit for SHORT positions (shorts need tighter control)
                    trade_manager.process_exits(current_date, transaction_price, direction_to_exit='Short', Trim=False)

        # --- Entry Conditions ---
        entry_params_base = {'price': transaction_price, 'atr': previous_day_atr, 'adx': previous_day_adx, 'size_limit': position_size}

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
        conditions.get('trend_strength_ok', False) &
        conditions.get('rsi_uptrend', False) | (buy_score_np >= actual_min_buy_np)) # | conditions.get('volume_ok', False)
        

    signals_df['sell_signal'] = (
        conditions.get('primary_trend_short', False) &
        conditions.get('trend_strength_ok', False) &
        conditions.get('rsi_weak', False) | (sell_score_np >= actual_min_sell_np)) # conditions.get('weekly_uptrend', False)

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
        self.allocated_capital = 0
        self.last_exit_was_full = False
        self.trade_log = deque()
        self.wins = deque()
        self.losses = deque()
        self.lengths = deque()

        # Define column dtypes for better memory usage
        self.dtypes = {
            'entry_date': 'datetime64[ns]',
            'direction': 'category', 
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
    def trailing_stops(self, current_price, current_date, current_atr, adx_value):
        if self.active_trades.empty:
            return False

        current_price_f32 = np.float32(current_price)
        current_atr_f32 = np.float32(current_atr)
        atr_multiplier = np.float32(2.5 if adx_value > 30 else 1.5)

        # Initialize a boolean Series for stops_hit, default to False for all trades
        stops_hit_flags = pd.Series([False] * len(self.active_trades), index=self.active_trades.index)

        # --- Process Long Positions ---
        long_trades_mask = self.active_trades['multiplier'] == 1
        if long_trades_mask.any():
            # Update highest prices for active long trades
            self.active_trades.loc[long_trades_mask, 'highest_close_since_entry'] = np.maximum(
                self.active_trades.loc[long_trades_mask, 'highest_close_since_entry'],
                current_price_f32
            )
            # Calculate new stop-loss levels for long trades
            new_stops_long = self.active_trades.loc[long_trades_mask, 'highest_close_since_entry'] - (atr_multiplier * current_atr_f32)
            # Update stop-loss (only if new stop is higher)
            self.active_trades.loc[long_trades_mask, 'stop_loss'] = np.maximum(
                self.active_trades.loc[long_trades_mask, 'stop_loss'],
                new_stops_long
            )
            # Check if current price hits the stop-loss for long trades
            stops_hit_flags[long_trades_mask] = current_price_f32 <= self.active_trades.loc[long_trades_mask, 'stop_loss']

        # --- Process Short Positions ---
        short_trades_mask = self.active_trades['multiplier'] == -1
        if short_trades_mask.any():
            # Update lowest prices for active short trades
            self.active_trades.loc[short_trades_mask, 'lowest_close_since_entry'] = np.minimum(
                self.active_trades.loc[short_trades_mask, 'lowest_close_since_entry'],
                current_price_f32
            )
            # Calculate new stop-loss levels for short trades
            new_stops_short = self.active_trades.loc[short_trades_mask, 'lowest_close_since_entry'] + (atr_multiplier * current_atr_f32)
            # Update stop-loss (only if new stop is lower)
            self.active_trades.loc[short_trades_mask, 'stop_loss'] = np.minimum(
                self.active_trades.loc[short_trades_mask, 'stop_loss'],
                new_stops_short
            )
            # Check if current price hits the stop-loss for short trades
            stops_hit_flags[short_trades_mask] = current_price_f32 >= self.active_trades.loc[short_trades_mask, 'stop_loss']

        # --- Process Exits for Hit Stops ---
        if stops_hit_flags.any():
            indices_to_remove = []
            trades_that_hit_stop = self.active_trades[stops_hit_flags]

            for idx, trade in trades_that_hit_stop.iterrows():
                pnl = self.exit_pnl(trade, current_date, current_price_f32, trade['share_amount'], 'Trailing Stop')
                self.portfolio_value += pnl
                # Reduce allocated capital. Consistent with process_exits, using current_price.
                # Consider if trade['entry_price'] * trade['share_amount'] is more appropriate for allocated capital.
                self.allocated_capital -= (current_price_f32 * trade['share_amount'])
                self.position_count -= 1
                indices_to_remove.append(idx)
            
            if indices_to_remove:
                self.active_trades = self.active_trades.drop(index=indices_to_remove).reset_index(drop=True)
            return True # Indicates one or more trailing stops were hit and processed

        return False # No trailing stops were hit
    # ----------------------------------------------------------------------------------------------------------
    def process_exits(self, current_date, current_price, direction_to_exit, Trim=False): 

        if self.active_trades.empty:
            return 0.0

        total_pnl = 0.0
        # Use a boolean mask for marking trades to remove. Initialize to False.
        # Important: Get indices from the original DataFrame before any potential filtering
        original_indices = self.active_trades.index
        indices_to_remove_mask = pd.Series(False, index=original_indices)
        
        # Iterate over a copy of the DataFrame filtered by direction_to_exit to avoid issues while modifying
        # Get current indices of trades matching the direction
        relevant_trade_indices = self.active_trades[self.active_trades['direction'] == direction_to_exit].index

        for idx in relevant_trade_indices:
            # If already marked for removal by a previous iteration/check (should not happen with this loop structure but good for safety)
            if indices_to_remove_mask.get(idx, False): # Use .get for safety if idx might somehow be invalid
                continue

            trade = self.active_trades.loc[idx] # Get the most current version of the trade
            current_shares = trade['share_amount']
            
            if current_shares <= 0: # Should ideally be caught earlier or not happen
                indices_to_remove_mask[idx] = True
                continue

            # Priority 1: Check Stop Loss for this specific trade
            sl_hit = False
            if trade['direction'] == 'Long' and current_price <= trade['stop_loss']:
                sl_hit = True
            elif trade['direction'] == 'Short' and current_price >= trade['stop_loss']:
                sl_hit = True
            
            if sl_hit:
                pnl = self.exit_pnl(trade, current_date, current_price, 
                                current_shares, 'Stop Loss')
                total_pnl += pnl
                self.portfolio_value += pnl
                indices_to_remove_mask[idx] = True # Mark for removal
                self.position_count -= 1
                self.allocated_capital -= (current_price * current_shares) # Assuming current_price for de-allocation
                continue # Stop further processing for this trade if SL is hit

            # Priority 2: Check Take Profit for this specific trade
            tp_hit = False
            if pd.notna(trade['take_profit']): # Ensure take_profit is not NaN
                if trade['direction'] == 'Long' and current_price >= trade['take_profit']:
                    tp_hit = True
                elif trade['direction'] == 'Short' and current_price <= trade['take_profit']:
                    tp_hit = True
            
            if tp_hit:
                pnl = self.exit_pnl(trade, current_date, current_price, 
                                current_shares, 'Take Profit')
                total_pnl += pnl
                self.portfolio_value += pnl
                indices_to_remove_mask[idx] = True # Mark for removal
                self.position_count -= 1
                self.allocated_capital -= (current_price * current_shares)
                continue # Stop further processing for this trade if TP is hit

            # Priority 3: Process Trim or Full Signal Exit (if not SL/TP hit for this trade)
            # This part is only reached if the specific trade was not stopped out by SL or TP above.
            if Trim:
                shares_to_exit = int(current_shares * 0.3)
                if shares_to_exit > 0:
                    pnl = self.exit_pnl(trade, current_date, current_price, 
                                    shares_to_exit, 'Partial Signal Exit')
                    total_pnl += pnl
                    self.portfolio_value += pnl
                    self.allocated_capital -= (current_price * shares_to_exit)
                    
                    remaining_shares = current_shares - shares_to_exit
                    if remaining_shares > 0:
                        self.active_trades.loc[idx, 'share_amount'] = remaining_shares
                        # position_count is not decremented for partial exits
                    else: # Trim resulted in full exit
                        indices_to_remove_mask[idx] = True # Mark for removal
                        self.position_count -= 1
            else:  # Full Signal Exit
                pnl = self.exit_pnl(trade, current_date, current_price,
                                current_shares, 'Full Signal Exit')
                total_pnl += pnl
                self.portfolio_value += pnl
                indices_to_remove_mask[idx] = True # Mark for removal
                self.position_count -= 1
                self.allocated_capital -= (current_price * current_shares)
            
        # Remove all marked positions at once after iteration
        if indices_to_remove_mask.any():
            self.active_trades = self.active_trades[~indices_to_remove_mask].reset_index(drop=True)
            
        return total_pnl
    # ----------------------------------------------------------------------------------------------------------
    def process_entry(self, current_date, entry_params, direction=None):
        
        is_long = direction == 'Long'
        direction_mult = 1 if is_long else -1
        
        # 1. Calculate Initial Stop (ATR + ADX adjusted)
        atr_multiplier = 2.5 if entry_params['adx'] > 25 else 1.5
        min_stop = entry_params['price'] * 0.005  # 0.5% minimum
        stop_distance = max(entry_params['atr'] * atr_multiplier, min_stop)
        initial_stop = entry_params['price'] - (stop_distance * direction_mult)
        
        # 2. Position Sizing Based on Risk
        risk_per_share = abs(entry_params['price'] - initial_stop)
        max_risk_amount = entry_params['portfolio_value'] * entry_params['risk']
        shares_by_risk = int(max_risk_amount / max(risk_per_share, 1e-9))  # Avoid division by zero

        # Calculate shares based on position size limit
        max_position_value = entry_params['portfolio_value'] * entry_params['size_limit']
        shares_by_size = int(max_position_value / entry_params['price'])

        shares = min(shares_by_risk, shares_by_size)

        # Calculate dollar amount for this position
        position_dollar_amount = shares * entry_params['price']
        actual_position_size = position_dollar_amount / entry_params['portfolio_value']

        # 3. Check total portfolio exposure (add here)
        if not self.active_trades.empty:
            total_exposure = self.active_trades['position_size'].sum()
            max_total_exposure = 1.0  # 100% total exposure limit
            
            if total_exposure + actual_position_size > max_total_exposure:
                # Calculate how much exposure is available
                available_exposure = max_total_exposure - total_exposure
                if available_exposure <= 0:
                    return False
                    
                # Reduce position size to fit within available exposure
                adjusted_shares = int(available_exposure * entry_params['portfolio_value'] / entry_params['price'])
                if adjusted_shares <= 0:
                    return False
                    
                shares = adjusted_shares
                position_dollar_amount = shares * entry_params['price']
                actual_position_size = position_dollar_amount / entry_params['portfolio_value']

        # Check position size constraints
        if actual_position_size > entry_params['size_limit']:
            # Reduce shares to meet position size limit
            shares = int(entry_params['size_limit'] * entry_params['portfolio_value'] / entry_params['price'])
            position_dollar_amount = shares * entry_params['price']

        # Check if we have enough available capital
        available_capital = self.portfolio_value - self.allocated_capital
        if position_dollar_amount > available_capital:
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
        'direction': direction,
        'multiplier': direction_mult,
        'entry_price': entry_params['price'],
        'stop_loss': initial_stop,
        'take_profit': take_profit,
        'position_size': actual_position_size,  # Store actual position size used
        'share_amount': shares,
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
        
        self.allocated_capital += position_dollar_amount
        self.position_count += 1
        return True
    # ----------------------------------------------------------------------------------------------------------    
    def exit_pnl(self, trade_series, exit_date, exit_price, shares_to_exit, reason):
        
        entry_price = trade_series['entry_price']
        entry_date = trade_series['entry_date']
        trade_direction = trade_series['direction']

        pnl = 0
        if trade_direction == 'Long':
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
            'Direction': trade_direction,
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Shares': shares_to_exit,
            'PnL': pnl,
            'Duration': duration,
            'Exit Reason': reason
        })
        return pnl
    # ----------------------------------------------------------------------------------------------------------
    def calculate_unrealized_pnl(self, current_price):
        if self.active_trades.empty: return 0.0
        pnl_values = (current_price - self.active_trades['entry_price']) * \
                     self.active_trades['share_amount'] * \
                     self.active_trades['multiplier'] # Assuming 'multiplier' column exists and is correct
        
        return pnl_values.sum()
    # ----------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
def create_empty_stats(risk_free_rate=0.04):
    return {
        'Total Trades': 0,
        'Win Rate': 0,
        'Return (%)': 0,
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
        'Return (%)': net_profit_pct,
        'Profit Factor': profit_factor,
        'Expectancy (%)': expectancy_pct,
        'Max Drawdown (%)': max_drawdown,
        'Annualized Return (%)': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio
    }
# --------------------------------------------------------------------------------------------------------------------------
def parameter_test(df, long_risk, long_reward, short_risk, short_reward, position_size, 
                           tech_params, trial=None):
    bad_metrics_template = []
    for metric_name in OPTIMIZATION_DIRECTIONS:
        if OPTIMIZATION_DIRECTIONS[metric_name] == 'maximize':
            bad_metrics_template.append(-np.inf)
        else: # minimize
            bad_metrics_template.append(np.inf)
    
    # For debugging  
    if trial:
        trial_num = trial.number
    else:
        trial_num = "Unknown"
        
    try:
        trade_log, trade_stats, equity_curve, returns_series = momentum(
            df,
            long_risk=long_risk,
            long_reward=long_reward,
            short_risk=short_risk,
            short_reward=short_reward,
            position_size=position_size,
            max_positions=tech_params.get('MAX_OPEN_POSITIONS', MAX_OPEN_POSITIONS),
            adx_threshold_sig=tech_params.get('ADX_THRESHOLD', ADX_THRESHOLD_DEFAULT),
            min_buy_score_sig=tech_params.get('MIN_BUY_SCORE', MIN_BUY_SCORE_DEFAULT),
            min_sell_score_sig=tech_params.get('MIN_SELL_SCORE', MIN_SELL_SCORE_DEFAULT),
            signal_weights_sig=tech_params.get('SIGNAL_WEIGHTS', SIGNAL_WEIGHTS),
        )
        
        # 5. Process metrics
        required_metrics_keys = ['Sharpe Ratio', 'Profit Factor', 'Return (%)', 'Max Drawdown (%)']
        if trade_stats and all(metric in trade_stats for metric in required_metrics_keys):
            metrics = [
                trade_stats['Sharpe Ratio'],
                trade_stats['Profit Factor'],
                trade_stats['Return (%)'],
                trade_stats['Max Drawdown (%)']
            ]
            
            if trial:
                trial.set_user_attr('sharpe', metrics[0])
                trial.set_user_attr('profit_factor', metrics[1])
                trial.set_user_attr('returns', metrics[2])
                trial.set_user_attr('max_drawdown', metrics[3])
                trial.set_user_attr('num_trades', len(trade_log) if trade_log else 0)
                
            return metrics

        return bad_metrics_template
        
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Error in parameter evaluation for trial {trial_num}: {e}")
        return bad_metrics_template
# -------------------------------------------------------------------------------------------------------------
def objectives(trial, base_df):
    
    # Define parameter search spaces for both basic and technical parameters
    bad_metrics_template = []
    for metric_name in OPTIMIZATION_DIRECTIONS: # Ensure order matches OPTIMIZATION_DIRECTIONS
        if OPTIMIZATION_DIRECTIONS[metric_name] == 'maximize':
            bad_metrics_template.append(-np.inf)
        else: # minimize
            bad_metrics_template.append(np.inf)
    params = {
        # Basic parameters
        'long_risk': trial.suggest_float('long_risk', 0.01, 0.05, step=0.01),
        'long_reward': trial.suggest_float('long_reward', 0.02, 0.15, step=0.01),
        'short_risk': trial.suggest_float('short_risk', 0.01, 0.03, step=0.01),
        'short_reward': trial.suggest_float('short_reward', 0.02, 0.9, step=0.01),
        'position_size': trial.suggest_float('position_size', 0.02, 0.20, step=0.01),
        'max_open_positions': trial.suggest_int('max_open_positions', 3, 20,),
        
        # Technical parameters
        'adx_threshold': trial.suggest_float('adx_threshold', 15.0, 35.0, step=1.0),
        'min_buy_score': trial.suggest_float('min_buy_score', 1.0, 5.0, step=0.5),
        'min_sell_score': trial.suggest_float('min_sell_score', 1.0, 5.0, step=0.5),
        
        # Signal weights
        'trend_primary': trial.suggest_float('trend_primary', 0.5, 8.0, step=0.5),
        'trend_strength': trial.suggest_float('trend_strength', 0.5, 8.0, step=0.5),
        'weekly_trend': trial.suggest_float('weekly_trend', 0.5, 8.0, step=0.5),
        'rsi': trial.suggest_float('rsi', 0.5, 8.0, step=0.5),
        'volume': trial.suggest_float('volume', 0.5, 8.0, step=0.5),
        'bollinger': trial.suggest_float('bollinger', 0.5, 8.0, step=0.5)
    }
    
    # 1. Risk/Reward Validation
    if (params['long_risk'] >= params['long_reward'] or 
        params['short_risk'] >= params['short_reward']):
        trial.set_user_attr("invalid_reason", "Risk >= Reward")
        return bad_metrics_template

    if (params['long_reward']/params['long_risk'] > 3 or
        params['short_reward']/params['short_risk'] > 3):
        trial.set_user_attr("invalid_reason", "High Reward/Risk Ratio")
        return bad_metrics_template

    # 2. Position Sizing Validation
    max_position_risk = params['position_size'] * params['max_open_positions']
    if max_position_risk > 1.0:  # >100% exposure
        trial.set_user_attr("invalid_reason", "Excessive position risk")
        return bad_metrics_template
    
    # 3. Signal Weight Validation
    total_weights = sum([
        params['trend_primary'],
        params['trend_strength'],
        params['weekly_trend'],
        params['rsi'],
        params['volume'],
        params['bollinger']
    ])
    if total_weights < 1.5 or total_weights > 22.5:  # Arbitrary threshold for excessive weights
        trial.set_user_attr("invalid_reason", "Excessive signal weights")
        return bad_metrics_template
    
    # Set up signal weights
    signal_weights = SIGNAL_WEIGHTS.copy()
    signal_weights['primary_trend'] = params['trend_primary']
    signal_weights['trend_strength'] = params['trend_strength']
    signal_weights['weekly_trend'] = params['weekly_trend']
    signal_weights['rsi'] = params['rsi']
    signal_weights['volume'] = params['volume']
    signal_weights['bollinger'] = params['bollinger']

    # Set up combined tech params
    tech_params = { 
        'ADX_THRESHOLD': params['adx_threshold'],
        'MIN_BUY_SCORE': params['min_buy_score'],
        'MIN_SELL_SCORE': params['min_sell_score'],
        'SIGNAL_WEIGHTS': signal_weights,
        'MAX_OPEN_POSITIONS': params['max_open_positions']
    }
    
    # Evaluate this parameter set
    return parameter_test(
        base_df.copy(),
        params['long_risk'],
        params['long_reward'],
        params['short_risk'],
        params['short_reward'],
        params['position_size'],
        tech_params,
        trial
    )
# --------------------------------------------------------------------------------------------------------------
def optimize(prepared_data):
    # Optimizing parameters for Optuna
    target_metrics = list(OPTIMIZATION_DIRECTIONS.keys())
    opt_directions = [OPTIMIZATION_DIRECTIONS[metric] for metric in target_metrics]
    n_trials=500
    min_completed_trials=20
    timeout=3600

    data = prepared_data.copy()
    if data.empty:
        print("Warning: Empty dataframe provided to optimize.")
        return None
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base_pruner=optuna.pruners.HyperbandPruner(
            min_resource=5, max_resource=n_trials, reduction_factor=3)
    # Create Optuna study with pruning
    study = optuna.create_study(
        directions=opt_directions,  # Direction for each metric
        study_name=f"strategy_optimization",
        pruner=optuna.pruners.PatientPruner(base_pruner, patience=5),
        sampler=optuna.samplers.NSGAIIISampler(seed=42, population_size=50)  # Use NSGA-III for multi-objective
    )

    # Define the objective function
    objective_func = partial(objectives, base_df=data)

    # Start timing
    start_time = datetime.now()
        
    # Run optimization with progress bar
    completed_trials=0
    try:
        study.optimize(objective_func, n_trials=n_trials, timeout=timeout, 
                n_jobs=max(1, mp.cpu_count() - 1),
                show_progress_bar=True)
        completed_trials = len(study.trials)
    except Exception as e:
        print(f"Optimization error: {e}")

    # Ensure minimum number of trials are completed
    if completed_trials < min_completed_trials:
        print(f"Only {completed_trials}/{min_completed_trials} minimum trials completed. Running additional trials...")
        additional_trials = min_completed_trials - completed_trials
        study.optimize(objective_func, n_trials=additional_trials,
                      n_jobs=1, show_progress_bar=True)
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"\nOptimization completed in {elapsed_time}")
    
    # Get Pareto front solutions
    all_trials = study.trials

    # Filter out failed trials and sort by custom criteria
    filtered_trials = []
    for trial in all_trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.values is not None:
            # Get trial metrics
            metrics_dict = {
                'sharpe': trial.values[0],
                'profit_factor': trial.values[1],
                'returns': trial.values[2],
                'max_drawdown': trial.values[3]
            }
            
            # Skip trials with invalid metrics (-inf/inf)
            if any(np.isinf(v) for v in trial.values):
                continue
                
            # Calculate combined score for ranking
            combined_score = (
                OBJECTIVE_WEIGHTS['sharpe'] * metrics_dict['sharpe'] +
                OBJECTIVE_WEIGHTS['profit_factor'] * min(metrics_dict['profit_factor'], 100) +
                OBJECTIVE_WEIGHTS['returns'] * metrics_dict['returns'] -
                OBJECTIVE_WEIGHTS['max_drawdown'] * abs(metrics_dict['max_drawdown'])
            )
            
            filtered_trials.append((trial, combined_score))

    # Sort trials by combined score
    filtered_trials.sort(key=lambda x: x[1], reverse=True)

    # Extract just the trials for display
    pareto_front = [trial_tuple[0] for trial_tuple in filtered_trials]

    if not pareto_front:
        print("No valid trials found after filtering. Try adjusting the optimization parameters.")
        return None, study

    selected_params = None
    while True:  # Loop until user confirms a selection
        print("\nTop Pareto Optimal Solutions:")
        display_count = min(10, len(pareto_front))  # Show at most 5 trials
        if display_count == 0:
            print("No valid trials to display.")
            return None, study
            
        top_trials(pareto_front[:display_count])
        
        # Ask user to select a parameter set
        try:
            choice_str = input(f"\nSelect parameter set (1-{min(10, len(pareto_front))}) to simulate, or 'exit': ")
            if choice_str.lower() == 'exit':
                print("Exiting parameter selection.")
                return None, study
            choice = int(choice_str)
            if choice < 1 or choice > min(10, len(pareto_front)):
                print(f"Invalid selection. Please choose a number between 1 and {min(10, len(pareto_front))}")
                continue
            # Process selected parameters
            selected_trial = pareto_front[choice-1]
            selected_params = params_to_result(selected_trial.params)
            tech_params = selected_params.get('tech_params', {})
            
            # Run simulation with selected parameters
            print(f"\nSimulating with parameter set {choice}...\n")
            
            # Call test function with selected parameters
            test_result = test(
                df_input=data,
                long_risk=selected_params['long_risk'],
                long_reward=selected_params['long_reward'],
                short_risk=selected_params['short_risk'],
                short_reward=selected_params['short_reward'],
                position_size=selected_params['position_size'],
                max_positions_param=tech_params.get('MAX_OPEN_POSITIONS', MAX_OPEN_POSITIONS),
                adx_thresh=tech_params.get('ADX_THRESHOLD', ADX_THRESHOLD_DEFAULT),
                min_buy_s=tech_params.get('MIN_BUY_SCORE', MIN_BUY_SCORE_DEFAULT),
                min_sell_s=tech_params.get('MIN_SELL_SCORE', MIN_SELL_SCORE_DEFAULT),
                signal_weights=tech_params.get('SIGNAL_WEIGHTS', SIGNAL_WEIGHTS),
            )
            
            # Ask for confirmation
            confirm = input("\nUse these parameters? (Yes/No): ").strip().lower()
            if confirm in ["yes", "y"]:
                if test_result and 'trade_stats' in test_result:
                    trade_stats = test_result['trade_stats']
                    # Update performance_metrics with the new set of keys
                    selected_params['performance_metrics'].update({
                        'sharpe': trade_stats.get('Sharpe Ratio', 0),
                        'profit_factor': trade_stats.get('Profit Factor', 0),
                        'returns': trade_stats.get('Return (%)', 0), # Or 'Annualized Return (%)'
                        'max_drawdown': trade_stats.get('Max Drawdown (%)', 0),
                        'win_rate': trade_stats.get('Win Rate', 0), # Keep additional useful stats
                        'num_trades': trade_stats.get('Total Trades', 0)
                    })
                break  # Exit the loop if confirmed
            # Loop continues showing the top 5 again if user says no
            
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"Error simulating parameters: {e}")
            traceback.print_exc()
    
    return selected_params, study
# -------------------------------------------------------------------------------------------------------------
def top_trials(top_trials):
    print("\nPareto Front Solutions:")
    
    score_headers = [
        "Trial", "Sharpe", "Profit.F", "Return", "MaxDD%", "Combined Score"
    ]
    
    score_rows = []
    metric_keys_in_order = list(OPTIMIZATION_DIRECTIONS.keys())
    
    for i, trial in enumerate(top_trials):
        if trial.values is None or len(trial.values) != len(metric_keys_in_order):
            print(f"Skipping trial {i+1} due to missing metrics")
            continue
            
        trial_metrics_dict = {key: trial.values[idx] for idx, key in enumerate(metric_keys_in_order)}

        # Calculate combined score for display
        combined_score = (
            OBJECTIVE_WEIGHTS['sharpe'] * trial_metrics_dict.get('sharpe', 0) +
            OBJECTIVE_WEIGHTS['profit_factor'] * min(trial_metrics_dict.get('profit_factor', 0), 100) +
            OBJECTIVE_WEIGHTS['returns'] * trial_metrics_dict.get('returns', 0) -
            OBJECTIVE_WEIGHTS['max_drawdown'] * abs(trial_metrics_dict.get('max_drawdown', 0))
        )

        # Create score row
        score_row = [
            f"{i+1}",  # Trial number
            f"{trial_metrics_dict.get('sharpe', 0):.2f}",
            f"{trial_metrics_dict.get('profit_factor', 0):.2f}",
            f"{trial_metrics_dict.get('returns', 0):.2f}%",
            f"{trial_metrics_dict.get('max_drawdown', 0):.2f}%",
            f"{combined_score:.2f}"
        ]
        score_rows.append(score_row)
    
    if score_rows:
        print("\nCombined Scores (weighted - for illustrative ranking):")
        print(tabulate(score_rows, headers=score_headers, tablefmt="grid"))
    else:
        print("No valid trials to display.")
# -------------------------------------------------------------------------------------------------------------
def params_to_result(params):
    # Extract base parameters
    base_params = {
        'long_risk': params['long_risk'],
        'long_reward': params['long_reward'],
        'short_risk': params['short_risk'],
        'short_reward': params['short_reward'],
        'position_size': params['position_size']
    }
    
    # Configure signal weights
    signal_weights = SIGNAL_WEIGHTS.copy()
    signal_weights.update({
        'primary_trend': params['trend_primary'],
        'trend_strength': params['trend_strength'],
        'weekly_trend': params['weekly_trend'],
        'rsi': params['rsi'],
        'volume': params['volume'],
        'bollinger': params['bollinger']
    })

    # Configure technical parameters
    tech_params = {
        # Core parameters
        'FAST': FAST, 
        'SLOW': SLOW,
        'WEEKLY_MA_PERIOD': WEEKLY_MA_PERIOD,
        
        # RSI settings
        'RSI_LENGTH': RSI_LENGTH,
        'RSI_OVERSOLD': RSI_OVERSOLD, 
        'RSI_OVERBOUGHT': RSI_OVERBOUGHT,
        
        # Volatility settings
        'BB_LEN': BB_LEN,
        'ST_DEV': ST_DEV,
        
        # Signal generation parameters
        'ADX_THRESHOLD': params['adx_threshold'],
        'MIN_BUY_SCORE': params['min_buy_score'],
        'MIN_SELL_SCORE': params['min_sell_score'],
        'SIGNAL_WEIGHTS': signal_weights,
        
        # Position management
        'MAX_OPEN_POSITIONS': params['max_open_positions'],
    }

    # Combine all parameters into final result
    result = {
        **base_params,
        'tech_params': tech_params,
        # Initialize performance metrics
        'performance_metrics': {
            'win_rate': 0.0,
            'net_profit_pct': 0.0,
            'max_drawdown': 0.0,
            'sharpe': 0.0,
            'profit_factor': 0.0,
            'num_trades': 0
        }
    }
    
    return result
# -------------------------------------------------------------------------------------------------------------
def test(df_input, long_risk=DEFAULT_LONG_RISK, long_reward=DEFAULT_LONG_REWARD, short_risk=DEFAULT_SHORT_RISK, 
         short_reward=DEFAULT_SHORT_REWARD, position_size=DEFAULT_POSITION_SIZE, max_positions_param=MAX_OPEN_POSITIONS,
         adx_thresh=ADX_THRESHOLD_DEFAULT, min_buy_s=MIN_BUY_SCORE_DEFAULT, min_sell_s=MIN_SELL_SCORE_DEFAULT, signal_weights=SIGNAL_WEIGHTS): 
    
    df = df_input.copy()
    
    trade_log, trade_stats, portfolio_equity, trade_returns = momentum(
        df_input, long_risk=long_risk, long_reward=long_reward, short_risk=short_risk, short_reward=short_reward,
        position_size=position_size, risk_free_rate=0.04, leverage_ratio=LEVERAGE, max_positions=max_positions_param,
        adx_threshold_sig=adx_thresh, min_buy_score_sig=min_buy_s, min_sell_score_sig=min_sell_s,
        signal_weights_sig=signal_weights)

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

    print(f"\n=== STRATEGY SUMMARY ===") 
    print(f"Long Risk: {long_risk*100:.1f}% | Long Reward: {long_reward*100:.1f}%")
    print(f"Short Risk: {short_risk*100:.1f}% | Short Reward: {short_reward*100:.1f}%")
    print(f"Position Size: {position_size*100:.1f}% | Max Open Positions: {max_positions_param}")
    print(f"ADX Threshold: {adx_thresh:.1f} | Min Buy Score: {min_buy_s:.1f} | Min Sell Score: {min_sell_s:.1f}")
    print(f"Signal Weights: {signal_weights}")
    
    metrics = [
        ["Starting Capital [$]", f"{portfolio_equity.iloc[0]:,.2f}"],
        ["Ending Capital [$]", f"{portfolio_equity.iloc[-1]:,.2f}"],
        ["Start", f"{df.index[0].strftime('%Y-%m-%d')}"],
        ["End", f"{df.index[-1].strftime('%Y-%m-%d')}"],
        ["Duration [days]", f"{exposure_time}"],
        ["Equity Final [$]", f"{portfolio_equity.iloc[-1]:,.2f}"],
        ["Equity Peak [$]", f"{peak_equity:,.2f}"],
        ["Return [%]", f"{trade_stats['Return (%)']:.2f}"],
        ["Buy & Hold Return [%]", f"{buy_hold_return:.2f}"],
        ["Annual Return [%]", f"{trade_stats['Annualized Return (%)']:.2f}"],
        ["Sharpe Ratio", f"{trade_stats['Sharpe Ratio']:.2f}"],
        ["Sortino Ratio", f"{trade_stats['Sortino Ratio']:.2f}"],
        ["Max. Drawdown [%]", f"{trade_stats['Max Drawdown (%)']:.2f}"],
    ]
    print(tabulate(metrics, tablefmt="simple", colalign=("left", "right")))
    
    print(f"\n=== TRADE SUMMARY ===") 
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
     
    return None
# -------------------------------------------------------------------------------------------------------------
def monte_carlo_returns_test(data, params, num_simulations=5000):
    # Run original strategy to get baseline performance
    trade_log, trade_stats, equity_curve, returns_series = momentum(
        data,
        long_risk=params['long_risk'],
        long_reward=params['long_reward'],
        short_risk=params['short_risk'],
        short_reward=params['short_reward'],
        position_size=params['position_size'],
        max_positions=params['tech_params']['MAX_OPEN_POSITIONS'],
        adx_threshold_sig=params['tech_params']['ADX_THRESHOLD'],
        min_buy_score_sig=params['tech_params']['MIN_BUY_SCORE'],
        min_sell_score_sig=params['tech_params']['MIN_SELL_SCORE'],
        signal_weights_sig=params['tech_params']['SIGNAL_WEIGHTS']
    )
    # Clean returns series
    clean_returns = returns_series.replace([np.inf, -np.inf], np.nan).dropna()
    observed_sharpe = clean_returns.mean() / clean_returns.std() * np.sqrt(252)

    # Stationary bootstrap implementation
    def stationary_bootstrap(series, block_size=7):
        n = len(series)
        indices = []
        current_idx = np.random.randint(0, n)
        
        while len(indices) < n:
            # Generate geometric length for current block
            block_length = np.random.geometric(1/block_size)
            indices.extend(range(current_idx, min(current_idx + block_length, n)))
            current_idx = np.random.randint(0, n)
            
        return series.iloc[indices[:n]]
    
    # Generate null distribution
    null_stats = {
        'sharpe_ratios': [],
        'returns': [],
        'max_drawdowns': [],
        'profit_factors': []
    }

    print("\nRunning Monte Carlo Simulations...")
    for i in tqdm(range(num_simulations)):
        # Resample returns while preserving time series properties
        resampled_returns = stationary_bootstrap(clean_returns)
        
        # Calculate metrics for this simulation
        sim_sharpe = resampled_returns.mean() / resampled_returns.std() * np.sqrt(252)
        sim_return = (1 + resampled_returns).prod() - 1
        
        # Calculate drawdown for this simulation
        cum_returns = (1 + resampled_returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Store results
        null_stats['sharpe_ratios'].append(sim_sharpe)
        null_stats['returns'].append(sim_return)
        null_stats['max_drawdowns'].append(max_drawdown)
    
    # Calculate p-values and percentiles
    results = {
        'observed_metrics': {
            'sharpe': observed_sharpe,
            'return': clean_returns.sum(),
            'max_drawdown': trade_stats['Max Drawdown (%)'],
            'profit_factor': trade_stats['Profit Factor']
        },
        'simulation_metrics': {
            'sharpe_mean': np.mean(null_stats['sharpe_ratios']),
            'sharpe_std': np.std(null_stats['sharpe_ratios']),
            'return_mean': np.mean(null_stats['returns']),
            'return_std': np.std(null_stats['returns']),
            'max_drawdown_mean': np.mean(null_stats['max_drawdowns']),
            'max_drawdown_std': np.std(null_stats['max_drawdowns'])
        },
        'p_values': {
            'sharpe': (np.array(null_stats['sharpe_ratios']) >= observed_sharpe).mean(),
            'return': (np.array(null_stats['returns']) >= clean_returns.sum()).mean(),
            'max_drawdown': (np.array(null_stats['max_drawdowns']) <= trade_stats['Max Drawdown (%)']).mean()
        },
        'percentiles': {
            'sharpe': stats.percentileofscore(null_stats['sharpe_ratios'], observed_sharpe),
            'return': stats.percentileofscore(null_stats['returns'], clean_returns.sum()),
            'max_drawdown': stats.percentileofscore(null_stats['max_drawdowns'], trade_stats['Max Drawdown (%)'])
        }
    }
    
    # Print summary
    print("\n=== Monte Carlo Simulation Results ===")
    print(f"Number of simulations: {num_simulations}")
    print("\nObserved vs Simulated Metrics:")
    print(f"Sharpe Ratio: {observed_sharpe:.2f} (p-value: {results['p_values']['sharpe']:.3f})")
    print(f"Total Return: {clean_returns.sum()*100:.2f}% (p-value: {results['p_values']['return']:.3f})")
    print(f"Max Drawdown: {trade_stats['Max Drawdown (%)']:.2f}% (p-value: {results['p_values']['max_drawdown']:.3f})")
    
    print("\nPercentile Rankings:")
    print(f"Sharpe Ratio: {results['percentiles']['sharpe']:.1f}th percentile")
    print(f"Total Return: {results['percentiles']['return']:.1f}th percentile")
    print(f"Max Drawdown: {results['percentiles']['max_drawdown']:.1f}th percentile")
    
    return results
# -------------------------------------------------------------------------------------------------------------
def walk_forward_analysis(prepared_data, parameters, train_months=24, test_months=6, min_train=12, n_jobs=-1):
    """
    Performs walk-forward analysis on the momentum strategy
    
    Args:
        prepared_data: DataFrame with prepared price data and indicators
        parameters: Dictionary of optimized strategy parameters
        train_months: Number of months for training window
        test_months: Number of months for testing window
        min_train: Minimum months required for training
        n_jobs: Number of parallel jobs (-1 for all cores)
    """
    from statsmodels.tsa.stattools import adfuller
    from joblib import Parallel, delayed
    
    def run_strategy_window(data_window):
        """Run strategy on a specific data window"""
        trade_log, trade_stats, equity_curve, returns_series = momentum(
            data_window,
            long_risk=parameters['long_risk'],
            long_reward=parameters['long_reward'],
            short_risk=parameters['short_risk'],
            short_reward=parameters['short_reward'],
            position_size=parameters['position_size'],
            max_positions=parameters['tech_params']['MAX_OPEN_POSITIONS'],
            adx_threshold_sig=parameters['tech_params']['ADX_THRESHOLD'],
            min_buy_score_sig=parameters['tech_params']['MIN_BUY_SCORE'],
            min_sell_score_sig=parameters['tech_params']['MIN_SELL_SCORE'],
            signal_weights_sig=parameters['tech_params']['SIGNAL_WEIGHTS']
        )
        return {'stats': trade_stats, 'returns_series': returns_series}

    def process_window(train_start, train_end, test_end, data):
        """Process individual walk-forward window"""
        try:
            # Split data into train and test periods
            train = data.loc[train_start:train_end].copy()
            test = data.loc[train_end:test_end].copy()
            
            if train.empty or test.empty:
                return None
            
            # Run strategy on both periods
            train_result = run_strategy_window(train)
            test_result = run_strategy_window(test)
            
            # Calculate performance decay
            train_sharpe = train_result['stats']['Sharpe Ratio']
            test_sharpe = test_result['stats']['Sharpe Ratio']
            decay = train_sharpe - test_sharpe
            
            # Check consistency
            consistency = 1 if test_sharpe > 0 else 0
            
            # Test stationarity between periods
            combined_returns = pd.concat([
                pd.Series(train_result['returns_series']),
                pd.Series(test_result['returns_series'])
            ])
            adf_p = adfuller(combined_returns.dropna())[1]
            
            return {
                'train_period': (train_start, train_end),
                'test_period': (train_end, test_end),
                'train_sharpe': train_sharpe,
                'test_sharpe': test_sharpe,
                'decay_ratio': decay/max(0.01, abs(train_sharpe)),
                'consistency': consistency,
                'stationarity_p': adf_p
            }
            
        except Exception as e:
            print(f"Error processing window {train_start} to {test_end}: {e}")
            return None

    # Generate windows for analysis
    dates = prepared_data.index.sort_values()
    windows = []
    train_start = dates[0]
    
    while train_start <= dates[-1] - pd.DateOffset(months=min_train+test_months):
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        
        if test_end > dates[-1]:
            break
            
        windows.append((train_start, train_end, test_end, prepared_data))
        train_start += pd.DateOffset(months=test_months)

    # Run parallel processing
    print("\nRunning Walk-Forward Analysis...")
    with tqdm(total=len(windows)) as pbar:
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_window)(*window) for window in windows
        )
        pbar.update()

    # Filter out None results and create DataFrame
    results = [r for r in results if r is not None]
    if not results:
        print("No valid results from walk-forward analysis")
        return None
        
    df = pd.DataFrame(results)
    
    # Calculate aggregate metrics
    decay_stats = {
        'mean_decay': df['decay_ratio'].mean(),
        'std_decay': df['decay_ratio'].std(),
        'probability_positive': df['test_sharpe'].gt(0).mean(),
        'stationary_fraction': df['stationarity_p'].lt(0.05).mean()
    }
    
    # Print summary
    print("\n=== Walk-Forward Analysis Results ===")
    print(f"Number of windows analyzed: {len(results)}")
    print(f"\nDecay Statistics:")
    print(f"Mean Performance Decay: {decay_stats['mean_decay']:.2f}")
    print(f"Decay Std Dev: {decay_stats['std_decay']:.2f}")
    print(f"Probability of Positive Sharpe: {decay_stats['probability_positive']*100:.1f}%")
    print(f"Fraction of Stationary Windows: {decay_stats['stationary_fraction']*100:.1f}%")
    
    print("\nWindow Details:")
    window_stats = pd.DataFrame({
        'Train Sharpe': df['train_sharpe'],
        'Test Sharpe': df['test_sharpe'],
        'Decay Ratio': df['decay_ratio']
    })
    print(tabulate(window_stats, headers='keys', tablefmt='grid'))
    
    return {'results': df, 'metrics': decay_stats}
# -------------------------------------------------------------------------------------------------------------
def main():
    """Improved main function with better error handling"""
    try:
        if isinstance(TICKER, list):
            ticker_str = TICKER[0]
        else:
            ticker_str = TICKER

        IS, OOS = get_data(ticker_str)
        df_prepared = prepare_data(IS)
        best_params, study = optimize(df_prepared)
        if best_params is None:
            mc_results = monte_carlo_returns_test(df_prepared, best_params)
            if mc_results['p_values']['sharpe'] < 0.05:
                print("✓ Strategy shows statistical significance in Monte Carlo testing")
                print("\n=== Strategy Robustness Analysis ===")
                oos_prepared = prepare_data(OOS)
                wfa_results = walk_forward_analysis(
                    oos_prepared, 
                    best_params,
                    train_months=12,  # 1 year training
                    test_months=3,    # 3 months testing
                    min_train=6       # Minimum 6 months required
                )
                if wfa_results and wfa_results['metrics']['probability_positive'] > 0.5:
                    print("✓ Strategy shows consistency in walk-forward testing")
                else:
                    print("⚠ Strategy shows poor consistency in walk-forward testing")
            else:
                print("⚠ Strategy fails to show statistical significance in Monte Carlo testing")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        return None
# -------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()