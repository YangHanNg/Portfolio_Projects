import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import yfinance as yf
from tabulate import tabulate  
import pandas_ta as ta 
import multiprocessing as mp
from multiprocessing import Pool, shared_memory
from collections import deque
from numba import jit
import optuna
import tqdm as tqdm
from functools import partial
from joblib import Parallel, delayed
import traceback
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
#==========================================================================================================================
#================== SCRIPT PARAMETERS =====================================================================================
#==========================================================================================================================

# CONTROLS
TYPE = 1 # 1. Full run # 2. Walk-Forward # 3. Monte Carlo # 4. Optimization # 5. Test
TICKER = 'SPY' # Ticker to analyze (only takes in one ticker at a time)
IS_HISTORY_DAYS = 1260  # Approx 5 years, for IS operations (Opt, MC, Test)

# TRADE CONTROL
INITIAL_CAPITAL = 25000.0 # Initial capital for the strategy
COMMISION = True # Set to True for commission

# OPTIMIZATION CONTROL
OPTIMIZATION = True # Set to True for optimization
TRIALS = 150 # Number of trials for optimization
TARGET_SCORE = 1.0
SCORE_TOLERANCE = 0.5

# MONTE CARLO CONTROL
BLOCK_SIZE = 20 # Size of blocks for random sampling

# WALK-FORWARD CONTROL
OPTIMIZATION_FREQUENCY = 252 # Number of days between optimizations (wfa)
OOS_WINDOW = 126 # Number of days for out-of-sample testing (wfa)
FINAL_OOS_YEARS = 5 # Number of years for final out-of-sample testing
FINAL_IS_YEARS = 5 # Number of years for in-sample testing
WFA_HISTORY_DAYS = IS_HISTORY_DAYS + FINAL_OOS_YEARS * 252  # Full history for WFA operations
TRADE_SUM = False # Set to True to sum trades in WFA

RISK_FREE_RATE_ANNUAL = 0.04   # Annual risk-free rate

#=========================================================================================================================
#================== STRATEGY PARAMETERS ==================================================================================
#=========================================================================================================================

# Optimization directions
OPTIMIZATION_DIRECTIONS = {
    'profit_factor': 'maximize',
    'avg_win_loss_ratio': 'maximize',
    'expectancy': 'maximize',
    'max_drawdown': 'minimize',
}
# Optimization objectives
OBJECTIVE_WEIGHTS = {       
    'profit_factor': 0.25, 
    'avg_win_loss_ratio': 0.25,
    'expectancy': 25.0,   
    'max_drawdown': 0.25   
}

# Optimization parameters
ADX_THRESHOLD_DEFAULT = 25
DEFAULT_LONG_RISK = 0.05
MAX_OPEN_POSITIONS = 30
MAX_POSITION_DURATION = 15

#=========================================================================================================================
#================== INDICATOR PARAMETERS =================================================================================
#=========================================================================================================================

# Moving average strategy parameters
FAST = 20
SLOW = 50
WEEKLY_MA_PERIOD = 50
LOOKBACK_BUFFER_DAYS = 75
# Average True Range
ATR_LENGTH = 14
# Average Directional Index
ADX_LENGTH = 14
# RSI strategy parameters
RSI_LENGTH = 14
# Bollinger Bands strategy parameters
BB_LEN = 20
ST_DEV = 2.0
MOMENTUM_LOOKBACK = 14
MOMENTUM_VOLATILITY_LOOKBACK = 21
RANKING_LOOKBACK_WINDOW = 60
VIX_ENTRY_THRESHOLD = 25  # Example: Only enter trades if VIX is below this
VIX_MA_PERIOD = 20        # Moving average period for VIX factor calculation

DEFAULT_SIGNAL_PROCESSING_PARAMS = {
    'weights': {
        'price_trend': 0.20,        # Element 1
        'rsi_zone': 0.20,           # Element 2
        'adx_slope': 0.20,          # Element 3
        'vol_accel': 0.20,          # Element 4
        'vix_factor': 0.20          # Element 5
    },
    'thresholds': {
        'buy_score': 55,
        'exit_score': 35,
        'immediate_exit_score': 25
    },
    'ranking_lookback_window': RANKING_LOOKBACK_WINDOW,
    'momentum_volatility_lookback': MOMENTUM_VOLATILITY_LOOKBACK
}

#==========================================================================================================================
#================== DATA RETRIEVAL & HANDLING =============================================================================
#==========================================================================================================================
def get_data(ticker):
    """
    Download historical data for the given ticker and split it into in-sample and out-of-sample datasets.
    """
    trading_days_per_year = 252
    oos_period_days = trading_days_per_year * FINAL_OOS_YEARS
    target_is_days = trading_days_per_year * FINAL_IS_YEARS
    data_start_year = 2013

    print(f"\nDownloading data for {ticker}...")

    try:
        data = yf.download(ticker, start=f"{data_start_year}-01-01", auto_adjust=True)

        if data.empty:
            print(f"No data downloaded for {ticker}.")
            return pd.DataFrame(), pd.DataFrame()
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            print(f"Missing required columns: {missing}")
            return pd.DataFrame(), pd.DataFrame()

        # Download VIX
        vix = yf.download("^VIX", start=data.index[0], end=data.index[-1], auto_adjust=True)['Close']
        vix = vix.reindex(data.index, method='ffill')
        data['VIX'] = vix

        core_cols = required_cols + ['VIX']
        still_missing = [col for col in core_cols if col not in data.columns]
        if still_missing:
            print(f"Missing core columns before dropna: {still_missing}")
            return pd.DataFrame(), pd.DataFrame()

        # Ensure all core columns are numeric float32
        for col in core_cols:
            if col in data.columns:
                col_data = data[col]
                if isinstance(col_data, pd.Series):
                    try:
                        data[col] = pd.to_numeric(col_data, errors='coerce').astype('float32')
                    except Exception as e:
                        print(f"Warning: Could not convert column '{col}' to float32. Error: {e}")
                else:
                    print(f"Warning: Column '{col}' is not a Series. Type: {type(col_data)}")
            else:
                print(f"Warning: Column '{col}' not found in DataFrame before conversion.")

        # Drop NaNs
        data = data.dropna(subset=core_cols)
        if data.empty:
            print("All rows dropped after NaN removal.")
            return pd.DataFrame(), pd.DataFrame()

        # Slice OOS and IS
        if len(data) < oos_period_days:
            oos_df = data.copy()
            data_before_oos = pd.DataFrame(columns=data.columns, index=pd.to_datetime([]))
        else:
            oos_df = data.iloc[-oos_period_days:].copy()
            data_before_oos = data.iloc[:-oos_period_days].copy()

        required_is = target_is_days + LOOKBACK_BUFFER_DAYS
        if len(data_before_oos) < required_is:
            is_df = data_before_oos.copy()
        else:
            is_df = data_before_oos.iloc[-required_is:].copy()

        print(f"IS: {len(is_df)} rows | OOS: {len(oos_df)} rows")
        return is_df, oos_df

    except Exception as e:
        print(f"Error in get_data: {e}")
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()
# -------------------------------------------------------------------------------------------------------------------------
def prepare_data(df_input, type=None):
    """
    Prepares the data with important indicators.
    """
    # 1. Data copy and critical checks
    df = df_input.copy()
    if df.empty:
        print("Warning: Empty dataframe provided to prepare_data.")
        return df
    
    if 'Close' not in df.columns or df['Close'].isnull().all():
        print("CRITICAL ERROR: 'Close' column is missing or unusable before indicator calculation.")
        return pd.DataFrame()
    
    # List of indicator columns that will be created
    indicator_columns = [
        f'{FAST}_ma', f'{SLOW}_ma', 'Volume_MA20', f'Weekly_MA{WEEKLY_MA_PERIOD}',
        'RSI', 'Upper_Band', 'Lower_Band', 'ATR', 'Close_26_ago', 'ADX',
        'volume_confirmed', 'weekly_uptrend',
        'price_roc_raw', 'ma_dist_raw', 'vol_accel_raw', 'adx_slope_raw', 'atr_pct_raw'
    ]

    # 2. Calculate Core Indicators
    try:
        # Main Indicators
        df[f'{FAST}_ma'] = df['Close'].rolling(window=FAST, min_periods=1).mean().fillna(0)
        df[f'{SLOW}_ma'] = df['Close'].rolling(window=SLOW, min_periods=1).mean().fillna(0)
        df['Volume_MA20'] = df['Volume'].rolling(window=20, min_periods=1).mean().fillna(0)
        
        # Weekly Moving Average
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df_weekly = df['Close'].resample('W').last()
        df[f'Weekly_MA{WEEKLY_MA_PERIOD}'] = (df_weekly.rolling(window=WEEKLY_MA_PERIOD, min_periods=1)
            .mean().reindex(df.index, method='ffill')).fillna(0) 
        
        # Relative Strength Index
        rsi_series = ta.rsi(df['Close'], length=RSI_LENGTH)
        df['RSI'] = rsi_series.fillna(0) # NaNs will persist if rsi_series has them
        
        # Bollinger Bands
        bb = ta.bbands(df['Close'], length=BB_LEN, std=ST_DEV)
        if isinstance(bb, pd.DataFrame):  # If bb is a DataFrame with multiple columns
            # Access specific columns by their proper names
            df['Upper_Band'] = bb[f'BBU_{BB_LEN}_{ST_DEV}'].fillna(0)
            df['Lower_Band'] = bb[f'BBL_{BB_LEN}_{ST_DEV}'].fillna(0)
        else:
            print(f"Warning: Bollinger Bands calculation failed or returned unexpected format: {type(bb)}")
            # Create empty columns to avoid errors
            df['Upper_Band'] = 0
            df['Lower_Band'] = 0

        # Average True Range
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=ATR_LENGTH).fillna(0)

        # Close price 26 periods ago
        df['Close_26_ago'] = df['Close'].shift(26).fillna(0)

        # Average Directional Index
        adx_result = ta.adx(df['High'], df['Low'], df['Close'], length=ADX_LENGTH)
        df['ADX'] = adx_result[f'ADX_{ADX_LENGTH}'].fillna(0)
        df['adx_level_raw'] = df['ADX'].fillna(0)
        
        # Volume and Weekly Trend Confirmations
        df['volume_confirmed'] = df['Volume'] > df['Volume_MA20'].fillna(False)
        
        weekly_ma_series = df[f'Weekly_MA{WEEKLY_MA_PERIOD}']
        df['weekly_uptrend'] = (df['Close'] > weekly_ma_series) & \
                                (weekly_ma_series.shift(1).ffill() < weekly_ma_series).fillna(False)
        
        # Calculate raw components for momentum score
        df['price_roc_raw'] = df['Close'].pct_change(MOMENTUM_LOOKBACK).fillna(0)
        
        ma_dist_raw_values = np.where(df[f'{FAST}_ma'].values != 0, 
                                      (df['Close'].values / df[f'{FAST}_ma'].values - 1), 0)
        df['ma_dist_raw'] = pd.Series(ma_dist_raw_values, index=df.index).fillna(0)

        vol_accel_raw_values = np.where(df['Volume_MA20'].values != 0, 
                                        df['Volume'].values / df['Volume_MA20'].values, 1.0)
        df['vol_accel_raw'] = pd.Series(vol_accel_raw_values, index=df.index).fillna(1.0)
        
        df['adx_slope_raw'] = df['ADX'].diff(MOMENTUM_LOOKBACK).fillna(0)
        
        atr_pct_raw_values = np.where(df['Close'].values != 0, 
                                      df['ATR'].values / df['Close'].values, 0)
        df['atr_pct_raw'] = pd.Series(atr_pct_raw_values, index=df.index).fillna(0)

        # VIX Factor Calculation
        if 'VIX' in df.columns:
            df['VIX_MA'] = df['VIX'].rolling(window=VIX_MA_PERIOD, min_periods=1).mean().fillna(method='bfill').fillna(0)
            # Factor: VIX_MA / VIX. Higher is "better" (VIX below its MA).
            # Add a small epsilon to VIX to prevent division by zero, though VIX is rarely zero.
            df['vix_factor_raw'] = np.where(df['VIX'].values > 1e-6, 
                                            df['VIX_MA'].values / (df['VIX'].values + 1e-6), 
                                            1.0) # Default to 1.0 if VIX is near zero
            df['vix_factor_raw'] = df['vix_factor_raw'].fillna(1.0) # Fill any remaining NaNs
        else:
            print("Warning: VIX column not found. Cannot calculate VIX factor.")
            df['vix_factor_raw'] = 1.0 # Neutral value if VIX is not available

        # Define RSI ideal zone parameters
        rsi_values = df['RSI']
        rsi_lower_taper_end = 20.0  # RSI values below this will have a score of 0
        rsi_ideal_low = 40.0       # Start of the ideal zone (score 1)
        rsi_ideal_high = 70.0      # End of the ideal zone (score 1)
        rsi_upper_taper_end = 90.0  # RSI values above this will have a score of 0

        # Conditions for np.select
        conditions = [
            (rsi_values >= rsi_ideal_low) & (rsi_values <= rsi_ideal_high),              # In ideal zone
            (rsi_values >= rsi_lower_taper_end) & (rsi_values < rsi_ideal_low),        # Tapering up to ideal zone
            (rsi_values > rsi_ideal_high) & (rsi_values <= rsi_upper_taper_end)          # Tapering down from ideal zone
        ]

        # Corresponding choices for scores
        # Ensure no division by zero if taper_end equals ideal_low/high (though not the case here)
        choices = [
            1.0,
            (rsi_values - rsi_lower_taper_end) / (rsi_ideal_low - rsi_lower_taper_end) 
                if (rsi_ideal_low - rsi_lower_taper_end) != 0 else 0.0,
            (rsi_upper_taper_end - rsi_values) / (rsi_upper_taper_end - rsi_ideal_high)
                if (rsi_upper_taper_end - rsi_ideal_high) != 0 else 0.0
        ]
        
        # Apply conditions to calculate raw score, default to 0.0 outside defined tapers
        rsi_zone_scores = np.select(conditions, choices, default=0.0)
        
        df['rsi_ideal_zone_raw'] = pd.Series(rsi_zone_scores, index=df.index).clip(0, 1).fillna(0.5)

        # Check if all expected indicator columns were created
        missing_calculated_indicators = [col for col in indicator_columns if col not in df.columns and col not in ['volume_confirmed', 'weekly_uptrend']] # Booleans have defaults
        if missing_calculated_indicators:
            print(f"CRITICAL ERROR in prepare_data: The following indicator columns were expected but not created: {missing_calculated_indicators}. Aborting preparation.")
            return pd.DataFrame()

    except Exception as e:
        print(f"CRITICAL ERROR during indicator calculation in prepare_data: {e}. Aborting preparation.")
        traceback.print_exc()
        return pd.DataFrame()
    
    # 3. Data type split
    if type == 1:
        # In-Sample Data
        df = df.iloc[-IS_HISTORY_DAYS:].copy()
        # Modify check to exclude boolean columns from zero value check
        boolean_columns = ['volume_confirmed', 'weekly_uptrend']
        potentially_zero_raw_cols = ['price_roc_raw', 'ma_dist_raw', 'adx_slope_raw', 'atr_pct_raw']
        numeric_columns = [col for col in df.columns 
                                          if col not in boolean_columns 
                                          and col not in potentially_zero_raw_cols
                                          and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
        
        # Check if df has null values in any column or zero values in numeric columns only
        has_null = df.isnull().any().any()
        has_zero_in_numeric = (df[numeric_columns] == 0).all().any() if numeric_columns else False
        
        if has_null or has_zero_in_numeric:
            print("Warning: DataFrame contains null or zero values after indicator calculation. Returning empty DataFrame.")
            
            return pd.DataFrame()
        else:
            print("DataFrame is valid after indicator calculation.")
        return df
    elif type == 2:
        # Full Data
        df = df.iloc[-WFA_HISTORY_DAYS:].copy()
        # Modify check to exclude boolean columns from zero value check
        boolean_columns = ['volume_confirmed', 'weekly_uptrend']
        potentially_zero_raw_cols = ['price_roc_raw', 'ma_dist_raw', 'adx_slope_raw', 'atr_pct_raw']
        numeric_columns = [col for col in df.columns 
                                          if col not in boolean_columns 
                                          and col not in potentially_zero_raw_cols
                                          and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
        
        # Check if df has null values in any column or zero values in numeric columns only
        has_null = df.isnull().any().any()
        has_zero_in_numeric = (df[numeric_columns] == 0).all().any() if numeric_columns else False
        
        if has_null or has_zero_in_numeric:
            print("Warning: DataFrame contains null or zero values after indicator calculation. Returning empty DataFrame.")
            
            return pd.DataFrame()
        else:
            print("DataFrame is valid after indicator calculation.")
        return df
    else:
        print("Warning: Invalid type provided to prepare_data. Returning empty DataFrame.")
        return pd.DataFrame()

#==========================================================================================================================
#================== TRADING STRATEGY ======================================================================================
#==========================================================================================================================
def signals(df, adx_threshold, params):
    
    # Initialize signals DataFrame with the same index
    signals_df = pd.DataFrame(index=df.index)
    
    # Setup signal parameters
    fast_ma_col = f"{FAST}_ma"
    slow_ma_col = f"{SLOW}_ma"
    
    # Validate required columns exist in the dataframe
    required_indicator_cols = [
        fast_ma_col, slow_ma_col, 'RSI', 'Close', 'Volume', 'High', 'Low',
        'ATR', 'ADX', 'Volume_MA20', 'Open', 'volume_confirmed', 'weekly_uptrend'
    ]
    
    missing_cols = [col for col in required_indicator_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for signals: {missing_cols}. Returning default signals.")
        return signals_df
    
    # Extract weights, thresholds, and lookbacks from params
    weights = params.get('weights', DEFAULT_SIGNAL_PROCESSING_PARAMS['weights'])
    thresholds = params.get('thresholds', DEFAULT_SIGNAL_PROCESSING_PARAMS['thresholds'])
    current_ranking_lookback_window = params.get('ranking_lookback_window', DEFAULT_SIGNAL_PROCESSING_PARAMS['ranking_lookback_window'])
    current_momentum_volatility_lookback = params.get('momentum_volatility_lookback', DEFAULT_SIGNAL_PROCESSING_PARAMS['momentum_volatility_lookback'])

    # Extract NumPy arrays for performance
    close_np = df['Close'].values
    adx_np = df['ADX'].values
    fast_ma_np = df[fast_ma_col].values
    slow_ma_np = df[slow_ma_col].values
    vix_np = df['VIX'].values

    # Define signal conditions
    conditions = {
        # Primary trend conditions
        'trend_signal': (fast_ma_np > slow_ma_np),
        'trend_strength_ok': adx_np > adx_threshold,
        'vix_regime_permissive': vix_np < VIX_ENTRY_THRESHOLD
        }

    # ---- Rank Raw Components ----
    # Element 1 related (Trend Identification)
    df['price_roc'] = df['price_roc_raw'].rolling(
        window=current_ranking_lookback_window, min_periods=1).rank(pct=True).fillna(0.5)
    df['ma_dist'] = df['ma_dist_raw'].rolling(
        window=current_ranking_lookback_window, min_periods=1).rank(pct=True).fillna(0.5)
    
    # Element 2 related (Momentum Confirmation - RSI)
    df['rsi_ideal_zone_ranked'] = df['rsi_ideal_zone_raw'].rolling(
        window=current_ranking_lookback_window, min_periods=1).rank(pct=True).fillna(0.5)
    
    # Element 3 related (Trend Strength Filter - ADX)
    df['adx_slope'] = df['adx_slope_raw'].rolling(
        window=current_ranking_lookback_window, min_periods=1).rank(pct=True).fillna(0.5)
    
    # Element 4 related (Volume Confirmation)
    df['vol_accel'] = df['vol_accel_raw'].rolling(
        window=current_ranking_lookback_window, min_periods=1).rank(pct=True).fillna(0.5)

    # ---- Volatility Adjustment Component ----
    df['vol_adjustment_rank'] = df['atr_pct_raw'].rolling(
        current_momentum_volatility_lookback, min_periods=1).rank(pct=True).fillna(0.5) # Use param
    df['vol_adjustment'] = (1 - df['vol_adjustment_rank']).clip(0.5, 1.5)

    # Element 5 related (Market Sentiment - VIX)
    df['vix_factor_ranked'] = df['vix_factor_raw'].rolling(
        window=current_ranking_lookback_window, min_periods=1).rank(pct=True).fillna(0.5)
    

    # ---- Calculate Momentum Score ----
    raw_score = (
        weights['price_trend'] * (df['price_roc'] * 0.6 + df['ma_dist'] * 0.4) +   
        weights['rsi_zone'] * df['rsi_ideal_zone_ranked'] +                               
        weights['adx_slope'] * df['adx_slope'] +                        
        weights['vol_accel'] * df['vol_accel'] +
        weights['vix_factor'] * df['vix_factor_ranked']               
    ) 
    
    # Apply volatility adjustment
    momentum_score_values = ((raw_score * df['vol_adjustment']).clip(0, 1) * 100).values
    
    buy_signal = (
        (conditions['trend_signal']) &
        (conditions['trend_strength_ok']) &
        (conditions['vix_regime_permissive']) &
        (momentum_score_values > thresholds['buy_score']) 
    )
    
    exit_signal = (
        (momentum_score_values < thresholds['exit_score']) | (df['RSI'].values > 70)
    )
    
    immediate_exit = (
        (momentum_score_values < thresholds['immediate_exit_score']) | 
        (adx_np < adx_threshold/2) |
        (df['RSI'].values > 80)
    )
    
    # Assign all signals to the DataFrame
    signals_df['buy_signal'] = buy_signal
    signals_df['exit_signal'] = exit_signal
    signals_df['momentum_score'] = momentum_score_values
    signals_df['immediate_exit'] = immediate_exit

    return signals_df
# --------------------------------------------------------------------------------------------------------------------------
def process_signals(signals_df, i, previous_day_atr, previous_day_adx):
    """Extract signal processing logic to a separate function"""
    current_date = signals_df.index[i]
    buy_signal = signals_df['buy_signal'].iloc[i-1]
    exit_signal = signals_df['exit_signal'].iloc[i-1]
    immediate_exit = signals_df['immediate_exit'].iloc[i-1]
    momentum_score = signals_df['momentum_score'].iloc[i-1]
    
    return {
        'current_date': current_date,
        'buy_signal': buy_signal,
        'exit_signal': exit_signal,
        'immediate_exit': immediate_exit,
        'momentum_score': momentum_score,
        'previous_day_atr': previous_day_atr,
        'previous_day_adx': previous_day_adx,
    }
# --------------------------------------------------------------------------------------------------------------------------
def momentum(df_with_indicators, 
             long_risk=DEFAULT_LONG_RISK, 
             max_positions=MAX_OPEN_POSITIONS, 
             adx_threshold=ADX_THRESHOLD_DEFAULT, 
             max_position_duration=MAX_POSITION_DURATION, 
             params=None):

    if df_with_indicators.empty and df_with_indicators.index.empty:
        print("Warning: Empty dataframe (df_with_indicators) provided to momentum.")
        return [], {}, pd.Series(dtype='float64'), pd.Series(dtype='float64')
    
    # --- Determine current strategy and signal parameters ---
    current_signal_processing_params = {} # This will be the nested dict for signals()
    
    # These will be the actual parameters used in this run of momentum
    current_long_risk = long_risk
    current_max_positions = max_positions
    current_adx_threshold = adx_threshold
    current_max_position_duration = max_position_duration

    if params: # params is the flat dictionary from Optuna or other callers
        # Extract strategy parameters from the flat 'params' dict
        current_long_risk = params.get('long_risk', long_risk)
        current_max_positions = params.get('max_open_positions', max_positions) # Optuna uses 'max_open_positions'
        current_adx_threshold = params.get('adx_threshold', adx_threshold)
        current_max_position_duration = params.get('max_position_duration', max_position_duration)

        # Construct the nested current_signal_processing_params for the signals() function
        current_signal_processing_params['weights'] = {
            'price_trend': params.get('weight_price_trend', DEFAULT_SIGNAL_PROCESSING_PARAMS['weights']['price_trend']),
            'rsi_zone': params.get('weight_rsi_zone', DEFAULT_SIGNAL_PROCESSING_PARAMS['weights']['rsi_zone']),
            'adx_slope': params.get('weight_adx_slope', DEFAULT_SIGNAL_PROCESSING_PARAMS['weights']['adx_slope']),
            'vol_accel': params.get('weight_vol_accel', DEFAULT_SIGNAL_PROCESSING_PARAMS['weights']['vol_accel']),
            'vix_factor': params.get('weight_vix_factor', DEFAULT_SIGNAL_PROCESSING_PARAMS['weights']['vix_factor'])
        }
        current_signal_processing_params['thresholds'] = {
            'buy_score': params.get('threshold_buy_score', DEFAULT_SIGNAL_PROCESSING_PARAMS['thresholds']['buy_score']),
            'exit_score': params.get('threshold_exit_score', DEFAULT_SIGNAL_PROCESSING_PARAMS['thresholds']['exit_score']),
            'immediate_exit_score': params.get('threshold_immediate_exit_score', DEFAULT_SIGNAL_PROCESSING_PARAMS['thresholds']['immediate_exit_score'])
        }
        current_signal_processing_params['ranking_lookback_window'] = params.get('ranking_lookback_window_opt', DEFAULT_SIGNAL_PROCESSING_PARAMS['ranking_lookback_window'])
        current_signal_processing_params['momentum_volatility_lookback'] = params.get('momentum_volatility_lookback_opt', DEFAULT_SIGNAL_PROCESSING_PARAMS['momentum_volatility_lookback'])
    
    else: # params is None, use defaults for everything
        # Strategy parameters are already set to defaults above
        current_signal_processing_params = DEFAULT_SIGNAL_PROCESSING_PARAMS # Use the global default nested dict
    # --- End parameter setup ---

    # 1. Generate signals using the indicator-laden DataFrame
    signals_df = signals(df_with_indicators.copy(), adx_threshold=current_adx_threshold, params=current_signal_processing_params)

    # 2. Initialize Trade Managers and Tracking
    trade_manager = TradeManager(INITIAL_CAPITAL, current_max_positions) 
    equity_curve = pd.Series(INITIAL_CAPITAL, index=df_with_indicators.index)
    returns_series = pd.Series(0.0, index=df_with_indicators.index)

    # Pre-allocate numpy arrays for critical data for faster access
    prices = df_with_indicators['Open'].values
    atrs = df_with_indicators['ATR'].values
    adxs = df_with_indicators['ADX'].values

    # 3. Main Processing Loop
    for i in range(1, len(df_with_indicators)):
        transaction_price = prices[i]
        current_date = df_with_indicators.index[i]

        # Process signals for this step
        signal_data = process_signals(signals_df, i, atrs[i-1], adxs[i-1])

        # --- Exit Conditions (Priority Order) ---
        if trade_manager.position_count > 0:
            # 1. Check trailing stop first
            any_trailing_stop_hit = trade_manager.trailing_stops(
                transaction_price, current_date, signal_data['previous_day_atr'], signal_data['previous_day_adx']
            )
            if not any_trailing_stop_hit:
                if signal_data['immediate_exit']:
                    trade_manager.process_exits(current_date, transaction_price, direction_to_exit='Long', Trim=0.0, reason_override="Immediate Exit")
                elif signal_data['exit_signal']:
                    trim_amount = 0.05 if signal_data['momentum_score'] > 50 else 0.0
                    reason = "Partial Exit" if trim_amount > 0 else "Exit Signal"
                    trade_manager.process_exits(current_date, transaction_price, direction_to_exit='Long', Trim=trim_amount, reason_override=reason)
            
            if trade_manager.position_count > 0 and not any_trailing_stop_hit: # Re-check count
                trade_manager.position_health( # position_health now returns a dict, but its return value isn't directly used here to alter flow
                    transaction_price, signal_data['previous_day_atr'], current_date,  
                    signal_data['momentum_score'], current_max_position_duration # Use determined duration
                )
                
        # --- Entry Conditions ---
        if (signal_data['buy_signal'] and trade_manager.position_count < current_max_positions):
            entry_params = {
                'price': transaction_price * (1 + 0.001),
                'portfolio_value': trade_manager.portfolio_value,
                'risk': current_long_risk, # Use determined risk
                'atr': signal_data['previous_day_atr'],
                'adx': signal_data['previous_day_adx']
            }
            trade_manager.process_entry(current_date, entry_params, direction='Long')
        
        # Performance Tracking
        total_value = trade_manager.portfolio_value + trade_manager.unrealized_pnl(transaction_price)
        equity_curve.iloc[i] = max(total_value, 1e-9)  # Ensure positive for log

        # Calculate periodic log returns
        prev_equity_val = equity_curve.iloc[i-1]
        current_equity_val = equity_curve.iloc[i]
        if prev_equity_val > 1e-9: # Avoid division by zero or log of zero/negative
            returns_series.iloc[i] = np.log(current_equity_val / prev_equity_val)
        else:
            returns_series.iloc[i] = 0.0

    # Final portfolio value calculation
    final_close = df_with_indicators['Close'].iloc[-1] if not df_with_indicators.empty else INITIAL_CAPITAL
    final_unrealized_pnl = trade_manager.unrealized_pnl(final_close)
    final_equity_value = equity_curve.iloc[-1] if not equity_curve.empty else INITIAL_CAPITAL
    
    # Calculate statistics
    final_stats = {
        'Equity Final': final_equity_value,
        'Open Position Value': final_unrealized_pnl,
        'Total Portfolio Value': final_equity_value + final_unrealized_pnl
    }
    
    stats_dict = trade_statistics(
        equity_curve, 
        trade_manager.trade_log, 
        trade_manager.wins, 
        trade_manager.losses, 
    )
    stats_dict.update(final_stats)

    return trade_manager.trade_log, stats_dict, equity_curve, returns_series
# --------------------------------------------------------------------------------------------------------------------------
class TradeManager:
    def __init__(self, initial_capital, max_positions):
        self.portfolio_value = initial_capital
        self.max_positions = max_positions
        self.position_count = 0
        self.allocated_capital = 0
        self.position_counter = 0
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
            'lowest_close_since_entry': 'float32',
            'position_id': 'int32',
            'remaining_shares': 'int32'
        }
        
        # Initialize active trades DataFrame with proper dtypes
        self.active_trades = pd.DataFrame(columns=self.dtypes.keys()).astype(self.dtypes)
     # ----------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    def unrealized_pnl(self, current_price):
        if self.active_trades.empty: return 0.0
        pnl_values = (current_price - self.active_trades['entry_price']) * \
                 self.active_trades['remaining_shares'] * \
                 self.active_trades['multiplier']  # Assuming 'multiplier' column exists and is correct
        
        return pnl_values.sum()
    # ----------------------------------------------------------------------------------------------------------
    def position_health(self, current_price, current_atr, current_date, current_score, max_position_duration):
        """
        Assess position health and take appropriate exit actions based on analysis.
        Returns a combined dictionary with health metrics and total PnL from any health-based exits.
        """
        # Initialize return value
        total_pnl_from_health_exits = 0.0
        
        if self.active_trades.empty:
            # Return empty health metrics if no positions
            return {
                'profit_factor': 0.0, 
                'strength': 'none', 
                'position_duration': {}, 
                'take_profit_levels': {},
                'health_pnl': total_pnl_from_health_exits
            }
            
        # Cast inputs to proper types
        current_price_np = np.float32(current_price)
        current_atr_np = np.float32(current_atr)
        current_score_np = np.float32(current_score)

        # Calculate unrealized PnL for all active trades (vectorized)
        unrealized_pnls_series = (
            (current_price_np - self.active_trades['entry_price']) * 
            self.active_trades['remaining_shares'] * 
            self.active_trades['multiplier']
        )
        unrealized_pnls_sum = unrealized_pnls_series.sum()

        # Calculate total initial risk for all active trades (vectorized)
        # Assuming stop_loss is per share and fixed at entry for this calculation
        total_initial_risk_at_entry = (
            abs(self.active_trades['entry_price'] - self.active_trades['stop_loss'].astype(float)) * 
            self.active_trades['share_amount'] # Use original share_amount for initial total risk
        ).sum()
        
        strength_label = 'none'  # Default strength label
        atr_pct = current_atr_np / current_price_np if current_price_np != 0 else 0.0 
        adj_score  = current_score_np * (1.2 - 0.5 * atr_pct)  # Adjust score by ATR percentage
        if adj_score > 75:
            strength_label = 'hyper'
        elif adj_score > 60:
            strength_label = 'very_strong'
        elif adj_score > 45:
            strength_label = 'strong'
        elif adj_score > 30:
            strength_label = 'moderate'
        else:
            strength_label = 'weak'
        
        position_durations = {} # Keyed by active_trades DataFrame index
        take_profit_levels = {} # Keyed by active_trades DataFrame index
        
        # Create a list of indices to iterate over, as self.active_trades can change size during iteration
        indices_to_iterate = self.active_trades.index.tolist()

        for idx in indices_to_iterate:
            # Check if trade still exists in active_trades (it might have been removed by process_exits)
            if idx not in self.active_trades.index:
                continue
            
            trade = self.active_trades.loc[idx]

            if trade['direction'] == 'Long': # Assuming only long for now
                # Calculate duration
                duration = 0
                if isinstance(trade['entry_date'], pd.Timestamp) and isinstance(current_date, pd.Timestamp):
                    duration = (current_date - trade['entry_date']).days
                position_durations[idx] = duration

                # Skip if already exited (e.g. remaining_shares became 0 from a previous health check in this same call)
                if trade['remaining_shares'] <= 0:
                    continue

                # --- Dynamic Exit Logic ---
                # 1. Time Exit (only if momentum faded)
                if duration > max_position_duration:
                    if current_score_np > 50:
                        if idx in self.active_trades.index:
                            pnl = self.process_exits(
                                current_date, current_price_np, 
                                direction_to_exit='Long',
                                Trim=0.07,
                                reason_override="Partial Trim"
                            )
                            total_pnl_from_health_exits += pnl
                    else:
                    # Ensure position still exists before trying to exit
                        if idx in self.active_trades.index:
                            pnl = self.process_exits(
                                current_date, current_price_np, 
                                direction_to_exit='Long',
                                Trim=0.0, # Full exit
                                reason_override="Max Duration"
                            )
                            total_pnl_from_health_exits += pnl
                    continue # Move to next trade if this one was exited

                # Calculate risk per share for R-multiple calculation
                risk_per_share = abs(trade['entry_price'] - float(trade['stop_loss']))
                
                current_profit_per_share = (current_price_np - trade['entry_price'])
                r_multiple = 0.0
                if risk_per_share > 1e-9: # Avoid division by zero
                    r_multiple = current_profit_per_share / risk_per_share
                
                # Store take profit levels keyed by DataFrame index 'idx'
                # R-multiple targets (momentum-scaled)
                take_profit_levels[idx] = {
                    '1R_target': trade['entry_price'] + risk_per_share * (1 + current_score_np/100),
                    '2R_target': trade['entry_price'] + risk_per_share * (2 + current_score_np/50),
                    'current_r_multiple': r_multiple,
                    # Original R levels for reference if needed
                    'static_1R': trade['entry_price'] + risk_per_share * 1,
                    'static_2R': trade['entry_price'] + risk_per_share * 2,
                    'static_3R': trade['entry_price'] + risk_per_share * 3,
                }
        
        # --- Portfolio-Level Profit-Taking Rules (based on overall profit factor) ---
        # Calculate current total risk based on remaining shares and current stop losses
        current_total_risk_active = (
            abs(self.active_trades['entry_price'] - self.active_trades['stop_loss'].astype(float)) * 
            self.active_trades['remaining_shares']
        ).sum()

        profit_factor_overall = unrealized_pnls_sum / current_total_risk_active if current_total_risk_active > 0 else 0.0

        # Matrix-based trimming (momentum-dependent)
        PROFIT_RULES = {
            'hyper': {'pf_thresholds': [2.0, 3.0], 'trim_pcts': [0.1, 0.2]},
            'very_strong': {'pf_thresholds': [1.5, 2.5], 'trim_pcts': [0.15, 0.25]},
            'strong': {'pf_thresholds': [1.2, 2.0], 'trim_pcts': [0.2, 0.3]},
            'moderate': {'pf_thresholds': [1.0], 'trim_pcts': [0.3]},
            'weak': {'pf_thresholds': [0.8], 'trim_pcts': [0.5]}  # Aggressive exit
        }
        
        rules = PROFIT_RULES.get(strength_label, {})
        if not self.active_trades.empty: # Only apply if there are active trades
            for threshold_pf, trim_pct in zip(rules.get('pf_thresholds', []), 
                                        rules.get('trim_pcts', [])):
                if profit_factor_overall >= threshold_pf:
                    # This process_exits call might affect self.active_trades
                    # It will iterate through trades and apply trimming if applicable
                    pnl_from_trimming = self.process_exits(
                        current_date, current_price_np,
                        direction_to_exit='Long', # Assuming long only for now
                        Trim=trim_pct,
                        reason_override="Profit Take"
                    )
                    total_pnl_from_health_exits += pnl_from_trimming
                    break # Apply only one rule per call
            
        return {
            'profit_factor': profit_factor_overall, # Use overall profit factor
            'strength': strength_label, 
            'position_duration': position_durations, 
            'take_profit_levels': take_profit_levels,
            'health_pnl': total_pnl_from_health_exits
        }
    # ----------------------------------------------------------------------------------------------------------
    def trailing_stops(self, current_price, current_date, current_atr, adx_value):
        """
        Simplified trailing stop implementation using vectorized operations.
        """
        if self.active_trades.empty:
            return False

        # Vectorized operations for performance
        current_price_arr = np.float32(current_price)
        current_atr_arr = np.float32(current_atr)
        
        # Update highest prices since entry
        self.active_trades['highest_close_since_entry'] = np.maximum(
            self.active_trades['highest_close_since_entry'],
            current_price_arr
        )

        # --- 1. Dynamic ATR Multiplier Based on ADX ---
        if adx_value < 20:  # Weak trend: tight stops
            base_multiplier = 1.0  
        elif 20 <= adx_value <= 40:  # Normal trend
            base_multiplier = 2.0  
        else:  # Strong trend (ADX > 40): wider stops
            base_multiplier = 1.5  

        # --- 2. Profit-Based Multiplier Boosts ---
        profit_pct = ((self.active_trades['highest_close_since_entry'] - 
                    self.active_trades['entry_price']) / self.active_trades['entry_price'])
        
        # Tiered profit locking (adjust thresholds as needed)
        profit_factor = np.where(
            profit_pct > 0.10, 1.5,  # Lock in profits aggressively after +10%
            np.where(
                profit_pct > 0.05, 1.2,  # Moderate locking after +5%
                1.0  # Default
            )
        )
        
        # --- 3. Calculate Final Stops ---
        final_multiplier = base_multiplier * profit_factor
        new_stops = self.active_trades['highest_close_since_entry'] - (final_multiplier * current_atr)

        # --- 4. Apply Stop Rules ---
        # Rule 1: Never move stops backward
        new_stops = np.maximum(new_stops, self.active_trades['stop_loss'])
        
        # Rule 2: Never risk giving back >50% of unrealized gains
        unrealized_gains = current_price - self.active_trades['entry_price']
        new_stops = np.where(
            unrealized_gains > 0,
            np.maximum(new_stops, self.active_trades['entry_price'] + (unrealized_gains * 0.65)),
            new_stops
        )

        # Update stops
        self.active_trades['stop_loss'] = new_stops
        
        # Check for exits
        stops_hit = current_price_arr <= self.active_trades['stop_loss'].values
        
        if np.any(stops_hit):
            hit_indices = np.where(stops_hit)[0]
            total_pnl = 0.0

            for idx in hit_indices:
                if idx < len(self.active_trades):
                    trade_idx = self.active_trades.index[idx]
                    trade = self.active_trades.loc[trade_idx]

                    pnl = self.process_exits(
                    current_date, 
                    current_price, 
                    direction_to_exit='Long', 
                    Trim=0.0,  # Full exit
                    reason_override='Trailing Stop'
                    )
                    total_pnl += pnl
            return True
        return False
    # ----------------------------------------------------------------------------------------------------------
    def process_exits(self, current_date, current_price, direction_to_exit, Trim=0.0, reason_override=None): 

        if self.active_trades.empty:
            return 0.0

        # Convert inputs to proper types
        current_price = np.float32(current_price)*(1 - 0.001)
        total_pnl = 0.0
        
        # Initialize removal mask
        indices_to_remove = []
        
        # Get relevant trades (long only)
        long_trades = self.active_trades[self.active_trades['direction'] == 'Long']
        
        for idx, trade in long_trades.iterrows():
            current_shares = trade['remaining_shares']
            
            if current_shares <= 0:
                if idx not in indices_to_remove:
                    indices_to_remove.append(idx)
                continue

            # Priority 3: Signal-based Exits (Full or Partial)
            base_signal_exit_reason = 'Full Signal Exit'
            if Trim > 0.0:
                base_signal_exit_reason = 'Partial Signal Exit'
            
            # Use override if provided, otherwise use base
            final_exit_reason = reason_override if reason_override else base_signal_exit_reason
            if Trim > 0.0:
                # Partial exit
                shares_to_exit = int(current_shares * Trim)
                if shares_to_exit > 0:
                    pnl = self.exit_pnl(
                        trade, 
                        current_date, 
                        current_price,
                        shares_to_exit, 
                        final_exit_reason
                    )
                    total_pnl += pnl
                    self.portfolio_value += pnl
                    self.allocated_capital -= (current_price * shares_to_exit)
                    
                    remaining_shares = current_shares - shares_to_exit
                    if remaining_shares > 0:
                        self.active_trades.loc[idx, 'remaining_shares'] = remaining_shares
                    else:
                        indices_to_remove.append(idx)
                        self.position_count -= 1
            else:
                # Full exit
                pnl = self.exit_pnl(
                    trade, 
                    current_date, 
                    current_price,
                    current_shares, 
                    final_exit_reason
                )
                total_pnl += pnl
                self.portfolio_value += pnl
                self.allocated_capital -= (current_price * current_shares)
                self.position_count -= 1
                indices_to_remove.append(idx)
                continue
            
            # Priority 2: Take Profit Check
            if pd.notna(trade['take_profit']) and current_price >= trade['take_profit']:
                pnl = self.exit_pnl(
                    trade, 
                    current_date, 
                    current_price,
                    current_shares, 
                    'Take Profit'
                )
                total_pnl += pnl
                self.portfolio_value += pnl
                self.allocated_capital -= (current_price * current_shares)
                self.position_count -= 1
                indices_to_remove.append(idx)
                continue

            # Priority 1: Stop Loss Check
            if current_price <= trade['stop_loss']:
                pnl = self.exit_pnl(
                    trade, 
                    current_date, 
                    current_price,
                    current_shares, 
                    'Stop Loss'
                )
                total_pnl += pnl
                self.portfolio_value += pnl
                self.allocated_capital -= (current_price * current_shares)
                self.position_count -= 1
                indices_to_remove.append(idx)
                continue

        
        # Remove closed positions
        if indices_to_remove:
            self.active_trades = self.active_trades.drop(index=indices_to_remove).reset_index(drop=True)
            
        return total_pnl
    # ----------------------------------------------------------------------------------------------------------
    def process_entry(self, current_date, entry_params, direction=None):
        is_long = direction == 'Long'
        direction_mult = 1 if is_long else -1

        # --- 1. ADX-Based ATR Multiplier & Stop-Loss ---
        adx = entry_params['adx']
        atr = entry_params['atr']
        
        # Dynamic ATR Multiplier (adjust based on ADX)
        if adx < 20:  # Weak trend: tighter stops, smaller size
            atr_multiplier = 1.0  
            risk_pct = entry_params['risk'] * 0.5  # Halve risk in choppy markets
        elif 20 <= adx <= 40:  # Strong trend: default
            atr_multiplier = 2.5  
            risk_pct = entry_params['risk']
        else:  # Very strong trend: secure profits faster
            atr_multiplier = 1.5  
            risk_pct = entry_params['risk'] * 1.2  # Slightly increase risk
        
        stop_distance = atr * atr_multiplier
        initial_stop = entry_params['price'] - (stop_distance * direction_mult)

        # --- 2. Position Sizing ---
        risk_per_share = abs(entry_params['price'] - initial_stop)
        if risk_per_share < 1e-9:
            return False

        max_risk_amount = entry_params['portfolio_value'] * risk_pct  # Now ADX-adjusted
        shares_by_risk = int(max_risk_amount / risk_per_share)

        # Cap shares by available capital and exposure
        shares = shares_by_risk
        position_dollar_amount = shares * entry_params['price']
        actual_position_size = position_dollar_amount / entry_params['portfolio_value']

        # Exposure check (unchanged)
        max_total_exposure = 0.95
        current_exposure = 0.0
        if not self.active_trades.empty:
            current_prices = self.active_trades['entry_price'].astype(np.float32)
            remaining_shares = self.active_trades['remaining_shares'].astype(np.float32)
            current_exposure = (remaining_shares * current_prices).sum() / entry_params['portfolio_value']

        total_exposure = current_exposure + actual_position_size
        if total_exposure > max_total_exposure:
            available_exposure = max_total_exposure - current_exposure
            if available_exposure > 0:
                adjusted_shares = int((available_exposure * entry_params['portfolio_value']) / entry_params['price'])
                shares = min(shares, adjusted_shares)
                position_dollar_amount = shares * entry_params['price']
                actual_position_size = position_dollar_amount / entry_params['portfolio_value']
            else:
                return False

        # --- 3. Take-Profit Calculation (ADX-Boosted) ---
        base_profit_distance = 3.0 * atr  # Default 3:1 reward:risk
        if adx > 40:  # Strong trend: wider profit target
            profit_multiplier = 1 + (adx / 100)  # Up to 1.4x
        else:
            profit_multiplier = 1.0

        take_profit = entry_params['price'] + (base_profit_distance * profit_multiplier * direction_mult)

        # --- 4. Final Checks ---
        available_capital = self.portfolio_value - self.allocated_capital
        if position_dollar_amount > available_capital or shares <= 0:
            return False

        commission = self.calculate_commission(shares, entry_params['price'])
        min_position_value = entry_params['portfolio_value'] * 0.001
        if position_dollar_amount < min_position_value:
            return False

        self.position_count += 1

        # 5. Create Trade
        new_trade = pd.DataFrame([{
            'entry_date': current_date,
            'direction': direction,
            'multiplier': direction_mult,
            'entry_price': entry_params['price'],
            'stop_loss': initial_stop,
            'take_profit': take_profit,
            'position_size': actual_position_size,
            'share_amount': shares,
            'commission': commission,
            'highest_close_since_entry': entry_params['price'] if is_long else np.nan,
            'lowest_close_since_entry': entry_params['price'] if not is_long else np.nan,
            'position_id': self.position_counter,  # Add position ID
            'remaining_shares': shares  # Track initial shares
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
        self.portfolio_value -= commission
        self.position_counter += 1
        return True
    # ----------------------------------------------------------------------------------------------------------    
    def calculate_commission(self, shares, price):
        if COMMISION:
            # Fixed commission model
            fixed_fee = 5.00  # $5 per trade
            
            # Per-share commission model
            per_share_fee = 0.005 * shares  # 0.5 cents per share
            
            # Percentage-based model
            percentage_fee = shares * price * 0.0005  # 0.1% of trade value
            
            # Tiered model example
            if shares * price < 5000:
                tiered_fee = 5.00
            elif shares * price < 10000:
                tiered_fee = 7.50
            else:
                tiered_fee = 10.00
            
            # Choose your model
            commission = percentage_fee  # or per_share_fee, percentage_fee, tiered_fee
            
            # Add minimum commission if needed
            return max(1.0, commission)  # Minimum $1.00 commission
        else:
            return 0.0
    # ----------------------------------------------------------------------------------------------------------
    def exit_pnl(self, trade_series, exit_date, exit_price, shares_to_exit, reason):
        
        entry_price = trade_series['entry_price']
        entry_date = trade_series['entry_date']
        trade_direction = trade_series['direction']
        position_id = trade_series['position_id']
        original_shares = trade_series['share_amount']
        entry_commission = trade_series['commission']
        exit_price = exit_price

        # Calculate if this is a complete exit
        is_complete_exit = (shares_to_exit >= trade_series['remaining_shares'])

        gross_pnl = 0
        if trade_direction == 'Long':
            gross_pnl = (exit_price - entry_price) * shares_to_exit
        else:  # Short
            gross_pnl = (entry_price - exit_price) * shares_to_exit

        duration = 0
        # Ensure entry_date is a Timestamp if it's not already
        if not isinstance(entry_date, pd.Timestamp):
            entry_date = pd.Timestamp(entry_date)
        if pd.notnull(entry_date) and pd.notnull(exit_date):
            if not isinstance(exit_date, pd.Timestamp): # Ensure exit_date is also Timestamp
                exit_date = pd.Timestamp(exit_date)
            duration = (exit_date - entry_date).days
        
        self.lengths.append(duration)

        exit_commission = self.calculate_commission(shares_to_exit, exit_price)

        pnl_net = gross_pnl - exit_commission

        if pnl_net > 0:
            self.wins.append(pnl_net)
        else:
            self.losses.append(pnl_net)

        self.trade_log.append({
            'Entry Date': entry_date,
            'Exit Date': exit_date,
            'Direction': trade_direction,
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Shares': shares_to_exit,
            'Original Shares': original_shares,
            'PnL': pnl_net, 
            'Gross PnL': gross_pnl, 
            'Entry Commission Initial': entry_commission, 
            'Exit Commission Current': exit_commission, 
            'Duration': duration,
            'Exit Reason': reason,
            'Position ID': position_id,
            'Is Complete Exit': is_complete_exit,
            'Remaining Shares': trade_series['remaining_shares'] - shares_to_exit if not is_complete_exit else 0
        })
        return pnl_net
    # ----------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
@jit(nopython=True)
def risk_metrics(returns_array, risk_free_daily):
    """Numba-optimized calculation of risk metrics with numerical safeguards"""
    if len(returns_array) <= 1:
        return 0.0, 0.0
    
    unique_values = np.unique(returns_array)
    if len(unique_values) <= 1:
        return 0.0, 0.0
    
    excess_returns = returns_array - risk_free_daily
    returns_mean = np.mean(excess_returns)
    returns_std = np.std(excess_returns)
    
    # Sharpe Ratio with safeguards
    # Add minimum threshold for standard deviation to prevent explosion
    min_std_threshold = 1e-8  # Adjust based on your typical return values
    
    if returns_std > min_std_threshold:
        sharpe = (returns_mean / returns_std) * np.sqrt(252)
        # Bound extreme values
    else:
        # Near-zero std scenario - use sign of mean to determine direction
        sharpe = 0.0 if returns_mean == 0 else (np.sign(returns_mean) * 5.0)
    
    # Sortino Ratio with similar safeguards  
    downside_returns = returns_array[returns_array < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
    
    if downside_std > min_std_threshold:
        sortino = ((np.mean(returns_array) - risk_free_daily) / downside_std) * np.sqrt(252)
    else:
        sortino = 0.0 if returns_mean == 0 else (np.sign(returns_mean) * 5.0)
    
    return sharpe, sortino
# --------------------------------------------------------------------------------------------------------------------------
def trade_statistics(equity, trade_log, wins, losses, risk_free_rate=RISK_FREE_RATE_ANNUAL):
    """Vectorized trade statistics calculation using NumPy and Numba"""
    
    # Convert lists to NumPy arrays
    wins_array = np.array(wins, dtype=float) if wins else np.array([0.0], dtype=float)
    losses_array = np.array(losses, dtype=float) if losses else np.array([0.0], dtype=float)
    
    # Basic trade statistics (vectorized)
    total_trades = len(trade_log)
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0.0
    
    # Profit metrics (vectorized)
    gross_profit = np.sum(wins_array)
    gross_loss = np.sum(losses_array) # Note: losses_array contains non-positive PnL values
    net_profit = gross_profit + gross_loss # gross_loss is typically negative or zero
    
    # Portfolio metrics (vectorized)
    initial_capital = equity.iloc[0]
    final_capital = equity.iloc[-1]
    net_profit_pct = ((final_capital / initial_capital) - 1) * 100 if initial_capital > 0 else 0.0
    
    # Risk metrics (vectorized)
    if gross_loss == 0:
        # If there are no losses (or only zero-value losses)
        profit_factor = np.inf if gross_profit > 0 else 1.0 
        # If gross_profit is also 0 (e.g., no trades, or all trades PnL=0), PF is 1.0 (neutral)
    else:
        # gross_loss is negative, so abs() is important if not already handled by convention
        profit_factor = abs(gross_profit / gross_loss) 
    
    # Expectancy calculation (vectorized)
    avg_win = np.mean(wins_array) if len(wins) > 0 else 0.0
    avg_loss = np.mean(losses_array) if len(losses) > 0 else 0.0 # avg_loss will be <= 0
    win_prob = win_rate / 100
    expectancy = (win_prob * avg_win) + ((1 - win_prob) * avg_loss) # avg_loss is non-positive
    expectancy_pct = (expectancy / initial_capital) * 100 if initial_capital > 0 else 0.0

    hit_count = 0
    # Hit Rate calculation
    if trade_log:
        for trade in trade_log:
            if 'Exit Reason' in trade:
                if trade['Exit Reason'] in ['Take Profit', 'Trailing Stop', 'Max Duration', 'Profit Take']:
                    hit_count += 1
    
    hit_rate = (hit_count / total_trades * 100) if total_trades > 0 else 0.0

    exit_reason_counts = {}
    if trade_log:
        for trade in trade_log:
            reason = trade.get('Exit Reason', 'Unknown') # Use .get() for safety
            exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1

    # Average Win/Loss Ratio
    if avg_loss == 0:
        # If average loss is zero (no losing trades or all losses were PnL=0)
        avg_win_loss_ratio = np.inf if avg_win > 0 else 1.0
        # If avg_win is also 0, ratio is 1.0 (neutral)
    else:
        # avg_loss is non-positive. abs() ensures positive ratio.
        avg_win_loss_ratio = abs(avg_win / avg_loss) 
    
    # Drawdown calculation (vectorized)
    equity_values = equity.values
    running_max = np.maximum.accumulate(equity_values)
    drawdowns = ((equity_values - running_max) / running_max) * 100
    # Handle cases where running_max could be zero if initial capital is zero and first trades are losses
    drawdowns = np.nan_to_num(drawdowns, nan=0.0, posinf=0.0, neginf=0.0)
    max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    # Time-based metrics
    if len(equity.index) > 1:
        days = (equity.index[-1] - equity.index[0]).days
        years = days / 365.25
        annualized_return = ((final_capital / initial_capital) ** (1/years) - 1) * 100 if years > 0 and initial_capital > 0 else 0.0
    else:
        days = 0
        years = 0
        annualized_return = 0.0

    # Calculate risk ratios using Numba-optimized function
    if len(equity_values) > 1:
        daily_returns = np.diff(np.log(np.maximum(equity_values, 1e-9))) # Add epsilon to avoid log(0)
        daily_rf_rate = risk_free_rate / 252.0 # Ensure float division
        sharpe_ratio, sortino_ratio = risk_metrics(daily_returns, daily_rf_rate)
    else:
        sharpe_ratio, sortino_ratio = 0.0, 0.0

    # Calculate Calmar Ratio
    if max_drawdown > 0:
        calmar_ratio = annualized_return / max_drawdown
    elif annualized_return > 0: # Max drawdown is 0, but returns are positive
        calmar_ratio = np.inf
    else: # Max drawdown is 0 and returns are not positive
        calmar_ratio = 0.0
    
    return {
        'Total Trades': total_trades,
        'Win Rate': win_rate,
        'Hit Rate': hit_rate,
        'Return (%)': net_profit_pct,
        'Net Profit': net_profit,
        'Profit Factor': profit_factor,
        'Expectancy (%)': expectancy_pct,
        'Max Drawdown (%)': max_drawdown,
        'Annualized Return (%)': annualized_return,
        'Sharpe Ratio':sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Avg Win/Loss Ratio': avg_win_loss_ratio,
        'Exit Reason Counts': exit_reason_counts
    }

#==========================================================================================================================
#================== OPTIMIZING STRATEGY ===================================================================================
#==========================================================================================================================
def objectives(trial, base_df):
    """Objective function for Optuna optimization that directly tests parameters"""
    # Define bad metrics template
    bad_metrics_template = []
    for metric_name in OPTIMIZATION_DIRECTIONS: # Ensure order matches OPTIMIZATION_DIRECTIONS
        if OPTIMIZATION_DIRECTIONS[metric_name] == 'maximize':
            bad_metrics_template.append(-np.inf)
        else: # minimize
            bad_metrics_template.append(np.inf)

    # Define parameter ranges for optimization
    optuna_params_dict = {
        # Basic parameters
        'long_risk': trial.suggest_float('long_risk', 0.02, 0.10, step=0.01),
        
        # Technical parameters
        'max_open_positions': trial.suggest_int('max_open_positions', 2, 30, step=1), # Changed step
        'adx_threshold': trial.suggest_float('adx_threshold', 20.0, 35.0, step=1.0),
        'max_position_duration': trial.suggest_int('max_position_duration', 5, 30, step=1), # Changed step

        'weight_price_trend': trial.suggest_float('weight_price_trend', 0.1, 0.4, step=0.05),
        'weight_rsi_zone': trial.suggest_float('weight_rsi_zone', 0.1, 0.4, step=0.05),
        'weight_adx_slope': trial.suggest_float('weight_adx_slope', 0.1, 0.4, step=0.05),
        'weight_vol_accel': trial.suggest_float('weight_vol_accel', 0.1, 0.4, step=0.05),
        'weight_vix_factor': trial.suggest_float('weight_vix_factor', 0.1, 0.4, step=0.05),
        
        'threshold_buy_score': trial.suggest_int('threshold_buy_score', 45, 70, step=1), # Changed step
        'threshold_exit_score': trial.suggest_int('threshold_exit_score', 25, 45, step=1), # Changed step
        'threshold_immediate_exit_score': trial.suggest_int('threshold_immediate_exit_score', 15, 30, step=1), # Changed step

        'ranking_lookback_window_opt': trial.suggest_int('ranking_lookback_window_opt', 20, 120, step=10), # Changed step
        'momentum_volatility_lookback_opt': trial.suggest_int('momentum_volatility_lookback_opt', 10, 60, step=5) # Changed step
    }
    
    # For debugging  
    trial_num = trial.number

    try:
        # Direct evaluation
        trade_log, stats, equity_curve, returns_series = momentum(
            base_df.copy(), # Pass a copy
            params=optuna_params_dict 
        )
        
        # Ensure stats dictionary is not None and contains all required keys
        if not stats:
            return bad_metrics_template

        # Process metrics according to OPTIMIZATION_DIRECTIONS keys
        metrics_for_optuna = []
        all_metrics_present = True
        for key in OPTIMIZATION_DIRECTIONS.keys():
            metric_value = np.nan
            if key == 'profit_factor':
                metric_value = stats.get('Profit Factor', -np.inf)
            elif key == 'avg_win_loss_ratio':
                metric_value = stats.get('Avg Win/Loss Ratio', -np.inf)
            elif key == 'expectancy':
                metric_value = stats.get('Expectancy (%)', -np.inf)
            elif key == 'max_drawdown':
                # Optuna minimizes, so we provide the positive drawdown value.
                # trade_statistics returns Max Drawdown (%) as a positive value.
                metric_value = stats.get('Max Drawdown (%)', np.inf) 
            
            if np.isnan(metric_value) or (key != 'max_drawdown' and np.isinf(metric_value) and metric_value < 0) or \
               (key == 'max_drawdown' and np.isinf(metric_value) and metric_value > 0): # Check for bad initial values
                all_metrics_present = False
                break
            metrics_for_optuna.append(metric_value)
            trial.set_user_attr(key, metric_value) # Set user attribute for each optimized metric

        if not all_metrics_present:
            return bad_metrics_template
            
        # Store additional non-optimized attributes in trial for later analysis
        trial.set_user_attr('num_trades', len(trade_log) if trade_log else 0)
        trial.set_user_attr('sharpe_ratio', stats.get('Sharpe Ratio', np.nan)) # Store Sharpe for reference
        trial.set_user_attr('return_pct', stats.get('Return (%)', np.nan)) # Store Return for reference
        
        avg_duration_val = np.nan
        if trade_log:
            durations = [t['Duration'] for t in trade_log if t.get('Duration') is not None]
            if durations:
                avg_duration_val = np.mean(durations)
        trial.set_user_attr('avg_trade_duration', avg_duration_val)
        
        total_pnl_val = np.nan
        if trade_log:
            pnls = [t['PnL'] for t in trade_log if t.get('PnL') is not None]
            if pnls:
                total_pnl_val = sum(pnls)
        trial.set_user_attr('total_pnl', total_pnl_val)
            
        return metrics_for_optuna

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Error in parameter evaluation for trial {trial_num}: {e}")
        traceback.print_exc()
        return bad_metrics_template
# -------------------------------------------------------------------------------------------------------------------------
def optimize(prepared_data):
    """Optimizing function to find the best parameters for the strategy"""
    # Define optimization parameters
    target_metrics = list(OPTIMIZATION_DIRECTIONS.keys())
    opt_directions = [OPTIMIZATION_DIRECTIONS[metric] for metric in target_metrics]
    n_trials=TRIALS
    timeout=1200

    data = prepared_data.copy()
    if data.empty:
        print("Warning: Empty dataframe provided to optimize.")
        return None
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create Optuna study with pruning
    study = optuna.create_study(
        directions=opt_directions,  # Direction for each metric
        study_name=f"strategy_optimization",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=5, max_resource=n_trials, reduction_factor=3),
        sampler=optuna.samplers.NSGAIIISampler(seed=42, population_size=60)  # Use NSGA-III for multi-objective
    )

    # Define the objective function
    objective_func = partial(objectives, base_df=data)
        
    # Run optimization with progress bar
    try:
        study.optimize(objective_func, n_trials=n_trials, timeout=timeout, 
                n_jobs=max(1, mp.cpu_count() - 1))
    except Exception as e:
        print(f"Optimization error: {e}")
    
    # Get Pareto front solutions
    all_trials = study.trials
    # Filter out failed trials and sort by custom criteria
    filtered_trials_with_data = [] # Renamed to avoid confusion

    for trial in all_trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.values is not None:
            if any(np.isinf(v) for v in trial.values): # Skip trials with inf values
                continue

            num_trades = trial.user_attrs.get('num_trades', 0)
            # Skip trials with zero trades (already implemented based on previous request)
            if num_trades == 0:
                continue
            
            metrics_dict = {}
            valid_trial_values = True
            for i, key in enumerate(target_metrics): # target_metrics is list(OPTIMIZATION_DIRECTIONS.keys())
                if i < len(trial.values):
                    metrics_dict[key] = trial.values[i]
                else: # Should not happen if Optuna runs correctly
                    metrics_dict[key] = -np.inf if OPTIMIZATION_DIRECTIONS[key] == 'maximize' else np.inf
                    valid_trial_values = False # Mark as invalid if values are missing
            
            if not valid_trial_values:
                continue

            combined_score = 0
            for key in target_metrics:
                weight = OBJECTIVE_WEIGHTS.get(key, 0)
                value = metrics_dict.get(key, 0)
                
                if key == 'max_drawdown':
                    combined_score -= weight * abs(value) 
                else: 
                    if key == 'profit_factor':
                        value = min(value, 100) 
                    combined_score += weight * value
            
            lower_bound_with_tolerance = TARGET_SCORE - SCORE_TOLERANCE
            upper_bound_with_tolerance = TARGET_SCORE + SCORE_TOLERANCE

            if not (lower_bound_with_tolerance <= combined_score <= upper_bound_with_tolerance):
                # If the score is not within the target range (inclusive of tolerance), skip this trial
                continue

            # Filter: Skip trials with less than 25 trades
            if num_trades < 25:
                continue

            # Store trial, num_trades, and combined_score (if needed)
            filtered_trials_with_data.append({'trial': trial, 'num_trades': num_trades, 'combined_score': combined_score})

    # Sort by number of trades in descending order, then by combined_score in descending order
    filtered_trials_with_data.sort(key=lambda x: (x['num_trades'], x['combined_score']), reverse=True)
    
    # Extract just the trial objects for the pareto_front list
    pareto_front = [item['trial'] for item in filtered_trials_with_data]
    
    if not pareto_front:
        return []
        
    return pareto_front[:15]

#==========================================================================================================================
#================== VIEW STRATEGY =========================================================================================
#==========================================================================================================================
def visualize(pareto_front, base_df):
    if not pareto_front:
        print("No Pareto front trials to visualize.")
        return

    target_metrics_keys = list(OPTIMIZATION_DIRECTIONS.keys()) # Get the order of objectives
    # OBJECTIVE_WEIGHTS is a global dictionary, accessible here

    while True:
        trial_metrics_display = []
        for i, trial in enumerate(pareto_front, 1):
            metrics_row = {'Trial': i}
            current_trial_combined_score = 0.0 # For combined score calculation
            
            if trial.values: # Ensure trial.values is not None
                for j, key in enumerate(target_metrics_keys):
                    display_name = key.replace('_', ' ').title()
                    value = trial.values[j]
                    
                    # For combined score calculation
                    weight = OBJECTIVE_WEIGHTS.get(key, 0)
                    
                    if key == 'max_drawdown':
                        display_name = 'MaxDD(%)'
                        metrics_row[display_name] = f"{abs(value):.1f}"
                        current_trial_combined_score -= weight * abs(value)
                    elif key == 'avg_win_loss_ratio':
                        display_name = 'AvgWinL(%)'
                        metrics_row[display_name] = f"{value:.1f}"
                        current_trial_combined_score += weight * value
                    elif key == 'profit_factor':
                        metrics_row[display_name] = f"{value:.2f}"
                        value_for_score = min(value, 100) # Cap profit factor for score
                        current_trial_combined_score += weight * value_for_score
                    else: # For other metrics like 'expectancy'
                        metrics_row[display_name] = f"{value:.2f}"
                        current_trial_combined_score += weight * value
                metrics_row['Combined Score'] = f"{current_trial_combined_score:.2f}"
            else: # trial.values is None
                for key in target_metrics_keys:
                    display_name = key.replace('_', ' ').title()
                    if key == 'max_drawdown': display_name = 'MaxDD(%)'
                    if key == 'avg_win_loss_ratio': display_name = 'AvgWinL(%)'
                    metrics_row[display_name] = "N/A"
                metrics_row['Combined Score'] = "N/A"

            metrics_row['Trades'] = trial.user_attrs.get('num_trades', 0)
            trial_metrics_display.append(metrics_row)

        print("\n=== Optimization Results (Pareto Front) ===")
        if not trial_metrics_display:
            print("No trial metrics to display.")
        else:
            # tabulate will use the keys from the first dictionary in trial_metrics_display as headers
            print(tabulate(
                trial_metrics_display,
                headers='keys', 
                tablefmt='grid',
                floatfmt='.2f'
            ))

        try:
            choice = input("\nEnter trial number to test (or 'exit' to quit): ").strip().lower()
            
            if choice == 'exit':
                break
            
            trial_num_input = int(choice) - 1
            if 0 <= trial_num_input < len(pareto_front):
                selected_trial = pareto_front[trial_num_input]
                
                print(f"\nTesting Trial {trial_num_input + 1} Parameters: {selected_trial.params}")
                test(
                    base_df.copy(), # Pass a copy of base_df
                    params_to_test=selected_trial.params # Pass the flat dictionary
                )
                input("\nPress Enter to return to trial selection...")
            else:
                print(f"Invalid trial number. Please select 1-{len(pareto_front)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'exit'.")
        except Exception as e:
            print(f"Error during visualization or test run: {e}")
            traceback.print_exc()
    return None
# -------------------------------------------------------------------------------------------------------------------------
def test(df_input, params_to_test=None): 
    
    df = df_input.copy()
    
    # These will be used for display, extracted from params_to_test or defaults
    long_risk_disp = DEFAULT_LONG_RISK
    max_positions_disp = MAX_OPEN_POSITIONS
    adx_thresh_disp = ADX_THRESHOLD_DEFAULT
    max_duration_disp = MAX_POSITION_DURATION

    if params_to_test:
        # momentum will use these. It extracts individual strategy params 
        # and also builds the nested signal_params.
        trade_log, stats, equity_curve, returns_series = momentum(
            df, 
            params=params_to_test # Pass the flat dictionary
        )
        # For display in test summary, extract the main strategy params from params_to_test
        long_risk_disp = params_to_test.get('long_risk', DEFAULT_LONG_RISK)
        max_positions_disp = params_to_test.get('max_open_positions', MAX_OPEN_POSITIONS)
        adx_thresh_disp = params_to_test.get('adx_threshold', ADX_THRESHOLD_DEFAULT)
        max_duration_disp = params_to_test.get('max_position_duration', MAX_POSITION_DURATION)
    else:
        # If no params_to_test, momentum uses its internal defaults (which source from global defaults)
        trade_log, stats, equity_curve, returns_series = momentum(df)
        # Display parameters remain as defaults
    # Add asset returns to dataframe for comparison
    df.loc[:, 'Asset_Returns'] = df['Close'].pct_change().fillna(0).cumsum()
    
    # Convert equity curve to returns for comparison
    df.loc[:, 'Strategy_Returns'] = returns_series.cumsum()

    first_close = df['Close'].iloc[0]
    last_close = df['Close'].iloc[-1]
    buy_hold_return = ((last_close / first_close) - 1) * 100

    peak_equity = equity_curve.max()
    exposure_time = (df.index[-1] - df.index[0]).days

    total_entry_commission = 0
    total_exit_commission = 0
    # Extract trade metrics from trade log
    if trade_log:
        best_trade_pct = max([t['PnL'] for t in trade_log if t['PnL'] is not None], default=0) / equity_curve.iloc[0] * 100 if equity_curve.iloc[0] !=0 else 0
        worst_trade_pct = min([t['PnL'] for t in trade_log if t['PnL'] is not None], default=0) / equity_curve.iloc[0] * 100 if equity_curve.iloc[0] !=0 else 0
        avg_trade_pnl_values = [t['PnL'] for t in trade_log if t['PnL'] is not None]
        avg_trade_pct = (sum(avg_trade_pnl_values) / len(avg_trade_pnl_values) / equity_curve.iloc[0] * 100) if avg_trade_pnl_values and equity_curve.iloc[0] !=0 else 0

        durations = [t['Duration'] for t in trade_log if t['Duration'] is not None]
        max_duration = max(durations) if durations else 0
        avg_duration = sum(durations) / len(durations) if durations else 0
        # Calculate total commissions
        for t in trade_log:
            total_entry_commission += t.get('Entry Commission Initial', 0) # .get for safety if key missing
            total_exit_commission += t.get('Exit Commission Current', 0)   # .get for safety
    else:
        best_trade_pct = worst_trade_pct = avg_trade_pct = max_duration = avg_duration = 0
        total_entry_commission = 0
        total_exit_commission = 0

    total_commission_paid = total_entry_commission + total_exit_commission

    # Display results
    print(f"\n=== STRATEGY SUMMARY ===") 
    print(f"Long Risk: {long_risk_disp*100:.1f}%") # Use _disp variables
    print(f"Max Open Positions: {max_positions_disp} | ADX Threshold: {adx_thresh_disp:.1f} | Max Duration: {max_duration_disp} days")
    print(f"Weights - Price Trend: {params_to_test.get('weight_price_trend', 0.0):.2f}, RSI Zone: {params_to_test.get('weight_rsi_zone', 0.0):.2f}, ADX Slope: {params_to_test.get('weight_adx_slope', 0.0):.2f}, Volatility Acceleration: {params_to_test.get('weight_vol_accel', 0.0):.2f}, VIX Factor: {params_to_test.get('weight_vix_factor', 0.0):.2f}")
    print(f"Thresholds - Buy Score: {params_to_test.get('threshold_buy_score', 0)}, Exit Score: {params_to_test.get('threshold_exit_score', 0)}, Immediate Exit Score: {params_to_test.get('threshold_immediate_exit_score', 0)}")
    print(f"Lookback Window: {params_to_test.get('ranking_lookback_window_opt', 0)} days, Momentum/Volatility Lookback: {params_to_test.get('momentum_volatility_lookback_opt', 0)} days")
    
    metrics = [
        ["Start", f"{df.index[0].strftime('%Y-%m-%d')}"],
        ["End", f"{df.index[-1].strftime('%Y-%m-%d')}"],
        ["Duration [days]", f"{exposure_time}"],
        ["Starting Capital [$]", f"{equity_curve.iloc[0]:,.2f}"],
        ["Ending Cash [$]", f"{stats['Equity Final']:,.2f}"],
        ["Open Position Value [$]", f"{stats['Open Position Value']:,.2f}"],
        ["Total Portfolio Value [$]", f"{stats['Total Portfolio Value']:,.2f}"],
        ["Equity Peak [$]", f"{peak_equity:,.2f}"],
        ["Return [%]", f"{stats['Return (%)']:.2f}"],
        ["Buy & Hold Return [%]", f"{buy_hold_return:.2f}"],
        ["Annual Return [%]", f"{stats['Annualized Return (%)']:.2f}"],
        ["Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}"],
        ["Sortino Ratio", f"{stats['Sortino Ratio']:.2f}"],
        ["Calmar Ratio", f"{stats['Calmar Ratio']:.2f}"],
        ["Avg. Win/Loss Ratio", f"{stats['Avg Win/Loss Ratio']:.2f}"],
        ["Max. Drawdown [%]", f"{stats['Max Drawdown (%)']:.2f}"],
    ]
    print(tabulate(metrics, tablefmt="simple", colalign=("left", "right")))

    if 'Exit Reason Counts' in stats and stats['Exit Reason Counts']:
        print(f"\n=== EXIT REASON SUMMARY ===")
        exit_reasons_data = []
        total_exits_for_percentage = sum(stats['Exit Reason Counts'].values())
        
        # Create dictionaries to track PnL and wins by exit reason
        pnl_by_reason = {}
        wins_by_reason = {}
        total_by_reason = {}
        
        # Calculate PnL and win counts by exit reason
        if trade_log:
            for trade in trade_log:
                reason = trade.get('Exit Reason', 'Unknown') 
                pnl = trade.get('PnL', 0.0)
                is_win = pnl > 0
                
                pnl_by_reason[reason] = pnl_by_reason.get(reason, 0.0) + pnl
                wins_by_reason[reason] = wins_by_reason.get(reason, 0) + (1 if is_win else 0)
                total_by_reason[reason] = total_by_reason.get(reason, 0) + 1
        
        # Sort by count for better readability
        sorted_exit_reasons = sorted(stats['Exit Reason Counts'].items(), key=lambda item: item[1], reverse=True)

        for reason, count in sorted_exit_reasons:
            percentage = (count / total_exits_for_percentage * 100) if total_exits_for_percentage > 0 else 0
            pnl = pnl_by_reason.get(reason, 0.0)
            
            # Calculate win percentage for this exit reason
            win_pct = (wins_by_reason.get(reason, 0) / total_by_reason.get(reason, 1)) * 100
            
            exit_reasons_data.append([
                reason, 
                count, 
                f"{percentage:.2f}%", 
                f"${pnl:.2f}",
                f"{win_pct:.1f}%"
            ])
        
        if exit_reasons_data:
            print(tabulate(exit_reasons_data, 
                        headers=["Exit Reason", "Count", "Percentage", "Total PnL", "Win %"], 
                        tablefmt="simple", 
                        colalign=("left", "right", "right", "right", "right")))
        else:
            print("No exit reason data to display.")
    
    print(f"\n=== TRADE SUMMARY ===") 
    trade_metrics = [
        ["Total Trades", f"{stats['Total Trades']:.0f}"], 
        ["Win Rate [%]", f"{stats['Win Rate']:.2f}"],
        ["Hit Rate [%]", f"{stats['Hit Rate']:.2f}"],
        ["Best Trade [%]", f"{best_trade_pct:.2f}"],
        ["Worst Trade [%]", f"{worst_trade_pct:.2f}"],
        ["Avg. Trade [%]", f"{avg_trade_pct:.2f}"],
        ["Max. Trade Duration [days]", f"{max_duration}"],
        ["Avg. Trade Duration [days]", f"{avg_duration:.1f}"],
        ["Profit Factor", f"{stats['Profit Factor']:.2f}"],
        ["Expectancy [%]", f"{stats['Expectancy (%)']:.2f}"],
        ["Total PnL [$]", f"{stats.get('Net Profit', 0.0):,.2f}"],
        ["Total Commission Paid [$]", f"{total_commission_paid:,.2f}"]
    ]
    print(tabulate(trade_metrics, tablefmt="simple", colalign=("left", "right")))
     
    return None

#==========================================================================================================================
#================== STRATEGY SIGNIFICANCE TESTING =========================================================================
#==========================================================================================================================
def determine_optimal_block_length(series, max_lag=50, default_block_size=BLOCK_SIZE):
    """
    Determines an optimal block length for stationary bootstrap using ACF decay.
    The block length is chosen as the first lag where the ACF is no longer
    statistically significant (i.e., its confidence interval contains zero).
    """
    if not isinstance(series, pd.Series):
        print(f"Warning: Input to determine_optimal_block_length is not a Series. Type: {type(series)}. Using default: {default_block_size}")
        return default_block_size
    if series.empty or len(series) < max_lag + 1 or series.var() == 0: # Added variance check for constant series
        print(f"Warning: Series too short, empty, or constant for ACF-based block length. Using default: {default_block_size}")
        return default_block_size

    try:
        # Calculate ACF and confidence intervals
        # acf_values is a tuple: (acf_array, confint_array)
        acf_result = acf(series, nlags=max_lag, fft=True, alpha=0.05) 
        
        actual_acf_values = acf_result[0]
        confint = acf_result[1]

        # We ignore lag 0 (ACF is always 1)
        # Find the first lag k (from 1 to max_lag) where the CI for ACF_k contains 0.
        for k in range(1, len(actual_acf_values)):
            lower_ci_k = confint[k, 0]
            upper_ci_k = confint[k, 1]
            
            # If the confidence interval for ACF at lag k contains 0,
            # it means ACF_k is not statistically significantly different from 0.
            if lower_ci_k <= 0 and upper_ci_k >= 0:
                optimal_length = k
                optimal_length = max(1, optimal_length) # Ensure min block length
                optimal_length = min(optimal_length, max_lag) 
                print(f"Determined optimal block length: {optimal_length} (ACF at lag {k}: {actual_acf_values[k]:.3f} is not significant, CI: [{lower_ci_k:.3f}, {upper_ci_k:.3f}])")
                # Optional: Plot ACF for visual inspection if needed for debugging
                # sm.graphics.tsa.plot_acf(series, lags=max_lag)
                # plt.show() # Requires matplotlib.pyplot as plt
                return optimal_length
        
        print(f"Warning: ACF remained significant up to {max_lag} lags. Using max_lag: {max_lag} as block length.")
        return max_lag # Fallback if no such lag is found
    except Exception as e:
        print(f"Error determining block length: {e}. Using default: {default_block_size}")
        traceback.print_exc()
        return default_block_size
# -------------------------------------------------------------------------------------------------------------------------
def stationary_bootstrap(data, block_size, num_samples= 1000, sample_length = None, seed = None): 
    n = len(data)
    if sample_length is None:
        sample_length = n
    
    # Generate all random indices at once
    current_block_size = max(1, int(block_size)) # Ensure block_size is at least 1 and an integer
    p = 1.0 / current_block_size
    
    all_random_starts = np.random.randint(0, n, size=(num_samples, sample_length))
    all_random_continues = np.random.random(size=(num_samples, sample_length)) < p
    
    # Process in batches
    bootstrap_samples = []
    batch_size = 50  # Process 50 samples at a time to manage memory
    
    for batch_idx in range(0, num_samples, batch_size):
        batch_end = min(batch_idx + batch_size, num_samples)
        batch_samples = []
        
        for i in range(batch_idx, batch_end):
            indices = np.zeros(sample_length, dtype=int)
            if n == 0: # Handle empty data case
                if isinstance(data, pd.DataFrame):
                    batch_samples.append(pd.DataFrame(columns=data.columns, index=data.index[:0]))
                elif isinstance(data, pd.Series):
                     batch_samples.append(pd.Series(dtype=data.dtype, index=data.index[:0]))
                else: # Fallback for other types, though DataFrame is expected
                    batch_samples.append(data[:0])
                continue

            t = all_random_starts[i, 0]
            indices[0] = t
            
            for j in range(1, sample_length):
                if all_random_continues[i, j]:
                    t = all_random_starts[i, j]
                else:
                    t = (t + 1) % n
                indices[j] = t
            
            # Create sample more efficiently
            bootstrap_sample = data.iloc[indices]
            bootstrap_sample.index = data.index[:len(bootstrap_sample)] # Preserve original index type for resampling
            batch_samples.append(bootstrap_sample)
        
        bootstrap_samples.extend(batch_samples)
        
    return bootstrap_samples
#--------------------------------------------------------------------------------------------------------------------------  
def monte_carlo(prepared_data, pareto_front, num_simulations=1500):
    """Monte Carlo analysis with improved statistical visualization"""
    
    mc_results = []
    total_iterations = len(pareto_front)
    
    # Convert trial parameters to proper format once
    param_sets = []
    for trial in pareto_front:
        # Store the full flat dictionary from Optuna
        current_full_params = trial.params.copy()
        
        # Ensure main strategy parameters have correct types for robustness,
        # although Optuna usually returns correct types.
        # The momentum function's .get() method will handle missing keys with defaults.
        if 'long_risk' in current_full_params:
            current_full_params['long_risk'] = float(current_full_params['long_risk'])
        if 'max_open_positions' in current_full_params: # Optuna uses 'max_open_positions'
            current_full_params['max_open_positions'] = int(current_full_params['max_open_positions'])
        if 'adx_threshold' in current_full_params:
            current_full_params['adx_threshold'] = float(current_full_params['adx_threshold'])
        if 'max_position_duration' in current_full_params:
            current_full_params['max_position_duration'] = int(current_full_params['max_position_duration'])
        # Other parameters (weights, thresholds, lookbacks) will be used as is from trial.params
        
        param_sets.append(current_full_params)
    
    # Determine optimal block length based on ACF of prepared_data returns
    dynamic_block_size = BLOCK_SIZE # Default
    if not prepared_data.empty and 'Close' in prepared_data.columns and len(prepared_data['Close']) > 1:
        returns_for_acf = prepared_data['Close'].pct_change().dropna()
        if not returns_for_acf.empty:
            dynamic_block_size = determine_optimal_block_length(returns_for_acf, max_lag=50, default_block_size=BLOCK_SIZE)
        else:
            print(f"Warning: Returns series for ACF calculation is empty. Using default block size: {BLOCK_SIZE}")
    else:
        print(f"Warning: 'prepared_data' is empty or lacks 'Close' column for ACF. Using default block size: {BLOCK_SIZE}")
    
    dynamic_block_size = 10

    # Generate bootstrap samples once
    print(f"\nGenerating bootstrap samples with block_size: {dynamic_block_size}...")
    bootstrap_samples = stationary_bootstrap(
        data=prepared_data,
        block_size=dynamic_block_size, # Use dynamically determined block size
        num_samples=num_simulations,
        sample_length=None,
        seed=42
    )

    def process_parameter_set(param_idx):
        """Process a single parameter set"""
        current_full_params_for_momentum = param_sets[param_idx]
        
        # Run original strategy to get baseline performance
        trade_log, observed_stats, _, _ = momentum(
            prepared_data.copy(), 
            params=current_full_params_for_momentum # Pass the full flat dictionary
        )
        
        # Define a mapping from internal keys (used in this function) to original stat keys
        metric_key_map = {
            'profit_factor': 'Profit Factor',
            'expectancy_pct': 'Expectancy (%)',
            'avg_win_loss_ratio': 'Avg Win/Loss Ratio',
            'max_drawdown': 'Max Drawdown (%)'
        }
        
        # Get observed metrics using the mapping
        observed_metrics = {}
        for internal_key, original_key in metric_key_map.items():
            if original_key in observed_stats:
                observed_metrics[internal_key] = observed_stats[original_key]
            else:
                observed_metrics[internal_key] = np.nan

        sim_metrics = {internal_key: [] for internal_key in metric_key_map.keys()}
    
        num_bootstrap_samples = len(bootstrap_samples)
        pbar = tqdm.tqdm(
            total=num_bootstrap_samples,
            desc=f"Set {param_idx+1}",
            position=param_idx,
            leave=True,
            ncols=80  # Fixed width
        )

        for sample_idx, sample in enumerate(bootstrap_samples):
            _, sim_stats_run, _, _ = momentum(
                sample.copy(), 
                params=current_full_params_for_momentum
            )
            
            for internal_key, original_key in metric_key_map.items():
                if original_key in sim_stats_run:
                    sim_metrics[internal_key].append(sim_stats_run[original_key])
                else:
                    sim_metrics[internal_key].append(np.nan)
            
            pbar.update(1)
        
        pbar.close()
        
        results = {
            'parameter_set': param_idx + 1,
            'params': current_full_params_for_momentum,
            'p_values': {},
            'percentiles': {},
            'observed_metrics': observed_metrics,
            'simulation_metrics': {}
        }
        
        for internal_key in sim_metrics:
            sim_array_raw = np.array(sim_metrics[internal_key], dtype=float)

            # --- Prepare array for p-value and overall distribution percentiles (5th, 95th) ---
            # Here, Inf is treated as a very large (good or bad) number.
            sim_array_for_pvalue_and_percentiles = sim_array_raw[~np.isnan(sim_array_raw)]

            if internal_key in ['profit_factor', 'expectancy_pct', 'avg_win_loss_ratio']: # Higher is better
                sim_array_for_pvalue_and_percentiles[sim_array_for_pvalue_and_percentiles == np.inf] = 1e9
                sim_array_for_pvalue_and_percentiles[sim_array_for_pvalue_and_percentiles == -np.inf] = -1e9
            elif internal_key == 'max_drawdown': # Lower is better
                # For max_drawdown, inf means a terrible drawdown.
                sim_array_for_pvalue_and_percentiles[sim_array_for_pvalue_and_percentiles == np.inf] = 1e9 # Represents a very large (bad) drawdown
                sim_array_for_pvalue_and_percentiles[sim_array_for_pvalue_and_percentiles == -np.inf] = -1e9 # Represents a very small (good) drawdown, unlikely
            # No else needed if all relevant internal_keys are covered above

            # --- Prepare array for calculating mean, std, skew, kurtosis of *finite* outcomes ---
            sim_array_for_finite_stats = sim_array_raw[np.isfinite(sim_array_raw)] # Only finite values

            # --- Observed Value Handling (similar capping for p-value comparison) ---
            observed_value = observed_metrics.get(internal_key, np.nan)
            observed_value_for_comparison = observed_value 

            if pd.notna(observed_value_for_comparison) and np.isinf(observed_value_for_comparison):
                if internal_key in ['profit_factor', 'expectancy_pct', 'avg_win_loss_ratio']:
                    observed_value_for_comparison = 1e9 if observed_value_for_comparison > 0 else -1e9
                elif internal_key == 'max_drawdown': # Max drawdown is positive
                    observed_value_for_comparison = 1e9 # Inf drawdown is very bad

            # --- Calculations ---
            if np.isnan(observed_value) or len(sim_array_for_pvalue_and_percentiles) == 0:
                results['p_values'][internal_key] = np.nan
                results['percentiles'][internal_key] = np.nan # Percentile of observed value
                results['simulation_metrics'][internal_key] = {
                    'mean': np.nan, 'std': np.nan, 'skew': np.nan, 'kurtosis': np.nan,
                    'p5': np.nan, 'p95': np.nan
                }
                continue
            
            # Calculate p-value using the array where Inf is capped
            if internal_key in ['profit_factor', 'expectancy_pct', 'avg_win_loss_ratio']: # Higher is better
                p_value = np.mean(sim_array_for_pvalue_and_percentiles >= observed_value_for_comparison)
            elif internal_key == 'max_drawdown':  # Lower is better (Max Drawdown is positive)
                p_value = np.mean(sim_array_for_pvalue_and_percentiles <= observed_value_for_comparison)
            else: 
                p_value = np.nan
            
            # Calculate percentile of observed value using the array where Inf is capped
            percentile_of_observed = stats.percentileofscore(sim_array_for_pvalue_and_percentiles, observed_value_for_comparison)
            
            # Calculate 5th and 95th percentiles of the simulated distribution (where Inf is capped)
            # These reflect the spread of the distribution including extreme (capped inf) values.
            p5 = np.percentile(sim_array_for_pvalue_and_percentiles, 5) if len(sim_array_for_pvalue_and_percentiles) > 0 else np.nan
            p95 = np.percentile(sim_array_for_pvalue_and_percentiles, 95) if len(sim_array_for_pvalue_and_percentiles) > 0 else np.nan
            
            results['p_values'][internal_key] = p_value
            results['percentiles'][internal_key] = percentile_of_observed
            
            # Calculate descriptive stats (mean, std, etc.) using only *finite* simulated values
            mean_finite = np.mean(sim_array_for_finite_stats) if len(sim_array_for_finite_stats) > 0 else np.nan
            std_finite = np.std(sim_array_for_finite_stats) if len(sim_array_for_finite_stats) > 1 else np.nan # std needs at least 2 points
            skew_finite = stats.skew(sim_array_for_finite_stats) if len(sim_array_for_finite_stats) > 2 else np.nan
            kurt_finite = stats.kurtosis(sim_array_for_finite_stats) if len(sim_array_for_finite_stats) > 3 else np.nan

            results['simulation_metrics'][internal_key] = {
                'mean': mean_finite,
                'std': std_finite,
                'skew': skew_finite,
                'kurtosis': kurt_finite,
                'p5': p5,  # These are from the distribution including capped infinities
                'p95': p95
            }
        
        return results
    
    # Run parallel processing with progress bar
    print(f"\nRunning Monte Carlo Analysis across {len(param_sets)} parameter sets...")
    print(f"Total parameter sets: {total_iterations}")
    
    mc_results = Parallel(n_jobs=-1)(
        delayed(process_parameter_set)(i) 
        for i in range(len(param_sets))
    )
        
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(mc_results)
    
    # Print new formatted summary statistics
    print("\n=== BOOTSTRAP MONTE CARLO RESULTS ===")
    
    # 1. Summary section
    print("1. Simulation:")
    print(f"   - Bootstrap Samples: {num_simulations:,}")
    print(f"   - Avg Block Length: {dynamic_block_size}")
    
    # Additional data for summary
    total_obs_metrics = {}
    metric_keys_for_summary = ['profit_factor', 'expectancy_pct', 'avg_win_loss_ratio', 'max_drawdown']
    for metric in metric_keys_for_summary:
        values = [result['observed_metrics'][metric] for result in mc_results 
                  if pd.notna(result['observed_metrics'].get(metric))] # Use .get for safety
        if values:
            total_obs_metrics[metric] = np.mean(values)
    
    if all(metric in total_obs_metrics for metric in metric_keys_for_summary):
        print(f"   - Avg Observed Profit Factor: {total_obs_metrics['profit_factor']:.2f} | " + f"Avg Observed Win/Loss Ration: {total_obs_metrics['avg_win_loss_ratio']:.2f} | " + f"Avg Observed Expectancy: {total_obs_metrics['expectancy_pct']:.2f}% | " + 
              f"Max DD: {total_obs_metrics['max_drawdown']:.2f}%")
    
    # Get significant metric counts
    sig_counts_p_lt_0_10 = {}
    marg_sig_counts_p_lt_0_20 = {}

    for metric in metric_keys_for_summary:
        sig_counts_p_lt_0_10[metric] = sum(1 for result in mc_results
                                     if result['p_values'].get(metric, 1.0) < 0.10)
        marg_sig_counts_p_lt_0_20[metric] = sum(1 for result in mc_results
                                          if 0.10 <= result['p_values'].get(metric, 1.0) < 0.20)
    
    print(f"   - Significant Results (p<0.10): " + 
          f"PF: {sig_counts_p_lt_0_10.get('profit_factor',0)}/{len(mc_results)}, " +
          f"Expect: {sig_counts_p_lt_0_10.get('expectancy_pct',0)}/{len(mc_results)}, " +
          f"W/L Ratio: {sig_counts_p_lt_0_10.get('avg_win_loss_ratio',0)}/{len(mc_results)}, " +
          f"MaxDD: {sig_counts_p_lt_0_10.get('max_drawdown',0)}/{len(mc_results)}")
    print(f"   - Marg. Significant (0.10<=p<0.20): " +
          f"PF: {marg_sig_counts_p_lt_0_20.get('profit_factor',0)}/{len(mc_results)}, " +
          f"Expect: {marg_sig_counts_p_lt_0_20.get('expectancy_pct',0)}/{len(mc_results)}, " +
          f"W/L Ratio: {marg_sig_counts_p_lt_0_20.get('avg_win_loss_ratio',0)}/{len(mc_results)}, " +
          f"MaxDD: {marg_sig_counts_p_lt_0_20.get('max_drawdown',0)}/{len(mc_results)}")
    
    # 2-5. Null Distribution tables for each key metric
    metric_display_names = {
        'profit_factor': 'Profit Factor', 
        'expectancy_pct': 'Expectancy (%)',
        'avg_win_loss_ratio': 'Win/Loss Ratio',
        'max_drawdown': 'Max Drawdown (%)'
    }
    
    # For each metric, create a separate table
    for metric_num, metric in enumerate(metric_keys_for_summary, 2):
        print(f"\n{metric_num}. Null Distribution - {metric_display_names[metric]}:")
        
        # Create table data for this metric
        table_data = []
        headers = ['Set', 'Observed', 'Mean', 'Std Dev', '5th %ile', '95th %ile', 'p-value']
        
        for result in mc_results:
            param_set = result['parameter_set']
            observed = result['observed_metrics'].get(metric, np.nan)
            sim_stats = result['simulation_metrics'].get(metric, {})
            
            if not sim_stats or np.isnan(observed): # Check if sim_stats is empty or observed is NaN
                continue
                
            p_value = result['p_values'].get(metric, np.nan)
            
            row = [
                f"{param_set}",
                f"{observed:.2f}",
                f"{sim_stats.get('mean', np.nan):.2f}",
                f"{sim_stats.get('std', np.nan):.2f}",
                f"{sim_stats.get('p5', np.nan):.2f}",
                f"{sim_stats.get('p95', np.nan):.2f}",
                f"{p_value:.3f}"
            ]
            table_data.append(row)
        
        # Print table for this metric
        if table_data:
            print(tabulate(table_data, headers=headers, tablefmt='grid'))
        else:
            print("   No valid simulation data available for this metric")
    
    # Find best parameter set overall (lowest combined p-values)
    if mc_results:
        p_value_sums = []
        for res in mc_results:
            current_sum = 0
            for key in metric_keys_for_summary: # Use the same keys for consistency
                p_val = res['p_values'].get(key, 1.0) # Default to 1.0 if missing
                current_sum += float(p_val) if pd.notna(p_val) else 1.0
            p_value_sums.append(current_sum)

        if p_value_sums:
            best_set_idx = np.argmin(p_value_sums)
            best_set_result = mc_results[best_set_idx]
            best_set_num = best_set_result['parameter_set']
            
            print(f"\nBest Overall Parameter Set (by sum of p-values): #{best_set_num}")
            print("Key Performance Metrics:")
            
            best_metrics_table = []
            for metric in metric_keys_for_summary:
                observed = best_set_result['observed_metrics'].get(metric, np.nan)
                p_val = best_set_result['p_values'].get(metric, np.nan)
                sim_mean = best_set_result['simulation_metrics'].get(metric, {}).get('mean', np.nan)
                
                sig_marker = ''
                if pd.notna(p_val):
                    p_val_float = float(p_val)
                    if p_val_float < 0.10:
                        sig_marker = '*'  # Significant
                    elif 0.10 <= p_val_float < 0.20:
                        sig_marker = '~'  # Marginally Significant
                
                best_metrics_table.append([
                    f"{metric_display_names[metric]}{sig_marker}", 
                    f"{observed:.2f}",
                    f"{sim_mean:.2f}",
                    f"{observed - sim_mean:.2f}" if pd.notna(observed) and pd.notna(sim_mean) else "N/A"
                ])
            
            print(tabulate(best_metrics_table, headers=['Metric', 'Observed', 'Sim Mean', 'Edge'], tablefmt='simple'))
        else:
            print("   Could not determine best parameter set due to missing p-value data.")
    
    return results_df

#==========================================================================================================================
#================== STRATEGY ROBUSTNESS TESTING ===========================================================================
#==========================================================================================================================
def walk_forward_analysis(initial_is_data_raw, full_oos_data_raw, initial_parameters, risk_free_annual=RISK_FREE_RATE_ANNUAL):
    
    # Global constants from your script parameters
    oos_window_size = OOS_WINDOW
    opt_frequency_days = OPTIMIZATION_FREQUENCY # Days of OOS data to process before re-optimizing
    daily_rf_rate = risk_free_annual / 252.0
    days_in_is_lookback = 252 * 3 # Approx 3 years for the sliding IS window part

    # Initialize active parameters with the initial set
    active_parameters_for_momentum = initial_parameters.copy()
    
    combined_raw_data = pd.concat([initial_is_data_raw, full_oos_data_raw])
    prepared_full_data = prepare_data(combined_raw_data.copy(), type=2)

    # This will store the raw training data used for the *last* optimization
    # Initially, it's the full initial in-sample data
    training_data_at_last_optimization_raw = initial_is_data_raw.copy()
    
    # Get the prepared version of the initial training data from the fully prepared data
    if not initial_is_data_raw.empty:
        print(f"Slicing initial prepared training data: {initial_is_data_raw.index.min().date()} to {initial_is_data_raw.index.max().date()} ({len(initial_is_data_raw)} days)")
        prepared_current_train = prepared_full_data.loc[initial_is_data_raw.index.min():initial_is_data_raw.index.max()].copy()
        if prepared_current_train.empty:
            print("Initial training data slice is empty. Aborting WFA.")
            return None
    else:
        print("Initial raw IS data is empty. Cannot create initial prepared training data.")
        prepared_current_train = pd.DataFrame() # Will be handled by downstream checks

    # The raw OOS pool remains the same
    current_oos_pool_raw = full_oos_data_raw.copy()
    # Get the prepared version of the OOS pool from the fully prepared data
    if not full_oos_data_raw.empty:
        print(f"Slicing initial prepared OOS pool: {full_oos_data_raw.index.min().date()} to {full_oos_data_raw.index.max().date()} ({len(full_oos_data_raw)} days)")
        prepared_oos_pool = prepared_full_data.loc[full_oos_data_raw.index.min():full_oos_data_raw.index.max()].copy()
        if prepared_oos_pool.empty:
            print("OOS data pool slice is empty despite raw OOS data existing. Aborting WFA.")
            return None
    else:
        print("Raw OOS data pool is empty. WFA might end quickly or not run.")
        prepared_oos_pool = pd.DataFrame()

    all_step_results = []
    days_processed_since_last_opt = 0 
    step_number = 0
    accumulated_raw_oos_for_next_train = pd.DataFrame() # To accumulate raw OOS chunks

    print(f"\nStarting Unanchored Walk-Forward Analysis...")
    if not prepared_current_train.empty:
        print(f"Initial prepared training data for first cycle: {prepared_current_train.index.min().date()} to {prepared_current_train.index.max().date()} ({len(prepared_current_train)} days)")
    else:
        print(f"Initial prepared training data for first cycle is empty.")
    print(f"Total OOS data available: {len(prepared_oos_pool)} days | OOS Window Size: {oos_window_size} days | Optimization Frequency: {opt_frequency_days} days")
    print(f"OOS Pool Date Range: {prepared_oos_pool.index.min().date()} to {prepared_oos_pool.index.max().date()} ({len(prepared_oos_pool)} days)")
    

    total_possible_steps = len(prepared_oos_pool) // oos_window_size if oos_window_size > 0 and not prepared_oos_pool.empty else 0
    progress_bar = tqdm.tqdm(
        total=total_possible_steps,
        desc="WFA Progress",
        position=0,
        leave=True,
        ncols=100
    )
    
    while len(current_oos_pool_raw) >= oos_window_size and len(prepared_oos_pool) >= oos_window_size:
        step_number += 1
        
        # 1. Re-optimization Check
        if days_processed_since_last_opt >= opt_frequency_days and step_number > 1:
            #print(f"\nStep {step_number}: Re-optimizing parameters...")
            #print(f"Accumulated {len(accumulated_raw_oos_for_next_train)} days of OOS data for new training set.")
            
            if len(training_data_at_last_optimization_raw) < days_in_is_lookback:
                base_for_new_train_raw_segment = training_data_at_last_optimization_raw.copy()
                print(f"Warning: Training data used in last optimization ({len(training_data_at_last_optimization_raw)} days) is shorter than IS lookback ({days_in_is_lookback} days). Using all available.")
            else:
                base_for_new_train_raw_segment = training_data_at_last_optimization_raw.iloc[-days_in_is_lookback:].copy()
            
            new_raw_training_data_for_opt = pd.concat([base_for_new_train_raw_segment, accumulated_raw_oos_for_next_train])
            new_raw_training_data_for_opt = new_raw_training_data_for_opt[~new_raw_training_data_for_opt.index.duplicated(keep='first')].sort_index()

            prepared_new_training_data_for_opt = pd.DataFrame()
            if not new_raw_training_data_for_opt.empty:
                #print(f"Slicing new prepared training data for optimization from {new_raw_training_data_for_opt.index.min().date()} to {new_raw_training_data_for_opt.index.max().date()} (length: {len(new_raw_training_data_for_opt)} days)...")
                try:
                    prepared_new_training_data_for_opt = prepared_full_data.loc[new_raw_training_data_for_opt.index.min():new_raw_training_data_for_opt.index.max()].copy()
                except KeyError:
                    print(f"Error: Date range for new training data not found in prepared_full_data. Min: {new_raw_training_data_for_opt.index.min()}, Max: {new_raw_training_data_for_opt.index.max()}")
                    prepared_new_training_data_for_opt = pd.DataFrame()


            if prepared_new_training_data_for_opt.empty:
                print("Cannot optimize: New prepared training data for optimization is empty. Continuing with old parameters.")
            else:
                #print(f"Optimizing with new training data from {prepared_new_training_data_for_opt.index.min().date()} to {prepared_new_training_data_for_opt.index.max().date()} ({len(prepared_new_training_data_for_opt)} days)")
                pareto_front = optimize(prepared_new_training_data_for_opt) 
                
                if pareto_front and len(pareto_front) > 0:
                    best_trial = pareto_front[0] 
                    active_parameters_for_momentum = best_trial.params.copy()
                    # Ensure correct types for main strategy parameters
                    if 'long_risk' in active_parameters_for_momentum:
                        active_parameters_for_momentum['long_risk'] = float(active_parameters_for_momentum['long_risk'])
                    if 'max_open_positions' in active_parameters_for_momentum:
                        active_parameters_for_momentum['max_open_positions'] = int(active_parameters_for_momentum['max_open_positions'])
                    if 'adx_threshold' in active_parameters_for_momentum:
                        active_parameters_for_momentum['adx_threshold'] = float(active_parameters_for_momentum['adx_threshold'])
                    if 'max_position_duration' in active_parameters_for_momentum:
                        active_parameters_for_momentum['max_position_duration'] = int(active_parameters_for_momentum['max_position_duration'])
                    #print("Parameters updated after re-optimization.")
                    
                    training_data_at_last_optimization_raw = new_raw_training_data_for_opt.copy()
                    prepared_current_train = prepared_new_training_data_for_opt.copy() 
                else:
                    print("Optimization failed or yielded no results. Continuing with previously optimized parameters.")
            
            days_processed_since_last_opt = 0 
            accumulated_raw_oos_for_next_train = pd.DataFrame()
        
        # 2. Define and Prepare Current Test Window (OOS Chunk)
        raw_oos_chunk = current_oos_pool_raw.iloc[:oos_window_size].copy()
        prepared_oos_chunk = prepared_oos_pool.iloc[:oos_window_size].copy()

        if prepared_oos_chunk.empty or raw_oos_chunk.empty:
            print(f"Step {step_number}: OOS chunk is empty. Ending WFA.")
            break

        # 3. Run strategy on the current prepared training data (for reference)
        # `prepared_current_train` is the data that was used (or would have been used) to get `active_parameters_for_momentum`
        if prepared_current_train.empty:
            print(f"Step {step_number}: Warning - prepared_current_train for reference run is empty.")
            train_log, train_stats, train_equity, train_returns = [], {}, pd.Series(dtype='float64'), pd.Series(dtype='float64')
        else:
            train_log, train_stats, train_equity, train_returns = momentum(
                prepared_current_train.copy(), 
                params=active_parameters_for_momentum
            )

        # 4. Run strategy on the prepared OOS chunk (current test window)
        oos_log, oos_stats, oos_equity, oos_returns = momentum(
            prepared_oos_chunk.copy(),
            params=active_parameters_for_momentum
        )
        
        # 5. Record results for this step (existing logic for metrics extraction)
        oos_trade_count = oos_stats.get('Total Trades', 0)
        train_trade_count = train_stats.get('Total Trades', 0)
        
        train_sharpe = train_stats.get('Sharpe Ratio', np.nan)
        oos_sharpe = oos_stats.get('Sharpe Ratio', np.nan)
        train_annualized_return = train_stats.get('Annualized Return (%)', np.nan)
        oos_annualized_return = oos_stats.get('Annualized Return (%)', np.nan)
        train_max_drawdown = train_stats.get('Max Drawdown (%)', np.nan)
        oos_max_drawdown = oos_stats.get('Max Drawdown (%)', np.nan)
        train_sortino = train_stats.get('Sortino Ratio', np.nan)
        oos_sortino = oos_stats.get('Sortino Ratio', np.nan)
        train_avg_win_loss = train_stats.get('Avg Win/Loss Ratio', np.nan)
        oos_avg_win_loss = oos_stats.get('Avg Win/Loss Ratio', np.nan)
        train_expectancy = train_stats.get('Expectancy (%)', np.nan)
        oos_expectancy = oos_stats.get('Expectancy (%)', np.nan)
        train_profit_factor = train_stats.get('Profit Factor', np.nan)
        oos_profit_factor = oos_stats.get('Profit Factor', np.nan)
        
        is_valid_train_period = train_trade_count > 0
        is_valid_oos_period = oos_trade_count > 0
        
        if not is_valid_oos_period: oos_sharpe = 0.0
        if not is_valid_train_period: train_sharpe = 0.0
        
        decay_ratio_val = np.nan
        decay_ratios = {}
        step_result_data = {} 
        valid_decays = []

        if is_valid_train_period and is_valid_oos_period:
            metric_pairs = {
                'profit_factor': (train_profit_factor, oos_profit_factor),
                'sharpe': (train_sharpe, oos_sharpe),
                'sortino': (train_sortino, oos_sortino),
                'avg_win_loss': (train_avg_win_loss, oos_avg_win_loss),
                'expectancy': (train_expectancy, oos_expectancy),
                'ann_return': (train_annualized_return, oos_annualized_return),
                'max_drawdown': (train_max_drawdown, oos_max_drawdown)
            }
            for metric_name, (train_val, oos_val) in metric_pairs.items():
                if pd.notna(train_val) and pd.notna(oos_val):
                    if metric_name == 'profit_factor':
                        if train_val == np.inf: decay_ratios[metric_name] = 0.0 if oos_val == np.inf else 1.0
                        elif oos_val == np.inf: decay_ratios[metric_name] = -1.0
                        else:
                            denominator = max(train_val, 0.1)
                            performance_ratio = oos_val / denominator if denominator != 0 else (1.0 if oos_val >=0 else -1.0) # Avoid div by zero
                            decay_ratios[metric_name] = 1.0 - performance_ratio
                    elif metric_name == 'max_drawdown':
                        if train_val == 0: decay_ratios[metric_name] = 0.0 if oos_val == 0 else 1.0
                        else:
                            denominator = max(abs(train_val), 0.1)
                            performance_ratio = min(oos_val / denominator if denominator != 0 else 1.0, 2.0) # Avoid div by zero
                            decay_ratios[metric_name] = 1.0 - (1.0 / performance_ratio if performance_ratio != 0 else 10.0) # Avoid div by zero
                    else:
                        if train_val == 0: decay_ratios[metric_name] = 0.0 if oos_val == 0 else -1.0
                        else:
                            denominator = max(abs(train_val), 0.1)
                            performance_ratio = min(oos_val / denominator if denominator != 0 else 1.0, 2.0) # Avoid div by zero
                            decay_ratios[metric_name] = 1.0 - performance_ratio
                    step_result_data[f'decay_{metric_name}'] = decay_ratios.get(metric_name, np.nan)
            valid_decays = [v for v in decay_ratios.values() if pd.notna(v)]
            if valid_decays: decay_ratio_val = np.mean(valid_decays)
        
        step_result_data.update({
            'step': step_number,
            'train_start_date': prepared_current_train.index[0].date() if not prepared_current_train.empty else None,
            'train_end_date': prepared_current_train.index[-1].date() if not prepared_current_train.empty else None,
            'train_days': len(prepared_current_train),
            'test_start_date': prepared_oos_chunk.index[0].date() if not prepared_oos_chunk.empty else None,
            'test_end_date': prepared_oos_chunk.index[-1].date() if not prepared_oos_chunk.empty else None,
            'test_days': len(prepared_oos_chunk),
            'train_sharpe': train_sharpe, 'test_sharpe': oos_sharpe, 'decay_ratio': decay_ratio_val,
            'decay_metrics_used': len(valid_decays), 'train_trades': train_trade_count, 'test_trades': oos_trade_count,
            'oos_trade_log': list(oos_log) if oos_log else [],
            'oos_returns_series': oos_returns, 
            'train_return_pct': train_stats.get('Return (%)', np.nan), 'test_return_pct': oos_stats.get('Return (%)', np.nan),
            'valid_train': is_valid_train_period, 'valid_test': is_valid_oos_period,
            'valid_comparison': is_valid_train_period and is_valid_oos_period,
            'train_ann_return': train_annualized_return, 'test_ann_return': oos_annualized_return,
            'train_max_drawdown': train_max_drawdown, 'test_max_drawdown': oos_max_drawdown,
            'train_sortino': train_sortino, 'test_sortino': oos_sortino,
            'train_avg_win_loss': train_avg_win_loss, 'test_avg_win_loss': oos_avg_win_loss,
            'train_expectancy': train_expectancy, 'test_expectancy': oos_expectancy,
            'train_profit_factor': train_profit_factor, 'test_profit_factor': oos_profit_factor,
            'parameters_used_snapshot': active_parameters_for_momentum.copy()
        })
        all_step_results.append(step_result_data)
        progress_bar.update(1)
        
        # 6. Update data for the next iteration
        # Accumulate the raw OOS chunk for the next potential training set (if re-optimization occurs)
        accumulated_raw_oos_for_next_train = pd.concat([accumulated_raw_oos_for_next_train, raw_oos_chunk])
        accumulated_raw_oos_for_next_train = accumulated_raw_oos_for_next_train[~accumulated_raw_oos_for_next_train.index.duplicated(keep='first')].sort_index()

        # Advance the OOS pool (both raw and prepared)
        chunk_len = len(raw_oos_chunk) # Number of days in the current OOS chunk
        current_oos_pool_raw = current_oos_pool_raw.iloc[chunk_len:].copy()
        prepared_oos_pool = prepared_oos_pool.iloc[chunk_len:].copy()
        
        # Update days processed counter for OOS data
        days_processed_since_last_opt += chunk_len
    
    progress_bar.close()

    if not all_step_results:
        print("No walk-forward steps were completed.")
        return None
    # ----------------------------------------------------------------------------------------------------------------------------
    # Create results DataFrame
    results_df = pd.DataFrame(all_step_results)
    
    # Process all OOS returns for overall metrics
    all_oos_log_returns_list = [res['oos_returns_series'] for res in all_step_results 
                               if isinstance(res['oos_returns_series'], pd.Series) and not res['oos_returns_series'].empty]
    
    # Collect all OOS trade logs
    all_oos_trades_list = []
    for res_dict in all_step_results: # Iterate through the list of dictionaries
        if 'oos_trade_log' in res_dict and res_dict['oos_trade_log']:
            all_oos_trades_list.extend(res_dict['oos_trade_log'])

    total_oos_wins = 0
    total_oos_losses = 0
    if all_oos_trades_list:
        for trade in all_oos_trades_list:
            if isinstance(trade, dict) and trade.get('PnL', 0) > 0:
                total_oos_wins += 1
            elif isinstance(trade, dict): # Count non-positive PnL as losses
                total_oos_losses += 1

    concatenated_oos_log_returns = pd.Series(dtype=float)
    overall_oos_sharpe = np.nan
    overall_oos_cumulative_return_pct = np.nan 
    overall_max_drawdown = np.nan

    if all_oos_log_returns_list:
        concatenated_oos_log_returns = pd.concat(all_oos_log_returns_list).sort_index()
        concatenated_oos_log_returns = concatenated_oos_log_returns[~concatenated_oos_log_returns.index.duplicated(keep='first')]
    
    # Calculate metrics for final report
    total_steps = len(results_df)
    valid_test_results_df = results_df[results_df['valid_test'] == True] if not results_df.empty else pd.DataFrame()
    valid_comparisons_df = results_df[results_df['valid_comparison'] == True] if not results_df.empty else pd.DataFrame()
    
    valid_tests_count = len(valid_test_results_df)
    zero_trade_count = total_steps - valid_tests_count
    
    # Calculate rolling decay ratios (for detailed analysis)
    rolling_decay = None
    if len(valid_comparisons_df) >= 2:
        decay_values = valid_comparisons_df['decay_ratio'].dropna()
        if len(decay_values) >= 2:
            # Calculate rolling window statistics if enough data points
            rolling_size = min(3, len(decay_values))
            rolling_decay = decay_values.rolling(rolling_size).mean()
    
    # Calculate overall OOS performance metrics
    active_trading_days = 0
    total_days_in_oos_concat = 0
    if len(concatenated_oos_log_returns) > 1:
        active_trading_days = len(concatenated_oos_log_returns[concatenated_oos_log_returns != 0])
        total_days_in_oos_concat = len(concatenated_oos_log_returns)
        
        if active_trading_days > 5:
            # Calculate OOS Sharpe
            excess_concatenated_log_returns = concatenated_oos_log_returns - daily_rf_rate
            mean_excess_log_return = excess_concatenated_log_returns.mean()
            std_excess_log_return = excess_concatenated_log_returns.std()
            
            if std_excess_log_return != 0 and pd.notna(std_excess_log_return):
                overall_oos_sharpe = (mean_excess_log_return / std_excess_log_return) * np.sqrt(252)
            else: 
                overall_oos_sharpe = 0.0 if mean_excess_log_return == 0 else np.nan
            
            # Calculate cumulative return
            total_cumulative_log_return = concatenated_oos_log_returns.sum()
            overall_oos_cumulative_return_pct = (np.exp(total_cumulative_log_return) - 1) * 100
            
            # Calculate max drawdown for overall OOS period
            cum_rets = concatenated_oos_log_returns.cumsum()
            cum_rets_exp = np.exp(cum_rets) - 1  # Convert to regular returns for drawdown calc
            running_max = np.maximum.accumulate(cum_rets_exp)
            drawdowns = ((cum_rets_exp - running_max) / (running_max + 1)) * 100  # As percentage
            overall_max_drawdown = np.nanmin(drawdowns)
    
    # Calculate additional performance metrics before summary
    if not valid_test_results_df.empty:
        # Calculate averages for key metrics
        avg_metrics = {
            'sharpe': valid_test_results_df['test_sharpe'].mean(),
            'sortino': valid_test_results_df['test_sortino'].mean(),
            'avg_win_loss': valid_test_results_df['test_avg_win_loss'].mean(),
            'expectancy': valid_test_results_df['test_expectancy'].mean(),
            'profit_factor': valid_test_results_df['test_profit_factor'].mean()
        }

        # Calculate decay metrics statistics
        if not valid_comparisons_df.empty:
            decay_metrics = {
                'sharpe_decay': valid_comparisons_df['decay_sharpe'].mean() if 'decay_sharpe' in valid_comparisons_df else np.nan,
                'sortino_decay': valid_comparisons_df['decay_sortino'].mean() if 'decay_sortino' in valid_comparisons_df else np.nan,
                'avg_win_loss_decay': valid_comparisons_df['decay_avg_win_loss'].mean() if 'decay_avg_win_loss' in valid_comparisons_df else np.nan,
                'expectancy_decay': valid_comparisons_df['decay_expectancy'].mean() if 'decay_expectancy' in valid_comparisons_df else np.nan,
                'profit_factor_decay': valid_comparisons_df['decay_profit_factor'].mean() if 'decay_profit_factor' in valid_comparisons_df else np.nan
            }

    # ----------------------------------------------------------------------------------------------------------------------------
    # Print the new formatted summary
    print("\n=== WFA FINAL SUMMARY ===")
    # ----------------------------------------------------------------------------------------------------------------------------
    # 1. Overview section
    print("1. Overview:")
    if total_steps > 0:
        print(f"   - Total WFA Steps: {total_steps}")
        print(f"   - Valid OOS Windows: {valid_tests_count}/{total_steps} ({valid_tests_count/total_steps*100:.1f}%)")
        
        # Calculate average number of decay metrics used across all valid steps
        if not valid_comparisons_df.empty:
            avg_decay_metrics = valid_comparisons_df['decay_metrics_used'].mean()
            print(f"   - Decay Metrics Used: {avg_decay_metrics:.1f}")
        
        if zero_trade_count > 0:
            zero_trade_pct = zero_trade_count/total_steps*100
            risk_level = "HIGH RISK" if zero_trade_pct > 40 else "MODERATE RISK" if zero_trade_pct > 20 else "LOW RISK"
            print(f"   - Zero-Trade Windows: {zero_trade_count}/{total_steps} ({zero_trade_pct:.1f}%) → {risk_level}")
    else:
        print("   - No WFA steps completed")
    # ----------------------------------------------------------------------------------------------------------------------------
    # 2. Performance metrics table
    print("\n2. Performance (Valid OOS):")
    if not valid_test_results_df.empty:
        metrics = {
            'OOS Sharpe': valid_test_results_df['test_sharpe'],
            'OOS Sortino': valid_test_results_df['test_sortino'],
            'OOS Avg Win/Loss': valid_test_results_df['test_avg_win_loss'],
            'OOS Expectancy (%)': valid_test_results_df['test_expectancy'],
            'OOS Profit Factor': valid_test_results_df['test_profit_factor'],
            'Ann. Return (%)': valid_test_results_df['test_ann_return'],
            'Max Drawdown (%)': valid_test_results_df['test_max_drawdown']
        }

        # Calculate statistics for each metric
        summary_data = []
        for name, values in metrics.items():
            valid_values = values.dropna()
            if not valid_values.empty:
                row = [
                    name,
                    f"{valid_values.mean():.2f}",
                    f"{valid_values.median():.2f}",
                    f"{valid_values.std():.2f}",
                    f"{valid_values.min():.2f}",
                    f"{valid_values.max():.2f}"
                ]
                summary_data.append(row)

        if summary_data:
            print(tabulate(
                summary_data,
                headers=['Metric', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                tablefmt='grid',
                numalign='right'
            ))
        else:
            print("   No valid metrics data available")
    else:
        print("   No valid OOS periods with trades for performance analysis")
    # ----------------------------------------------------------------------------------------------------------------------------
    # 3. Decay Analysis
    print("\n3. Decay Analysis:")
    if not valid_comparisons_df.empty:
        # Ensure 'decay_metrics_used' is excluded from the loop for individual metric decays
        decay_columns = [
            col for col in valid_comparisons_df.columns 
            if col.startswith('decay_') and col not in ['decay_ratio', 'decay_metrics_used']
        ]
        
        metric_decays_data_for_table = [] # Renamed to avoid confusion
        total_mean_decay_sum = 0
        total_std_decay_sum = 0
        actual_metric_count_for_combined_mean = 0
        
        
        if decay_columns: # Check if there are actual decay metrics to process
            # Process each individual decay metric (e.g., decay_sharpe, decay_profit_factor)
            for col in decay_columns:
                metric_name = col.replace('decay_', '').title()
                values = valid_comparisons_df[col].dropna()
                
                if not values.empty:
                    mean_decay_val = values.mean()
                    std_decay_val = values.std()
                    
                    if pd.notna(mean_decay_val) and pd.notna(std_decay_val):
                        metric_decays_data_for_table.append([
                            metric_name,
                            f"{mean_decay_val:.2f}",
                            f"{std_decay_val:.2f}"
                        ])
                        total_mean_decay_sum += mean_decay_val
                        total_std_decay_sum += std_decay_val
                        actual_metric_count_for_combined_mean += 1
            
            # Add combined means at the bottom, only if actual metrics were processed
            if actual_metric_count_for_combined_mean > 0:
                metric_decays_data_for_table.append([
                    "Combined Mean (Perf. Metrics)", # Clarified name
                    f"{total_mean_decay_sum / actual_metric_count_for_combined_mean:.2f}",
                    f"{total_std_decay_sum / actual_metric_count_for_combined_mean:.2f}"
                ])
        
        # Display the table if there's data
        if metric_decays_data_for_table:
            print(tabulate(
                metric_decays_data_for_table,
                headers=['Metric', 'Decay', 'Std Dev'], # Corrected header from previous suggestions if needed
                tablefmt='simple',
                numalign='right'
            ))
            print()  # Add spacing
        else:
            print("   No individual decay metric data to display.")
        
        # Overall decay analysis based on 'decay_ratio'
        decay_values_overall = valid_comparisons_df['decay_avg_win_loss'].dropna() # Use 'decay_avg_win_loss'
        
        if not decay_values_overall.empty:
            mean_decay = decay_values_overall.mean() # Use 'decay_avg_win_loss'
            std_decay = decay_values_overall.std()   # Call std() and use 'decay_avg_win_loss'
            
            # Classify decay ratio (using the same thresholds for now)
            decay_classification = ("CATASTROPHIC" if mean_decay > 0.7 else
                                    "SEVERE" if mean_decay > 0.5 else
                                    "SIGNIFICANT" if mean_decay > 0.3 else
                                    "MODERATE" if mean_decay > 0.1 else
                                    "MINIMAL")
            
            # Classify volatility
            vol_classification = ("EXTREME VOLATILITY" if std_decay > 0.4 else
                                "HIGH VOLATILITY" if std_decay > 0.2 else
                                "MODERATE VOLATILITY" if std_decay > 0.1 else
                                "STABLE")
            
            print(f"Overall Decay Analysis (Based on Avg Win/Loss Decay):") # Updated label
            print(f"   - Mean Avg Win/Loss Decay: {mean_decay:.2f} → {decay_classification}")
            print(f"   - Std Dev Avg Win/Loss Decay: {std_decay:.2f} → {vol_classification}")
            
            # Note: 'rolling_decay' should also be calculated based on 'decay_avg_win_loss' (see note below)
            # The 'decay_values_overall' variable used for 'recent_rolling_decay' print is now based on 'decay_avg_win_loss'
            if rolling_decay is not None and len(rolling_decay.dropna()) > 0:
                recent_decay_val = rolling_decay.dropna().iloc[-1]
                if pd.notna(recent_decay_val):
                    # 'decay_values_overall' for rolling_decay print is now consistent.
                    print(f"   - Recent Rolling Avg Win/Loss Decay (n={min(3, len(decay_values_overall))}): {recent_decay_val:.2f}")
        else:
            print("   No 'decay_avg_win_loss' data available for overall analysis.")
    else:
        print("   No valid comparison periods to calculate decay metrics")
    # ----------------------------------------------------------------------------------------------------------------------------
    # 4. Activity metrics
    print("\n4. Activity:")
    if total_days_in_oos_concat > 0:
        trading_activity_pct = active_trading_days / total_days_in_oos_concat * 100
        print(f"   - Trading Days: {active_trading_days}/{total_days_in_oos_concat} ({trading_activity_pct:.1f}%)")
        
        # Identify potential causes for low activity or zero-trade steps
        # zero_trade_count is already calculated as total_steps - valid_tests_count
        # valid_tests_count is where oos_trade_count > 0. So zero_trade_count is correct.
        
        if trading_activity_pct < 3 and zero_trade_count > 0:
            print("   - Low overall trading activity (<3%). Analyzing parameters of zero-trade steps:")
            zero_trade_steps_df = results_df[results_df['test_trades'] == 0]
            
            for _, step_row in zero_trade_steps_df.iterrows():
                params = step_row['parameters_used_snapshot']
                step_num = step_row['step']
                causes_for_step = []

                # Check entry threshold (threshold_buy_score)
                # Default buy score is DEFAULT_SIGNAL_PROCESSING_PARAMS['thresholds']['buy_score'] (e.g., 55)
                buy_score_thresh = params.get('threshold_buy_score')
                if buy_score_thresh is not None and buy_score_thresh > 65: # Example: if optimized significantly higher
                    causes_for_step.append(f"high buy_score_thresh ({buy_score_thresh})")

                # Check ADX threshold
                # Default ADX threshold is ADX_THRESHOLD_DEFAULT (e.g., 25)
                adx_val_thresh = params.get('adx_threshold')
                if adx_val_thresh is not None and adx_val_thresh > 30: # Example: if optimized significantly higher
                    causes_for_step.append(f"high adx_threshold ({adx_val_thresh:.1f})")
                
                # Check signal weights
                signal_weights_keys = [
                    'weight_price_trend', 'weight_rsi_zone', 'weight_adx_slope', 
                    'weight_vol_accel', 'weight_vix_factor'
                ]
                low_weights_details = []
                # Default weights are typically around 0.20 each
                for weight_key in signal_weights_keys:
                    weight_val = params.get(weight_key)
                    if weight_val is not None and weight_val < 0.05: # Example: if a weight is very low
                        low_weights_details.append(f"{weight_key.replace('weight_', '')}: {weight_val:.2f}")
                
                if low_weights_details:
                    causes_for_step.append(f"low signal weights ({', '.join(low_weights_details)})")

                if causes_for_step:
                    print(f"     - Step {step_num}: Zero trades. Potential causes: {'; '.join(causes_for_step)}.")
                else:
                    # If no obvious parameter red flags, it might be other combinations or market conditions
                    print(f"     - Step {step_num}: Zero trades. Cause: Parameter combination or specific market conditions not flagged by simple checks.")
            
        elif zero_trade_count > 0: # Overall activity is not <3%, but there were still some zero-trade windows
            print(f"   - Note: {zero_trade_count} OOS window(s) had zero trades. This might indicate parameter instability or specific unresponsive market conditions during those periods.")
    else:
        print("   - No trading activity data available")
    # ----------------------------------------------------------------------------------------------------------------------------
    # 5. Concatenated OOS Performance
    print("\n5. Concatenated OOS:")
    if len(concatenated_oos_log_returns) > 5:
        print(f"   - Ann. Sharpe: {overall_oos_sharpe:.2f} | Cum. Return: {overall_oos_cumulative_return_pct:.2f}% | Max DD: {overall_max_drawdown:.1f}%")
        print(f"   - Total OOS Trades: {total_oos_wins + total_oos_losses} (Wins: {total_oos_wins}, Losses: {total_oos_losses})")
        
        # Market classification based on overall cumulative return
        market_type = ""
        if pd.notna(overall_oos_cumulative_return_pct):
            if overall_oos_cumulative_return_pct > 50:
                market_type = "HIGHLY FAVORABLE"
            elif overall_oos_cumulative_return_pct > 20:
                market_type = "FAVORABLE" 
            elif overall_oos_cumulative_return_pct > -5:
                market_type = "NEUTRAL"
            elif overall_oos_cumulative_return_pct > -20:
                market_type = "CHALLENGING"
            else:
                market_type = "HIGHLY ADVERSE"
        else:
            market_type = "UNKNOWN (Return N/A)"
            
        print(f"   - Market Classification (by Return): {market_type}")
    else:
        print("   - Insufficient data for reliable OOS performance metrics")
    # ----------------------------------------------------------------------------------------------------------------------------
    # Display detailed results table
    if not results_df.empty:
        print("\n--- Detailed Step Results ---")
        display_cols = [
            'step', 'train_sharpe', 'test_sharpe', 'decay_ratio', 
            'train_ann_return', 'test_ann_return',
            'train_max_drawdown', 'test_max_drawdown',
            'train_profit_factor', 'test_profit_factor',
            'train_trades', 'test_trades'
        ]
        
        # Filter columns that exist in the dataframe
        available_cols = [col for col in display_cols if col in results_df.columns]
        if available_cols:
            # Create a mapping for prettier metric names
            metric_names = {
                'train_sharpe': 'Train Sharpe',
                'test_sharpe': 'Test Sharpe',
                'decay_ratio': 'Decay Ratio',
                'train_ann_return': 'Train Ann Ret(%)',
                'test_ann_return': 'Test Ann Ret(%)',
                'train_max_drawdown': 'Train MaxDD(%)',
                'test_max_drawdown': 'Test MaxDD(%)',
                'train_profit_factor': 'Train PF',
                'test_profit_factor': 'Test PF',
                'train_trades': 'Train Trades',
                'test_trades': 'Test Trades'
            }
            
            # Prepare the transposed data
            steps = results_df['step'].tolist()
            metrics_data = []
            
            for col in available_cols:
                if col != 'step':  # Skip the step column as it becomes our header
                    metric_name = metric_names.get(col, col)  # Get pretty name or use original
                    values = results_df[col].tolist()
                    metrics_data.append([metric_name] + [f"{v:.2f}" if isinstance(v, (float, np.float32, np.float64)) else str(v) for v in values])
            
            # Create headers for the table
            headers = ['Metric'] + [f'Step {s}' for s in steps]
            
            # Print the transposed table
            print(tabulate(
                metrics_data,
                headers=headers,
                tablefmt='grid',
                stralign='left',
                numalign='right'
            ))
    # ----------------------------------------------------------------------------------------------------------------------------
    # Return comprehensive results
    return {
        'step_results_df': results_df, 
        'concatenated_oos_returns': concatenated_oos_log_returns, 
        'overall_oos_sharpe': overall_oos_sharpe,
        'overall_oos_return_pct': overall_oos_cumulative_return_pct,
        'overall_oos_max_drawdown': overall_max_drawdown,
        'rolling_decay': rolling_decay if rolling_decay is not None else None
    }

#==========================================================================================================================
#================== MAIN PROGRAM EXECUTION ================================================================================
#==========================================================================================================================
def main():
    # Helper class for when optimization is skipped
    class MockOptunaTrial:
        def __init__(self, params_dict, optimization_directions_dict):
            self.params = params_dict
            # Create dummy objective values based on optimization directions
            self.values = []
            for metric_name in optimization_directions_dict.keys():
                if optimization_directions_dict[metric_name] == 'maximize':
                    self.values.append(1.0)  # Dummy "good" value for maximization
                else:  # minimize
                    self.values.append(10.0) # Dummy "good" value for minimization (e.g., low drawdown)

    default_params_dict = {
        'long_risk': DEFAULT_LONG_RISK,
        'max_open_positions': MAX_OPEN_POSITIONS,
        'adx_threshold': ADX_THRESHOLD_DEFAULT,
        'max_position_duration': MAX_POSITION_DURATION,

        'weight_price_trend': DEFAULT_SIGNAL_PROCESSING_PARAMS['weights']['price_trend'],
        'weight_rsi_zone': DEFAULT_SIGNAL_PROCESSING_PARAMS['weights']['rsi_zone'],
        'weight_adx_slope': DEFAULT_SIGNAL_PROCESSING_PARAMS['weights']['adx_slope'],
        'weight_vol_accel': DEFAULT_SIGNAL_PROCESSING_PARAMS['weights']['vol_accel'],
        'weight_vix_factor': DEFAULT_SIGNAL_PROCESSING_PARAMS['weights']['vix_factor'],

        'threshold_buy_score': DEFAULT_SIGNAL_PROCESSING_PARAMS['thresholds']['buy_score'],
        'threshold_exit_score': DEFAULT_SIGNAL_PROCESSING_PARAMS['thresholds']['exit_score'],
        'threshold_immediate_exit_score': DEFAULT_SIGNAL_PROCESSING_PARAMS['thresholds']['immediate_exit_score'],
        
        'ranking_lookback_window_opt': DEFAULT_SIGNAL_PROCESSING_PARAMS['ranking_lookback_window'],
        'momentum_volatility_lookback_opt': DEFAULT_SIGNAL_PROCESSING_PARAMS['momentum_volatility_lookback']
    }

    try:
        IS, OOS = get_data(TICKER)

        # ------------------------------------------------------------------------------------------------------------------
        if TYPE == 5:
            df_prepared_for_test = prepare_data(IS.copy(), type=1)
            if df_prepared_for_test is None or df_prepared_for_test.empty:
                print("Data for test is empty after preparation. Aborting.")
                return
            test(df_prepared_for_test)
        
        # -------------------------------------------------------------------------------------------------------------------
        elif TYPE == 4:
            df_prepared_for_opt = prepare_data(IS.copy(), type=1)
            if df_prepared_for_opt is None or df_prepared_for_opt.empty:
                print("Data for optimization is empty after preparation. Aborting.")
                return
            # Run optimization on in-sample data
            pareto_front = optimize(df_prepared_for_opt)
            if pareto_front:
                visualize(pareto_front, df_prepared_for_opt)
            else:
                print("Optimization did not yield any results.")

        # -------------------------------------------------------------------------------------------------------------------
        elif TYPE == 3:  # Monte Carlo Testing
            df_prepared_for_mc = prepare_data(IS.copy(), type=1)
            if df_prepared_for_mc is None or df_prepared_for_mc.empty:
                print("Data for Monte Carlo is empty after preparation. Aborting.")
                return
            
            pareto_front_mc = None
            if OPTIMIZATION:
                print("\nRunning optimization before Monte Carlo...")
                pareto_front_mc = optimize(df_prepared_for_mc)[:3] # Optimize first, take top 3
            else:
                print("OPTIMIZATION is False. Using default parameters for Monte Carlo.")
                pareto_front_mc = [MockOptunaTrial(default_params_dict, OPTIMIZATION_DIRECTIONS)]
            
            if pareto_front_mc and len(pareto_front_mc) > 0:
                mc_results_df = monte_carlo(df_prepared_for_mc, pareto_front_mc)

                if mc_results_df is not None and not mc_results_df.empty:
                    if 'p_values' in mc_results_df.columns and mc_results_df['p_values'].apply(lambda x: isinstance(x, dict)).all() and 'params' in mc_results_df.columns:

                        mc_results_df['p_value_sum'] = mc_results_df['p_values'].apply(
                            lambda p_dict: sum(val for val in p_dict.values() if pd.notna(val))
                        )

                        best_idx_loc = mc_results_df['p_value_sum'].idxmin()
                        best_row = mc_results_df.loc[best_idx_loc]
                        best_params_mc_flat = best_row['params'] 
                        
                        is_significant_p_lt_0_10 = any(p < 0.10 for p in best_row['p_values'].values() if pd.notna(p))
                        is_marginally_significant_p_lt_0_20 = any(0.10 <= p < 0.20 for p in best_row['p_values'].values() if pd.notna(p))

                        if is_significant_p_lt_0_10:
                            print("\n✓ Found statistically significant (p < 0.10) parameter set from Monte Carlo.")
                        elif is_marginally_significant_p_lt_0_20:
                            print("\n~ Found marginally significant (0.10 <= p < 0.20) parameter set from Monte Carlo. Proceeding with best found.")
                        else:
                            print("\n⚠ No statistically significant (p < 0.10) or marginally significant (p < 0.20) parameter sets found from Monte Carlo. Proceeding with best found.")
                        
                        print(f"\nTesting with parameters from Monte Carlo: ")
                        test(df_prepared_for_mc.copy(), params_to_test=best_params_mc_flat)
                    else:
                        print("Error: 'p_values' or 'params' column is missing or not in the expected format in mc_results_df.")
                else:
                    print("Monte Carlo analysis did not yield any results.")
            else:
                if OPTIMIZATION:
                    print("Optimization did not yield any Pareto front for Monte Carlo.")
                else:
                    print("Could not proceed with Monte Carlo using default parameters.")
        
        # -------------------------------------------------------------------------------------------------------------------
        elif TYPE == 2: # Walk-Forward Analysis
            df_prepared_is_for_wfa_opt = prepare_data(IS.copy(), type=1) 
            if df_prepared_is_for_wfa_opt is None or df_prepared_is_for_wfa_opt.empty:
                print("In-sample data for WFA initial optimization is empty after preparation. Aborting.")
                return

            pareto_front_wfa = None

            if OPTIMIZATION:
                print("Running initial optimization for Walk-Forward Analysis...")
                pareto_front_wfa = optimize(df_prepared_is_for_wfa_opt)
            else:
                print("OPTIMIZATION is False. Using default parameters for initial WFA step.")
                pareto_front_wfa = [MockOptunaTrial(default_params_dict, OPTIMIZATION_DIRECTIONS)]

            if pareto_front_wfa and len(pareto_front_wfa) > 0:
                best_trial = pareto_front_wfa[0]
                
                # Use the complete parameter set from the best trial
                current_wfa_parameters = best_trial.params.copy()
                
                # Ensure correct types for main strategy parameters
                if 'long_risk' in current_wfa_parameters:
                    current_wfa_parameters['long_risk'] = float(current_wfa_parameters['long_risk'])
                if 'max_open_positions' in current_wfa_parameters:
                    current_wfa_parameters['max_open_positions'] = int(current_wfa_parameters['max_open_positions'])
                if 'adx_threshold' in current_wfa_parameters:
                    current_wfa_parameters['adx_threshold'] = float(current_wfa_parameters['adx_threshold'])
                if 'max_position_duration' in current_wfa_parameters:
                    current_wfa_parameters['max_position_duration'] = int(current_wfa_parameters['max_position_duration'])

                # Determine the first objective's name for display
                print(f"Using optimized parameters from initial IS for WFA start.")
                
                wfa_summary = walk_forward_analysis(IS, OOS, current_wfa_parameters) 
                
                if wfa_summary:
                    print("\nAnchored Walk-Forward Analysis completed.")
                else:
                    print("Anchored Walk-Forward Analysis failed or produced no results.")
            else:
                if OPTIMIZATION:
                    print("Initial optimization for WFA failed or yielded no results.")
                else:
                    print("Could not proceed with WFA using default parameters.")
            
        # -------------------------------------------------------------------------------------------------------------------
        elif TYPE == 1: # Full Run (Opt -> MC -> WFA)
            df_prepared_full_run = prepare_data(IS.copy(), type=1)
            if df_prepared_full_run is None or df_prepared_full_run.empty:
                print("Data for Full Run (Type 1) is empty after preparation. Aborting.")
                return

            pareto_front_full_run_opt = None
            if OPTIMIZATION:
                print("Running optimization for Full Run...")
                pareto_front_full_run_opt = optimize(df_prepared_full_run)
            else:
                print("OPTIMIZATION is False. Using default parameters for Full Run optimization step.")
                pareto_front_full_run_opt = [MockOptunaTrial(default_params_dict, OPTIMIZATION_DIRECTIONS)]

            if pareto_front_full_run_opt and len(pareto_front_full_run_opt) > 0:
                mc_candidate_trials = pareto_front_full_run_opt[:3] # Use top 3 from Pareto for MC
                mc_results_df_full_run = monte_carlo(df_prepared_full_run, mc_candidate_trials)
                
                initial_params_for_wfa = None 
                
                if mc_results_df_full_run is not None and not mc_results_df_full_run.empty:
                    if 'p_values' in mc_results_df_full_run.columns and \
                       mc_results_df_full_run['p_values'].apply(lambda x: isinstance(x, dict)).all() and \
                       'params' in mc_results_df_full_run.columns:
                        
                        mc_results_df_full_run['p_value_sum'] = mc_results_df_full_run['p_values'].apply(
                            lambda p_dict: sum(val for val in p_dict.values() if pd.notna(val))
                        )
                        best_idx_loc_mc = mc_results_df_full_run['p_value_sum'].idxmin()
                        best_row_mc = mc_results_df_full_run.loc[best_idx_loc_mc]
                        best_params_from_mc_dict = best_row_mc['params'] 
                        
                        is_significant_p_lt_0_10 = any(p < 0.10 for p in best_row_mc['p_values'].values() if pd.notna(p))
                        is_marginally_significant_p_lt_0_20 = any(0.10 <= p < 0.20 for p in best_row_mc['p_values'].values() if pd.notna(p))

                        if is_significant_p_lt_0_10:
                            print("\n✓ Found statistically significant (p < 0.10) parameter set from Monte Carlo for Full Run. Using for WFA.")
                            initial_params_for_wfa = best_params_from_mc_dict.copy()
                        elif is_marginally_significant_p_lt_0_20:
                            print("\n~ Found marginally significant (0.10 <= p < 0.20) parameter set from MC for Full Run. Using for WFA.")
                            initial_params_for_wfa = best_params_from_mc_dict.copy()
                        else:
                            print("⚠ No statistically significant (p < 0.10) or marginally significant (p < 0.20) parameters from MC.")
                            # initial_params_for_wfa remains None, will trigger fallback
                    else:
                        print("Error in MC results format for Full Run. 'p_values' or 'params' column missing/invalid.")
                else:
                    print("Monte Carlo analysis for Full Run did not yield results.")

                # Fallback logic
                if initial_params_for_wfa is None: # True if MC failed, or was not significant enough
                    print("Falling back to best parameters from initial optimization for WFA.")
                    if pareto_front_full_run_opt: # Ensure Optuna results exist
                        best_optuna_trial_for_wfa = pareto_front_full_run_opt[0]
                        initial_params_for_wfa = best_optuna_trial_for_wfa.params.copy()
                    else: # This case should ideally not be reached if OPTIMIZATION=False (MockTrial)
                          # or if Optuna itself failed to produce a front.
                        print("Initial optimization also yielded no parameters. Cannot proceed with WFA.")
                
                # Common type conversion block and WFA execution
                if initial_params_for_wfa:
                    # Ensure correct types for main strategy parameters
                    if 'long_risk' in initial_params_for_wfa:
                        initial_params_for_wfa['long_risk'] = float(initial_params_for_wfa['long_risk'])
                    if 'max_open_positions' in initial_params_for_wfa:
                        initial_params_for_wfa['max_open_positions'] = int(initial_params_for_wfa['max_open_positions'])
                    if 'adx_threshold' in initial_params_for_wfa:
                        initial_params_for_wfa['adx_threshold'] = float(initial_params_for_wfa['adx_threshold'])
                    if 'max_position_duration' in initial_params_for_wfa:
                        initial_params_for_wfa['max_position_duration'] = int(initial_params_for_wfa['max_position_duration'])

                    print("\nProceeding to Walk-Forward Analysis for Full Run...")
                    wfa_summary_full_run = walk_forward_analysis(IS, OOS, initial_params_for_wfa)
                    if wfa_summary_full_run:
                        print("\nFull Run Walk-Forward Analysis completed.")
                    else:
                        print("\nFull Run Walk-Forward Analysis failed or produced no results.")
                else:
                    print("Could not determine initial parameters for WFA for Full Run.")
            else: # Optuna part failed or yielded no results
                if OPTIMIZATION:
                    print("Initial optimization for Full Run did not yield results.")
                else: 
                    print("Could not proceed with Full Run using default parameters (initial parameter setup failed).")
        
        # -------------------------------------------------------------------------------------------------------------------
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()
        return None
# -------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()