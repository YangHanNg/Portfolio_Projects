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
#================== SCRIPT PARAMETERS =======================================================================================================================
# Controls
TYPE = 3 # 1. Full run # 2. Walk-Forward # 3. Monte Carlo # 4. Optimization # 5. Test
TICKER = 'SPY'
INITIAL_CAPITAL = 25000.0
TRIALS = 35 # Number of trials for optimization
COMMISION = True # Set to True for commission
BLOCK_SIZE = 20 # Size of blocks for random sampling
OPTIMIZATION_FREQUENCY = 126 # Number of days between optimizations (wfa)
OOS_WINDOW = 42 # Number of days for out-of-sample testing (wfa)
FINAL_OOS_YEARS = 3 # Number of years for final out-of-sample testing
RISK_FREE_RATE_ANNUAL = 0.04   # Annual risk-free rate

# Optimization directions
OPTIMIZATION_DIRECTIONS = {
    'sharpe': 'maximize',
    'profit_factor': 'maximize',
    'avg_win_loss_ratio': 'maximize',
    'max_drawdown': 'minimize',
}
# Optimization objectives
OBJECTIVE_WEIGHTS = {
    'sharpe': 0.40,        
    'profit_factor': 0.25, 
    'avg_win_loss_ratio': 0.15,     
    'max_drawdown': 0.10   
}

# Optimization parameters
THRESHOLD = {
    'Entry': 0.65,
    'Exit': 0.40,
    'RSI_BUY': 50,
    'RSI_EXIT': 70,
}
ADX_THRESHOLD_DEFAULT = 25
DEFAULT_LONG_RISK = 0.05
DEFAULT_LONG_REWARD = 0.09
DEFAULT_POSITION_SIZE = 0.20
MAX_OPEN_POSITIONS = 30
PERSISTENCE_DAYS = 3
MAX_POSITION_DURATION = 15

# Other default parameters
DEFAULT_SHORT_RISK = 0.01  
DEFAULT_SHORT_REWARD = 0.03  
# Moving average strategy parameters
FAST = 20
SLOW = 50
WEEKLY_MA_PERIOD = 50
LOOKBACK_BUFFER_DAYS = 100
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
#================== DATA RETRIEVAL & HANDLING =================================================================================================================
def get_data(ticker):
    """
    Download historical data for the given ticker and split it into in-sample and out-of-sample datasets.
    """
    trading_days_per_year = 252
    oos_period_days = trading_days_per_year * FINAL_OOS_YEARS
    target_is_days = trading_days_per_year * 5
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
# --------------------------------------------------------------------------------------------------------------------------
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
        
        # Volume and Weekly Trend Confirmations
        df['volume_confirmed'] = df['Volume'] > df['Volume_MA20'].fillna(False)
        weekly_ma_series = df[f'Weekly_MA{WEEKLY_MA_PERIOD}']
        df['weekly_uptrend'] = (df['Close'] > weekly_ma_series) & \
                                (weekly_ma_series.shift(1).ffill() < weekly_ma_series).fillna(False)
        
        # Calculate raw components for momentum score
        df['price_roc_raw'] = df['Close'].pct_change(MOMENTUM_LOOKBACK).fillna(0)
        
        ma_dist_raw_values = np.where(df[f'{FAST}_ma'].values != 0, 
                                      (df['Close'].values / df[f'{FAST}_ma'].values - 1), 
                                      0)
        df['ma_dist_raw'] = pd.Series(ma_dist_raw_values, index=df.index).fillna(0)

        vol_accel_raw_values = np.where(df['Volume_MA20'].values != 0, 
                                        df['Volume'].values / df['Volume_MA20'].values, 
                                        1.0)
        df['vol_accel_raw'] = pd.Series(vol_accel_raw_values, index=df.index).fillna(1.0)
        
        df['adx_slope_raw'] = df['ADX'].diff(MOMENTUM_LOOKBACK).fillna(0)
        
        atr_pct_raw_values = np.where(df['Close'].values != 0, 
                                      df['ATR'].values / df['Close'].values, 
                                      0)
        df['atr_pct_raw'] = pd.Series(atr_pct_raw_values, index=df.index).fillna(0)

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
        df = df.iloc[-1260:].copy()
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
        df = df.iloc[-2016:].copy()
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
#================== MOMENTUM STRATEGY =========================================================================================================================
def signals(df, adx_threshold, threshold):
    
    # Initialize signals DataFrame with the same index
    signals_df = pd.DataFrame(index=df.index)
    
    # Setup signal parameters
    fast_ma_col = f"{FAST}_ma"
    slow_ma_col = f"{SLOW}_ma"
    
    # Validate threshold dictionary and apply defaults if needed
    params = ['Entry', 'Exit', 'RSI_BUY', 'RSI_EXIT']
    for param in params:
        if param in threshold:
            threshold[param] = float(threshold[param])
        else:
            print(f"Warning: {param} not found in threshold. Using default value.")
    
    # Validate required columns exist in the dataframe
    required_indicator_cols = [
        fast_ma_col, slow_ma_col, 'RSI', 'Close', 'Volume', 'High', 'Low',
        'ATR', 'ADX', 'Volume_MA20', 
        'Open', 'volume_confirmed', 'weekly_uptrend'
    ]
    
    missing_cols = [col for col in required_indicator_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for signals: {missing_cols}. Returning default signals.")
        return signals_df

    # Extract NumPy arrays for performance
    close_np = df['Close'].values
    adx_np = df['ADX'].values
    fast_ma_np = df[fast_ma_col].values
    slow_ma_np = df[slow_ma_col].values

    # Define signal conditions
    conditions = {
        # Primary trend conditions
        'trend_signal': (fast_ma_np > slow_ma_np),
        'trend_strength_ok': adx_np > adx_threshold,
        'volume_ok': df['volume_confirmed'].values.astype(bool),
        # Price confirmation
        'price_uptrend': (close_np > fast_ma_np) & (close_np > slow_ma_np),
        'weekly_uptrend': df['weekly_uptrend'].values.astype(bool),
    }

    # ---- Price Momentum (40%) ----
    # 1. Normalized ROC to avoid absolute value dependence
    df['price_roc'] = df['price_roc_raw'].rank(pct=True).fillna(0.5)

    # 2. Distance from fast MA (avoid binary crossover)
    df['ma_dist'] = df['ma_dist_raw'].rank(pct=True).fillna(0.5)

    # ---- Volume Confirmation (20%) ----
    # Volume acceleration vs. moving average
    df['vol_accel'] = df['vol_accel_raw'].rank(pct=True).fillna(0.5)

    # ---- Trend Strength (20%) ----
    # ADX slope (direction-agnostic)
    df['adx_slope'] = df['adx_slope_raw'].rank(pct=True).fillna(0.5)

    # ---- Volatility Adjustment (20%) ----
    # Scale scores down in high volatility (reduces false signals)
    df['vol_adjustment_rank'] = df['atr_pct_raw'].rolling(MOMENTUM_VOLATILITY_LOOKBACK, min_periods=1).rank(pct=True).fillna(0.5)
    df['vol_adjustment'] = (1 - df['vol_adjustment_rank']).clip(0.5, 1.5)

    # ---- Calculate Momentum Score ----
    raw_score = (
        0.4 * (df['price_roc'] * 0.6 + df['ma_dist'] * 0.4) +  # Price momentum 
        0.2 * df['vol_accel'] +                                   # Volume 
        0.2 * df['adx_slope']                                     # Trend strength
    ) 
    
    # Apply volatility adjustment
    momentum_score_values = ((raw_score * df['vol_adjustment']).clip(0, 1) * 100).values
    
    buy_signal = (
        (conditions['trend_signal']) &
        (momentum_score_values > 60) &
        (conditions['volume_ok']) & 
        (conditions['price_uptrend'] | conditions['weekly_uptrend'])
    )
    
    exit_signal = (
        (momentum_score_values < 25) |
        ((df['RSI'].values > 70) & (momentum_score_values < 45))
    )
    
    immediate_exit = (
        (momentum_score_values < 20) | 
        (adx_np < adx_threshold/3)
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
def momentum(df_with_indicators, long_risk=DEFAULT_LONG_RISK, long_reward=DEFAULT_LONG_REWARD, position_size=DEFAULT_POSITION_SIZE, risk_free_rate=RISK_FREE_RATE_ANNUAL, 
             max_positions=MAX_OPEN_POSITIONS, adx_threshold=ADX_THRESHOLD_DEFAULT, persistence_days=PERSISTENCE_DAYS, max_position_duration=MAX_POSITION_DURATION, threshold=THRESHOLD):

    if df_with_indicators.empty and df_with_indicators.index.empty:
        print("Warning: Empty dataframe (df_with_indicators) provided to momentum.")
        return [], {}, pd.Series(dtype='float64'), pd.Series(dtype='float64')

    # 1. Generate signals using the indicator-laden DataFrame
    signals_df = signals(df_with_indicators, adx_threshold=adx_threshold, threshold=threshold)

    # 2. Initialize Trade Managers and Tracking
    trade_manager = TradeManager(INITIAL_CAPITAL, max_positions)
    equity_curve = pd.Series(INITIAL_CAPITAL, index=df_with_indicators.index)
    returns_series = pd.Series(0.0, index=df_with_indicators.index)

    # Pre-allocate numpy arrays for critical data for faster access
    prices = df_with_indicators['Open'].values
    atrs = df_with_indicators['ATR'].values
    adxs = df_with_indicators['ADX'].values

    # 3. Main Processing Loop
    for i in range(1, len(df_with_indicators)):
        # Get transaction price for this step
        transaction_price = prices[i]
        current_date = df_with_indicators.index[i]

        # Process signals for this step
        signal_data = process_signals(signals_df, i, atrs[i-1], adxs[i-1])

        # --- Exit Conditions (Priority Order) ---
        if trade_manager.position_count > 0:
            # 1. Check trailing stop first
            any_trailing_stop_hit = trade_manager.trailing_stops(
                transaction_price, 
                current_date, 
                signal_data['previous_day_atr'],
                signal_data['previous_day_adx'], 
            )
            
            if not any_trailing_stop_hit:
                # 2. Check for immediate exit signal
                if signal_data['immediate_exit']:
                    trade_manager.process_exits(current_date, transaction_price, 
                                               direction_to_exit='Long', Trim=0.0, 
                                               reason_override="Immediate Exit")
                # 3. Normal exit with tiered approach
                elif signal_data['exit_signal']:
                    if signal_data['momentum_score'] > 50:
                        trim_ammount = 0.05
                        reason = "Partial Exit"
                    else:
                        trim_ammount = 0.0
                        reason = "Exit Signal"
                    
                    trade_manager.process_exits(current_date, transaction_price, 
                                                    direction_to_exit='Long', Trim=trim_ammount,
                                                    reason_override=reason)
                
            # 5. Check position health and apply exits based on health
            if trade_manager.position_count > 0 and not any_trailing_stop_hit:
                position_health = trade_manager.position_health(
                    transaction_price, 
                    signal_data['previous_day_atr'],
                    current_date,  
                    signal_data['momentum_score'],
                    max_position_duration
                )
                
        # --- Entry Conditions ---
        if (signal_data['buy_signal'] and trade_manager.position_count < max_positions):
            entry_params = {
                'price': transaction_price * (1 + 0.001),
                'portfolio_value': trade_manager.portfolio_value,
                'risk': long_risk, 
                'reward': long_reward,
                'atr': signal_data['previous_day_atr'],
                'adx': signal_data['previous_day_adx'],
                'position_size': position_size,
            }
            trade_manager.process_entry(current_date, entry_params, direction='Long')
        
        # Performance Tracking
        total_value = trade_manager.portfolio_value + trade_manager.unrealized_pnl(transaction_price)
        equity_curve.iloc[i] = max(total_value, 1e-9)  # Ensure positive for log

        # Calculate periodic log returns
        prev_equity_val = equity_curve.iloc[i-1]
        current_equity_val = equity_curve.iloc[i]
        returns_series.iloc[i] = np.log(current_equity_val / prev_equity_val)

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
        risk_free_rate
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
def trade_statistics(equity, trade_log, wins, losses, risk_free_rate=0.04):
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
    
    return {
        'Total Trades': total_trades,
        'Win Rate': win_rate,
        'Hit Rate': hit_rate,
        'Return (%)': net_profit_pct,
        'Profit Factor': profit_factor,
        'Expectancy (%)': expectancy_pct,
        'Max Drawdown (%)': max_drawdown,
        'Annualized Return (%)': annualized_return,
        'Sharpe Ratio':sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Avg Win/Loss Ratio': avg_win_loss_ratio,
        'Exit Reason Counts': exit_reason_counts
    }
#================== OPTIMIZING STRATEGY =======================================================================================================================
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
    params = {
        # Basic parameters
        'long_risk': trial.suggest_float('long_risk', 0.02, 0.10, step=0.01),
        'long_reward': trial.suggest_float('long_reward', 0.04, 0.15, step=0.01),
        'position_size': trial.suggest_float('position_size', 0.05, 0.20, step=0.01),

        # Technical parameters
        'max_open_positions': trial.suggest_int('max_open_positions', 2, 30),
        'adx_threshold': trial.suggest_float('adx_threshold', 20.0, 35.0, step=1.0),
        'persistence_days': trial.suggest_int('persistence_days', 3,10),
        'max_position_duration': trial.suggest_int('max_position_duration', 5, 30),

        # Thresholds
        'entry': trial.suggest_float('entry', 0.5, 0.7, step=0.01),
        'exit': trial.suggest_float('exit', 0.3, 0.5, step=0.01),
        'rsi_buy': trial.suggest_float('rsi_buy', 45, 55, step=1),
        'rsi_exit': trial.suggest_float('rsi_exit', 75, 90, step=1),
    }

    # Prune trials based on risk/reward ratio
    if params['long_reward'] < params['long_risk']:
        return bad_metrics_template
    
    if (params['long_reward'] / params['long_risk']) < 2.0 or (
        params['long_reward'] / params['long_risk']) > 4.0:
        return bad_metrics_template

    # Set up signal weights and threshold dict
    threshold = THRESHOLD.copy()
    threshold['Entry'] = params['entry']
    threshold['Exit'] = params['exit']
    threshold['RSI_BUY'] = params['rsi_buy']
    threshold['RSI_EXIT'] = params['rsi_exit']

    # Set up technical parameters for momentum function
    tech_params = { 
        'MAX_OPEN_POSITIONS': params['max_open_positions'],
        'ADX_THRESHOLD': params['adx_threshold'],
        'PERSISTENCE_DAYS': params['persistence_days'],
        'MAX_POSITION_DURATION': params['max_position_duration'],
        'THRESHOLD': threshold,
    }
    
    # For debugging  
    trial_num = trial.number
        
    try:
        # Direct evaluation
        trade_log, stats, equity_curve, returns_series = momentum(
            base_df,
            long_risk=params['long_risk'],
            long_reward=params['long_reward'],
            position_size=params['position_size'],
            max_positions=tech_params['MAX_OPEN_POSITIONS'],
            adx_threshold=tech_params['ADX_THRESHOLD'],
            persistence_days=tech_params['PERSISTENCE_DAYS'],
            max_position_duration=tech_params['MAX_POSITION_DURATION'],
            threshold=tech_params['THRESHOLD'],
        )
        
        # Process metrics
        required_metrics_keys = ['Sharpe Ratio', 'Profit Factor', 'Return (%)', 'Max Drawdown (%)']
        if stats and all(metric in stats for metric in required_metrics_keys):
            metrics = [
                stats['Sharpe Ratio'],
                stats['Profit Factor'],
                stats['Avg Win/Loss Ratio'],
                stats['Max Drawdown (%)']
            ]
            
            # Store additional attributes in trial
            trial.set_user_attr('sharpe', metrics[0])
            trial.set_user_attr('profit_factor', metrics[1])
            trial.set_user_attr('avg_win_loss_ratio', metrics[2])
            trial.set_user_attr('max_drawdown', metrics[3])
            trial.set_user_attr('num_trades', len(trade_log))
            
            # Store additional performance metrics
            trial.set_user_attr('avg_trade_duration', 
                np.mean([t['Duration'] for t in trade_log if t['Duration'] is not None]) if trade_log else 0)
            trial.set_user_attr('total_pnl', 
                sum(t['PnL'] for t in trade_log if t['PnL'] is not None))
            
            return metrics

        return bad_metrics_template
        
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Error in parameter evaluation for trial {trial_num}: {e}")
        return bad_metrics_template
# --------------------------------------------------------------------------------------------------------------------------
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
    print("\nOptimizing Parameters...")
    try:
        study.optimize(objective_func, n_trials=n_trials, timeout=timeout, 
                n_jobs=max(1, mp.cpu_count() - 1),
                show_progress_bar=True)
    except Exception as e:
        print(f"Optimization error: {e}")
    
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
                'avg_win_loss_ratio': trial.values[2],
                'max_drawdown': trial.values[3]
            }
            # Skip trials with invalid metrics (-inf/inf)
            if any(np.isinf(v) for v in trial.values):
                continue  
            # Calculate combined score for ranking
            combined_score = (
                OBJECTIVE_WEIGHTS['sharpe'] * metrics_dict['sharpe'] +
                OBJECTIVE_WEIGHTS['profit_factor'] * min(metrics_dict['profit_factor'], 100) +
                OBJECTIVE_WEIGHTS['avg_win_loss_ratio'] * metrics_dict['avg_win_loss_ratio'] -
                OBJECTIVE_WEIGHTS['max_drawdown'] * abs(metrics_dict['max_drawdown'])
            )
            filtered_trials.append((trial, combined_score))
    # Sort trials by combined score
    filtered_trials.sort(key=lambda x: x[1], reverse=True)
    # Extract just the trials for display
    pareto_front = [trial_tuple[0] for trial_tuple in filtered_trials]
    return pareto_front[:10]  # Return top 10 trialscl
#================== VIEW STRATEGY =======================================================================================================================
def visualize(pareto_front, base_df):
    while True:
        # Extract trial metrics for display
        trial_metrics = []
        for i, trial in enumerate(pareto_front, 1):
            metrics = {
                'Trial': i,
                'Sharpe': f"{trial.values[0]:.2f}",
                'ProfitFactor': f"{trial.values[1]:.2f}",
                'AVGWINL(%)': f"{trial.values[2]:.1f}",
                'MaxDD(%)': f"{abs(trial.values[3]):.1f}",
                'Trades': trial.user_attrs.get('num_trades', 0)
            }
            trial_metrics.append(metrics)

        # Display trials table
        print("\n=== Optimization Results ===")
        print(tabulate(
            trial_metrics,
            headers='keys',
            tablefmt='grid',
            floatfmt='.2f'
        ))

        # Get user input
        try:
            choice = input("\nEnter trial number to test (or 'exit' to quit): ").strip().lower()
            
            if choice == 'exit':
                break
            
            trial_num = int(choice) - 1
            if 0 <= trial_num < len(pareto_front):
                selected_trial = pareto_front[trial_num]
                
                # Extract parameters from selected trial
                params = {
                    'long_risk': float(selected_trial.params['long_risk']),
                    'long_reward': float(selected_trial.params['long_reward']),
                    'position_size': float(selected_trial.params['position_size']),
                    'max_positions_param': int(selected_trial.params['max_open_positions']),
                    'adx_thresh': float(selected_trial.params['adx_threshold']),
                    'persistence_days': int(selected_trial.params['persistence_days']),
                    'max_position_duration': int(selected_trial.params['max_position_duration']),
                    'threshold': {
                        'Entry': float(selected_trial.params['entry']),
                        'Exit': float(selected_trial.params['exit']),
                        'RSI_BUY': float(selected_trial.params['rsi_buy']),
                        'RSI_EXIT': float(selected_trial.params['rsi_exit'])
                    }
                }

                # Run test with selected parameters
                print(f"\nTesting Trial {trial_num + 1} Parameters...")
                test(
                    base_df,
                    long_risk=params['long_risk'],
                    long_reward=params['long_reward'],
                    position_size=params['position_size'],
                    max_positions_param=params['max_positions_param'],
                    adx_thresh=params['adx_thresh'],
                    persistence_days=params['persistence_days'],
                    max_position_duration=params['max_position_duration'],
                    threshold=params['threshold']
                )

                input("\nPress Enter to return to trial selection...")
                
            else:
                print(f"Invalid trial number. Please select 1-{len(pareto_front)}")
                
        except ValueError:
            print("Invalid input. Please enter a number or 'exit'")
        except Exception as e:
            print(f"Error: {e}")

    return None
# --------------------------------------------------------------------------------------------------------------------------
def test(df_input, long_risk=DEFAULT_LONG_RISK, long_reward=DEFAULT_LONG_REWARD, position_size=DEFAULT_POSITION_SIZE, max_positions_param=MAX_OPEN_POSITIONS,
         adx_thresh=ADX_THRESHOLD_DEFAULT, persistence_days=PERSISTENCE_DAYS, max_position_duration=MAX_POSITION_DURATION, threshold=THRESHOLD): 
    
    df = df_input.copy()
    
    trade_log, stats, equity_curve, returns_series = momentum(
        df_input, long_risk=long_risk, long_reward=long_reward, position_size=position_size, risk_free_rate=0.04, max_positions=max_positions_param,
        adx_threshold=adx_thresh, persistence_days=persistence_days, max_position_duration=max_position_duration, threshold=threshold)

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
    print(f"Long Risk: {long_risk*100:.1f}% | Long Reward: {long_reward*100:.1f}%")
    print(f"Position Size: {position_size*100:.1f}% | Max Open Positions: {max_positions_param}")
    print(f"ADX Threshold: {adx_thresh:.1f} | Persistence: {persistence_days} days | Max Duration: {max_position_duration} days")
    print(f"Signal Weights: {threshold}")
    
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
        ["Total Commission Paid [$]", f"{total_commission_paid:,.2f}"]
    ]
    print(tabulate(trade_metrics, tablefmt="simple", colalign=("left", "right")))
     
    return None
#================== STRATEGY SIGNIFICANCE TESTING ==============================================================================================================
def stationary_bootstrap(data, block_size = BLOCK_SIZE, num_samples= 1000, sample_length = None, seed = None):
    n = len(data)
    if sample_length is None:
        sample_length = n
    
    # Generate all random indices at once
    p = 1.0 / block_size
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
            bootstrap_sample.index = data.index[:len(bootstrap_sample)]
            batch_samples.append(bootstrap_sample)
        
        bootstrap_samples.extend(batch_samples)
        
    return bootstrap_samples
# --------------------------------------------------------------------------------------------------------------------------  
def monte_carlo(prepared_data, pareto_front, num_simulations=1000):
    """Monte Carlo analysis with improved statistical visualization"""
    
    mc_results = []
    total_iterations = len(pareto_front)
    
    # Convert trial parameters to proper format once
    param_sets = []
    for trial in pareto_front:
        params = {
            'long_risk': float(trial.params['long_risk']),
            'long_reward': float(trial.params['long_reward']),
            'position_size': float(trial.params['position_size']),
            'max_positions': int(trial.params['max_open_positions']),
            'adx_threshold': float(trial.params['adx_threshold']),
            'persistence_days': int(trial.params['persistence_days']),
            'max_position_duration': int(trial.params['max_position_duration']),
            'threshold': {
                'Entry': float(trial.params['entry']),
                'Exit': float(trial.params['exit']),
                'RSI_BUY': float(trial.params['rsi_buy']),
                'RSI_EXIT': float(trial.params['rsi_exit'])
            }
        }
        param_sets.append(params)
    
    # Generate bootstrap samples once
    print("\nGenerating bootstrap samples...")
    bootstrap_samples = stationary_bootstrap(
        data=prepared_data,
        block_size=BLOCK_SIZE,
        num_samples=num_simulations,
        sample_length=None,
        seed=42
    )

    def process_parameter_set(param_idx):
        """Process a single parameter set"""
        params = param_sets[param_idx]
        
        # Run original strategy to get baseline performance
        trade_log, observed_stats, _, _ = momentum(
            prepared_data,
            long_risk=params['long_risk'],
            long_reward=params['long_reward'],
            position_size=params['position_size'],
            max_positions=params['max_positions'],
            adx_threshold=params['adx_threshold'],
            persistence_days=params['persistence_days'],
            max_position_duration=params['max_position_duration'],
            threshold=params['threshold']
        )
        
        # Define a mapping from internal keys (used in this function) to original stat keys
        metric_key_map = {
            'sharpe': 'Sharpe Ratio',
            'profit_factor': 'Profit Factor',
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

        # Initialize arrays for simulation metrics using internal keys
        sim_metrics = {internal_key: [] for internal_key in metric_key_map.keys()}
    
        num_bootstrap_samples = len(bootstrap_samples)
        for sample_idx, sample in enumerate(tqdm.tqdm(bootstrap_samples, total=num_bootstrap_samples, desc=f"Sims for Set {param_idx+1}", leave=False, position=(param_idx % (mp.cpu_count() if mp.cpu_count() > 0 else 1)) )):
            _, sim_stats_run, _, _ = momentum(sample, **params)
            
            for internal_key, original_key in metric_key_map.items():
                if original_key in sim_stats_run:
                    sim_metrics[internal_key].append(sim_stats_run[original_key])
                else:
                    sim_metrics[internal_key].append(np.nan)
        
        # Calculate p-values and simulation statistics
        results = {
            'parameter_set': param_idx + 1,
            'params': params,
            'p_values': {},
            'percentiles': {},
            'observed_metrics': observed_metrics,
            'simulation_metrics': {}
        }
        
        for internal_key in sim_metrics:
            sim_array_raw = np.array(sim_metrics[internal_key], dtype=float)
            # Filter out NaNs that might have been appended if keys were missing
            sim_array = sim_array_raw[~np.isnan(sim_array_raw)]

            observed_value = observed_metrics.get(internal_key, np.nan)

            if np.isnan(observed_value) or len(sim_array) == 0:
                results['p_values'][internal_key] = np.nan
                results['percentiles'][internal_key] = np.nan
                results['simulation_metrics'][internal_key] = {
                    'mean': np.nan, 'std': np.nan, 'skew': np.nan, 'kurtosis': np.nan,
                    'p5': np.nan, 'p95': np.nan  # Adding percentiles
                }
                continue
            
            # Calculate p-value (proportion of simulations better than observed)
            if internal_key in ['sharpe', 'profit_factor', 'avg_win_loss_ratio']:
                p_value = np.mean(sim_array >= observed_value)
            else:  # max_drawdown - lower is better
                p_value = np.mean(sim_array <= observed_value)
            
            # Calculate percentile of observed value
            percentile = stats.percentileofscore(sim_array, observed_value)
            
            # Calculate percentiles for null distribution
            p5 = np.percentile(sim_array, 5) if len(sim_array) > 20 else np.nan
            p95 = np.percentile(sim_array, 95) if len(sim_array) > 20 else np.nan
            
            results['p_values'][internal_key] = p_value
            results['percentiles'][internal_key] = percentile
            results['simulation_metrics'][internal_key] = {
                'mean': np.mean(sim_array),
                'std': np.std(sim_array),
                'skew': stats.skew(sim_array),
                'kurtosis': stats.kurtosis(sim_array),
                'p5': p5,
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
    print(f"   - Avg Block Length: {BLOCK_SIZE}")
    
    # Additional data for summary
    total_obs_metrics = {}
    for metric in ['sharpe', 'profit_factor', 'avg_win_loss_ratio', 'max_drawdown']:
        values = [result['observed_metrics'][metric] for result in mc_results 
                  if pd.notna(result['observed_metrics'][metric])]
        if values:
            total_obs_metrics[metric] = np.mean(values)
    
    if 'sharpe' in total_obs_metrics and 'max_drawdown' in total_obs_metrics:
        print(f"   - Avg Observed Sharpe: {total_obs_metrics['sharpe']:.2f} | " + 
              f"Max DD: {total_obs_metrics['max_drawdown']:.2f}%")
    
    # Get significant metric counts
    sig_counts = {}
    for metric in ['sharpe', 'profit_factor', 'avg_win_loss_ratio', 'max_drawdown']:
        sig_counts[metric] = sum(1 for result in mc_results 
                                 if result['p_values'].get(metric, 1.0) < 0.05)
    
    print(f"   - Significant Results (p<0.05): " + 
          f"Sharpe: {sig_counts['sharpe']}/{len(mc_results)}, " +
          f"PF: {sig_counts['profit_factor']}/{len(mc_results)}")
    
    # 2-5. Null Distribution tables for each key metric
    metric_display_names = {
        'sharpe': 'Sharpe Ratio',
        'profit_factor': 'Profit Factor', 
        'avg_win_loss_ratio': 'Win/Loss Ratio',
        'max_drawdown': 'Max Drawdown (%)'
    }
    
    # For each metric, create a separate table
    for metric_num, metric in enumerate(['sharpe', 'profit_factor', 'avg_win_loss_ratio', 'max_drawdown'], 2):
        print(f"\n{metric_num}. Null Distribution - {metric_display_names[metric]}:")
        
        # Create table data for this metric
        table_data = []
        headers = ['Set', 'Observed', 'Mean', 'Std Dev', '5th %ile', '95th %ile', 'p-value']
        
        for result in mc_results:
            param_set = result['parameter_set']
            observed = result['observed_metrics'].get(metric, np.nan)
            sim_stats = result['simulation_metrics'].get(metric, {})
            
            if not sim_stats or np.isnan(observed):
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
        best_set_idx = np.argmin([sum(res['p_values'].values()) for res in mc_results])
        best_set = mc_results[best_set_idx]['parameter_set']
        
        print(f"\nBest Overall Parameter Set: #{best_set}")
        print("Key Performance Metrics:")
        
        # Format best set metrics nicely
        best_metrics = []
        for metric in ['sharpe', 'profit_factor', 'avg_win_loss_ratio', 'max_drawdown']:
            observed = mc_results[best_set_idx]['observed_metrics'].get(metric, np.nan)
            p_val = mc_results[best_set_idx]['p_values'].get(metric, np.nan)
            sim_mean = mc_results[best_set_idx]['simulation_metrics'].get(metric, {}).get('mean', np.nan)
            
            # Mark significant metrics with *
            sig_marker = '*' if pd.notna(p_val) and p_val < 0.05 else ''
            
            best_metrics.append([
                f"{metric_display_names[metric]}{sig_marker}", 
                f"{observed:.2f}",
                f"{sim_mean:.2f}",
                f"{observed - sim_mean:.2f}" if pd.notna(observed) and pd.notna(sim_mean) else "N/A"
            ])
        
        print(tabulate(best_metrics, headers=['Metric', 'Observed', 'Sim Mean', 'Edge'], tablefmt='simple'))
    
    return results_df
#================== STRATEGY ROBUSTNESS TESTING ================================================================================================================
def walk_forward_analysis(initial_is_data_raw, full_oos_data_raw, initial_parameters, risk_free_annual=RISK_FREE_RATE_ANNUAL):
    
    # Global constants from your script parameters
    oos_window_size = OOS_WINDOW  # e.g., 10 (days/periods in raw data)
    opt_frequency_days = OPTIMIZATION_FREQUENCY # e.g., 42 (days/periods in raw data)
    daily_rf_rate = risk_free_annual / 252.0 # Daily risk-free rate for Sharpe ratio calculation

    # Make copies to avoid modifying originals passed to the function
    current_train_raw = initial_is_data_raw.copy()
    remaining_oos_raw = full_oos_data_raw.copy()
    oos_length = len(remaining_oos_raw)

    # Combine datasets and prepare together
    full_data = pd.concat([current_train_raw, remaining_oos_raw])
    prepared_data = prepare_data(full_data.copy(), type=2)
    
    if prepared_data.empty:
        print("Combined data is empty after preparation. Aborting WFA.")
        return None
    
    # Split back into IS and OOS based on original sizes
    is_size = len(current_train_raw)
    prepared_current_train = prepared_data.iloc[:is_size].copy()
    prepared_remaining_oos = prepared_data.iloc[is_size:].copy()
    
    if prepared_current_train.empty:
        print("Initial training data is empty after preparation. Aborting WFA.")
        return None
        
    active_parameters = initial_parameters.copy() # Current parameters for the strategy
    all_step_results = []
    
    # For re-optimization tracking
    days_processed_since_last_opt = 0 

    step_number = 0
    print(f"\nStarting Anchored Walk-Forward Analysis...")
    if not current_train_raw.empty:
        print(f"Initial training data: {current_train_raw.index[0].date()} to {current_train_raw.index[-1].date()} ({len(current_train_raw)} days)")
    else:
        print("Initial training data: Empty")
    print(f"Total OOS data available: {len(remaining_oos_raw)} days")
    print(f"OOS Window Size: {oos_window_size} days, Optimization Frequency: {opt_frequency_days} days of new OOS data.")

    while len(prepared_remaining_oos) >= oos_window_size:
        step_number += 1
        print(f"\n--- WFA Step {step_number} ---")

        # 1. Re-optimization Check
        if days_processed_since_last_opt >= opt_frequency_days and step_number > 1:
            #print(f"Attempting re-optimization. Days processed since last opt: {days_processed_since_last_opt} >= {opt_frequency_days}")
            
            if prepared_current_train.empty:
                print("Cannot optimize: Current accumulated training data is empty after preparation.")
            else:
                pareto_front = optimize(prepared_current_train) 
                
                if pareto_front and len(pareto_front) > 0:
                    best_trial = pareto_front[0] 
                    active_parameters = {
                        'long_risk': float(best_trial.params['long_risk']),
                        'long_reward': float(best_trial.params['long_reward']),
                        'position_size': float(best_trial.params['position_size']),
                        'max_positions_param': int(best_trial.params['max_open_positions']),
                        'adx_thresh': float(best_trial.params['adx_threshold']),
                        'persistence_days': int(best_trial.params['persistence_days']),
                        'max_position_duration': int(best_trial.params['max_position_duration']),
                        'threshold': {
                            'Entry': float(best_trial.params['entry']),
                            'Exit': float(best_trial.params['exit']),
                            'RSI_BUY': float(best_trial.params['rsi_buy']),
                            'RSI_EXIT': float(best_trial.params['rsi_exit'])
                        }
                    }
                    #print(f"Parameters updated from optimization. New Sharpe from trial: {best_trial.values[0]:.2f}")
                else:
                    print("Optimization did not yield new parameters. Continuing with existing ones.")
            days_processed_since_last_opt = 0 
        
        # 2. Define and Prepare Current Test Window (OOS Chunk)
        prepared_oos_chunk = prepared_remaining_oos.iloc[:oos_window_size].copy()
        raw_oos_chunk = remaining_oos_raw.iloc[:oos_window_size].copy()

        if prepared_oos_chunk.empty:
            print(f"OOS chunk for step {step_number} is empty after preparation. Skipping this OOS chunk.")
            prepared_remaining_oos = prepared_remaining_oos.iloc[oos_window_size:].copy()
            remaining_oos_raw = remaining_oos_raw.iloc[oos_window_size:].copy()
            days_processed_since_last_opt += len(raw_oos_chunk) 
            continue

        #print(f"Training data for this step: {prepared_current_train.index[0].date()} to {prepared_current_train.index[-1].date()} ({len(prepared_current_train)} days)")
        #print(f"Testing with OOS data: {prepared_oos_chunk.index[0].date()} to {prepared_oos_chunk.index[-1].date()} ({len(prepared_oos_chunk)} days)")

        # 3. Run strategy on the current prepared training data (for reference)
        train_log, train_stats, train_equity, train_returns = momentum(
            prepared_current_train, 
            long_risk=active_parameters['long_risk'],
            long_reward=active_parameters['long_reward'],
            position_size=active_parameters['position_size'],
            max_positions=active_parameters['max_positions_param'], 
            adx_threshold=active_parameters['adx_thresh'],        
            persistence_days=active_parameters['persistence_days'],
            max_position_duration=active_parameters['max_position_duration'],
            threshold=active_parameters['threshold'],
            risk_free_rate=risk_free_annual
        )

        # 4. Run strategy on the prepared OOS chunk (current test window)
        oos_log, oos_stats, oos_equity, oos_returns = momentum(
            prepared_oos_chunk,
            long_risk=active_parameters['long_risk'],
            long_reward=active_parameters['long_reward'],
            position_size=active_parameters['position_size'],
            max_positions=active_parameters['max_positions_param'],
            adx_threshold=active_parameters['adx_thresh'],
            persistence_days=active_parameters['persistence_days'],
            max_position_duration=active_parameters['max_position_duration'],
            threshold=active_parameters['threshold'],
            risk_free_rate=risk_free_annual
        )
        
        # 5. Record results for this step
        oos_trade_count = oos_stats.get('Total Trades', 0)
        train_trade_count = train_stats.get('Total Trades', 0)
        
        # Get key metrics for tracking (added for improved tracking)
        train_sharpe = train_stats.get('Sharpe Ratio', np.nan)
        oos_sharpe = oos_stats.get('Sharpe Ratio', np.nan)
        
        # Additional metrics to track (added)
        train_annualized_return = train_stats.get('Annualized Return (%)', np.nan)
        oos_annualized_return = oos_stats.get('Annualized Return (%)', np.nan)
        
        train_max_drawdown = train_stats.get('Max Drawdown (%)', np.nan)
        oos_max_drawdown = oos_stats.get('Max Drawdown (%)', np.nan)
        
        train_win_rate = train_stats.get('Win Rate', np.nan)
        oos_win_rate = oos_stats.get('Win Rate', np.nan)
        
        train_profit_factor = train_stats.get('Profit Factor', np.nan)
        oos_profit_factor = oos_stats.get('Profit Factor', np.nan)
        
        # Flag for periods with no trading activity
        is_valid_train_period = train_trade_count > 0
        is_valid_oos_period = oos_trade_count > 0
        
        if not is_valid_oos_period:
            print(f"  WARNING: No trades executed in OOS period (Step {step_number}). Using neutral Sharpe (0.0).")
            oos_sharpe = 0.0
        
        if not is_valid_train_period:
            print(f"  WARNING: No trades executed in training period (Step {step_number}). Using neutral Sharpe (0.0).")
            train_sharpe = 0.0
        
        # Calculate decay with proper handling
        if is_valid_train_period and is_valid_oos_period:
            if (train_sharpe >= 0 and oos_sharpe >= 0) or (train_sharpe < 0 and oos_sharpe < 0):
                performance_ratio = abs(oos_sharpe) / max(abs(train_sharpe), 0.1)
                decay_ratio_val = 1.0 - performance_ratio
            else:
                decay_ratio_val = 1.5
        else:
            decay_ratio_val = np.nan
        
        # Create comprehensive step results
        step_result_data = {
            'step': step_number,
            'train_start_date': prepared_current_train.index[0].date() if not prepared_current_train.empty else None,
            'train_end_date': prepared_current_train.index[-1].date() if not prepared_current_train.empty else None,
            'train_days': len(prepared_current_train),
            'test_start_date': prepared_oos_chunk.index[0].date() if not prepared_oos_chunk.empty else None,
            'test_end_date': prepared_oos_chunk.index[-1].date() if not prepared_oos_chunk.empty else None,
            'test_days': len(prepared_oos_chunk),
            
            # Original metrics
            'train_sharpe': train_sharpe,
            'test_sharpe': oos_sharpe,
            'decay_ratio': decay_ratio_val,
            'train_trades': train_trade_count,
            'test_trades': oos_trade_count,
            'oos_returns_series': oos_returns, 
            'train_return_pct': train_stats.get('Return (%)', np.nan),
            'test_return_pct': oos_stats.get('Return (%)', np.nan),
            'valid_train': is_valid_train_period,
            'valid_test': is_valid_oos_period,
            'valid_comparison': is_valid_train_period and is_valid_oos_period,
            
            # Additional metrics (added)
            'train_ann_return': train_annualized_return,
            'test_ann_return': oos_annualized_return,
            'train_max_drawdown': train_max_drawdown,
            'test_max_drawdown': oos_max_drawdown,
            'train_win_rate': train_win_rate,
            'test_win_rate': oos_win_rate,
            'train_profit_factor': train_profit_factor,
            'test_profit_factor': oos_profit_factor,
            
            # Parameter snapshot
            'parameters_used_snapshot': active_parameters.copy()
        }
        all_step_results.append(step_result_data)
        
        # Display step results
        decay_ratio_str = f"{decay_ratio_val:.2f}" if pd.notna(decay_ratio_val) else 'N/A'
        print(f"  Step {step_number} Results: Train Sharpe: {train_sharpe:.2f}, Test Sharpe: {oos_sharpe:.2f}, Decay Ratio: {decay_ratio_str}")

        # 6. Update data for the next iteration (Anchoring)
        # Update prepared data by appending the current OOS chunk to training data
        prepared_current_train = pd.concat([prepared_current_train, prepared_oos_chunk])
        
        # Update raw data tracking for consistency
        current_train_raw = pd.concat([current_train_raw, raw_oos_chunk])
        
        # Remove the processed chunk from remaining OOS data
        prepared_remaining_oos = prepared_remaining_oos.iloc[oos_window_size:].copy()
        remaining_oos_raw = remaining_oos_raw.iloc[oos_window_size:].copy()
        
        # Update days processed counter
        days_processed_since_last_opt += len(raw_oos_chunk)

    if not all_step_results:
        print("No walk-forward steps were completed.")
        return None

    # Create results DataFrame
    results_df = pd.DataFrame(all_step_results)
    
    # Process all OOS returns for overall metrics
    all_oos_log_returns_list = [res['oos_returns_series'] for res in all_step_results 
                               if isinstance(res['oos_returns_series'], pd.Series) and not res['oos_returns_series'].empty]
    
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
    
    # Print the new formatted summary
    print("\n=== WFA FINAL SUMMARY ===")
    
    # 1. Overview section
    print("1. Overview:")
    if total_steps > 0:
        print(f"   - Total WFA Steps: {total_steps}")
        print(f"   - Valid OOS Windows: {valid_tests_count}/{total_steps} ({valid_tests_count/total_steps*100:.1f}%)")
        if zero_trade_count > 0:
            zero_trade_pct = zero_trade_count/total_steps*100
            risk_level = "HIGH RISK" if zero_trade_pct > 40 else "MODERATE RISK" if zero_trade_pct > 20 else "LOW RISK"
            print(f"   - Zero-Trade Windows: {zero_trade_count}/{total_steps} ({zero_trade_pct:.1f}%) → {risk_level}")
    else:
        print("   - No WFA steps completed")
    
    # 2. Performance metrics table
    print("\n2. Performance (Valid OOS):")
    if not valid_test_results_df.empty:
        # Create a summary table for OOS metrics
        metrics = {
            'OOS Sharpe': valid_test_results_df['test_sharpe'],
            'Ann. Return (%)': valid_test_results_df['test_ann_return'],
            'Max Drawdown (%)': valid_test_results_df['test_max_drawdown'],
            'Win Rate (%)': valid_test_results_df['test_win_rate'],
            'Profit Factor': valid_test_results_df['test_profit_factor']
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
                tablefmt='grid'
            ))
        else:
            print("   No valid metrics data available")
    else:
        print("   No valid OOS periods with trades for performance analysis")
    
    # 3. Decay Analysis
    print("\n3. Decay Analysis:")
    if not valid_comparisons_df.empty:
        decay_values = valid_comparisons_df['decay_ratio'].dropna()
        if not decay_values.empty:
            mean_decay = decay_values.mean()
            std_decay = decay_values.std()
            
            # Classify decay ratio
            decay_classification = ""
            if mean_decay > 0.7:
                decay_classification = "CATASTROPHIC"
            elif mean_decay > 0.5:
                decay_classification = "SEVERE"
            elif mean_decay > 0.3:
                decay_classification = "SIGNIFICANT" 
            elif mean_decay > 0.1:
                decay_classification = "MODERATE"
            else:
                decay_classification = "MINIMAL"
                
            # Classify volatility
            vol_classification = ""
            if std_decay > 0.4:
                vol_classification = "EXTREME VOLATILITY"
            elif std_decay > 0.2:
                vol_classification = "HIGH VOLATILITY" 
            elif std_decay > 0.1:
                vol_classification = "MODERATE VOLATILITY"
            else:
                vol_classification = "STABLE"
                
            print(f"   - Mean Decay Ratio: {mean_decay:.2f} → {decay_classification}")
            print(f"   - Std Dev Decay: {std_decay:.2f} → {vol_classification}")
            
            if rolling_decay is not None and len(rolling_decay.dropna()) > 0:
                recent_decay = rolling_decay.dropna().iloc[-1] if len(rolling_decay.dropna()) > 0 else np.nan
                if pd.notna(recent_decay):
                    print(f"   - Recent Rolling Decay (n={min(3, len(decay_values))}): {recent_decay:.2f}")
        else:
            print("   - No valid decay ratio data available")
    else:
        print("   - No valid comparison periods to calculate decay metrics")
    
    # 4. Activity metrics
    print("\n4. Activity:")
    if total_days_in_oos_concat > 0:
        trading_activity_pct = active_trading_days/total_days_in_oos_concat*100
        print(f"   - Trading Days: {active_trading_days}/{total_days_in_oos_concat} ({trading_activity_pct:.1f}%)")
        
        # Identify potential causes for low activity
        if trading_activity_pct < 30:
            # Analyze parameters to suggest causes
            if not results_df.empty and 'parameters_used_snapshot' in results_df.columns:
                entry_thresholds = []
                for _, row in results_df.iterrows():
                    if isinstance(row['parameters_used_snapshot'], dict):
                        threshold = row['parameters_used_snapshot'].get('threshold', {})
                        if isinstance(threshold, dict) and 'Entry' in threshold:
                            entry_thresholds.append(threshold['Entry'])
                
                if entry_thresholds:
                    avg_entry = np.mean(entry_thresholds)
                    if avg_entry > 0.8:
                        print("   - Zero-Trade Cause: Restrictive entry thresholds (too high)")
                    elif all(t.get('adx_thresh', 25) > 30 for t in results_df['parameters_used_snapshot'] if isinstance(t, dict)):
                        print("   - Zero-Trade Cause: High ADX threshold requirements")
                    else:
                        print("   - Zero-Trade Cause: Overfit parameter combination")
        elif zero_trade_count > 0:
            print("   - Zero-Trade Cause: Parameter instability between periods")
    else:
        print("   - No trading activity data available")
    
    # 5. Concatenated OOS Performance
    print("\n5. Concatenated OOS:")
    if len(concatenated_oos_log_returns) > 5:
        print(f"   - Ann. Sharpe: {overall_oos_sharpe:.2f} | Cum. Return: {overall_oos_cumulative_return_pct:.2f}% | Max DD: {overall_max_drawdown:.1f}%")
        
        # Market classification based on overall performance
        market_type = ""
        if overall_oos_sharpe > 1.5:
            market_type = "HIGHLY FAVORABLE"
        elif overall_oos_sharpe > 0.5:
            market_type = "FAVORABLE" 
        elif overall_oos_sharpe > -0.5:
            market_type = "NEUTRAL"
        elif overall_oos_sharpe > -1.5:
            market_type = "CHALLENGING"
        else:
            market_type = "HIGHLY ADVERSE"
            
        print(f"   - Market Classification: {market_type}")
    else:
        print("   - Insufficient data for reliable OOS performance metrics")
    
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
            print(tabulate(results_df[available_cols], headers='keys', tablefmt='grid', floatfmt=".2f"))
    
    # Return comprehensive results
    return {
        'step_results_df': results_df, 
        'concatenated_oos_returns': concatenated_oos_log_returns, 
        'overall_oos_sharpe': overall_oos_sharpe,
        'overall_oos_return_pct': overall_oos_cumulative_return_pct,
        'overall_oos_max_drawdown': overall_max_drawdown,
        'rolling_decay': rolling_decay if rolling_decay is not None else None
    }
#================== MAIN PROGRAM EXECUTION =====================================================================================================================
def main():
    try:
        IS, OOS = get_data(TICKER)

        # --------------------------------------------------------------------------------------------------------------------------
        if TYPE == 5:
            df_prepared_for_test = prepare_data(IS.copy(), type=1)
            if df_prepared_for_test is None or df_prepared_for_test.empty:
                print("Data for test is empty after preparation. Aborting.")
                return
            test(df_prepared_for_test, long_risk=DEFAULT_LONG_RISK, long_reward=DEFAULT_LONG_REWARD, position_size=DEFAULT_POSITION_SIZE, max_positions_param=MAX_OPEN_POSITIONS,
                adx_thresh=ADX_THRESHOLD_DEFAULT, persistence_days=PERSISTENCE_DAYS, max_position_duration=MAX_POSITION_DURATION, threshold=THRESHOLD)
        
        # --------------------------------------------------------------------------------------------------------------------------
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

        # --------------------------------------------------------------------------------------------------------------------------
        elif TYPE == 3:  # Monte Carlo Testing
            df_prepared_for_mc = prepare_data(IS.copy(), type=1)
            if df_prepared_for_mc is None or df_prepared_for_mc.empty:
                print("Data for Monte Carlo is empty after preparation. Aborting.")
                return
            pareto_front_mc = optimize(df_prepared_for_mc)[:3] # Optimize first
            if pareto_front_mc and len(pareto_front_mc) > 0:
                mc_results_df = monte_carlo(df_prepared_for_mc, pareto_front_mc)
                if mc_results_df is not None and not mc_results_df.empty:
                    # Find best parameter set (lowest combined p-value)
                    if 'p_values' in mc_results_df.columns and mc_results_df['p_values'].apply(lambda x: isinstance(x, dict)).all():
                        best_idx_loc = mc_results_df.apply(
                            lambda x: sum(val for val in x['p_values'].values() if pd.notna(val)), axis=1
                        ).idxmin()
                        best_row = mc_results_df.loc[best_idx_loc]
                        best_params_mc = best_row['params'] # These are already in the correct flat structure from MC
                        
                        if any(p < 0.05 for p in best_row['p_values'].values() if pd.notna(p)):
                            print("\n✓ Found statistically significant parameter set from Monte Carlo.")
                            # Parameters from mc_results_df['params'] are already structured for momentum
                            test_params_from_mc = {
                                'long_risk': best_params_mc['long_risk'],
                                'long_reward': best_params_mc['long_reward'],
                                'position_size': best_params_mc['position_size'],
                                'max_positions_param': best_params_mc['max_positions'], # Key from MC
                                'adx_thresh': best_params_mc['adx_threshold'],       # Key from MC
                                'persistence_days': best_params_mc['persistence_days'],
                                'max_position_duration': best_params_mc['max_position_duration'],
                                'threshold': best_params_mc['threshold'] 
                            }
                            test(df_prepared_for_mc, **test_params_from_mc)
                        else:
                            print("⚠ No statistically significant parameter sets found after Monte Carlo.")
                    else:
                        print("Error: 'p_values' column is missing or not in the expected format in mc_results_df.")
                else:
                    print("Monte Carlo analysis did not yield any results.")
            else:
                print("Optimization did not yield any Pareto front for Monte Carlo.")
        
        # --------------------------------------------------------------------------------------------------------------------------
        elif TYPE == 2: # Walk-Forward Analysis
            df_prepared_is_for_wfa_opt = prepare_data(IS.copy(), type=1) # Prepare IS data for initial optimization
            if df_prepared_is_for_wfa_opt is None or df_prepared_is_for_wfa_opt.empty:
                print("In-sample data for WFA initial optimization is empty after preparation. Aborting.")
                return

            pareto_front_wfa = optimize(df_prepared_is_for_wfa_opt)
            if pareto_front_wfa and len(pareto_front_wfa) > 0:
                best_trial = pareto_front_wfa[0]  # Assuming the first trial is the best
                
                # Correctly access parameters from the Optuna FrozenTrial object's .params attribute
                current_wfa_parameters = {
                    'long_risk': float(best_trial.params['long_risk']),
                    'long_reward': float(best_trial.params['long_reward']),
                    'position_size': float(best_trial.params['position_size']),
                    'max_positions_param': int(best_trial.params['max_open_positions']), # Key from Optuna trial
                    'adx_thresh': float(best_trial.params['adx_threshold']),         # Key from Optuna trial
                    'persistence_days': int(best_trial.params['persistence_days']),
                    'max_position_duration': int(best_trial.params['max_position_duration']),
                    'threshold': { # Reconstruct threshold dict from Optuna trial params
                        'Entry': float(best_trial.params['entry']),
                        'Exit': float(best_trial.params['exit']),
                        'RSI_BUY': float(best_trial.params['rsi_buy']),
                        'RSI_EXIT': float(best_trial.params['rsi_exit'])
                    }
                }
                print(f"Using optimized parameters from initial IS for WFA start. Sharpe: {best_trial.values[0]:.2f}")
                # walk_forward_analysis expects raw IS and OOS data
                wfa_summary = walk_forward_analysis(IS, OOS, current_wfa_parameters) 
                
                if wfa_summary:
                    print("\nAnchored Walk-Forward Analysis completed.")
                else:
                    print("Anchored Walk-Forward Analysis failed or produced no results.")
            else:
                print("Initial optimization for WFA failed or yielded no results.")
            
        # --------------------------------------------------------------------------------------------------------------------------
        elif TYPE == 1: # Full Run (Opt -> MC -> WFA)
            df_prepared_full_run = prepare_data(IS.copy(), type=1)
            if df_prepared_full_run is None or df_prepared_full_run.empty:
                print("Data for Full Run (Type 1) is empty after preparation. Aborting.")
                return

            pareto_front_full_run_opt = optimize(df_prepared_full_run) 

            if pareto_front_full_run_opt and len(pareto_front_full_run_opt) > 0:
                mc_candidate_trials = pareto_front_full_run_opt[:3]
                mc_results_df_full_run = monte_carlo(df_prepared_full_run, mc_candidate_trials)
                
                if mc_results_df_full_run is not None and not mc_results_df_full_run.empty:
                    if 'p_values' in mc_results_df_full_run.columns and mc_results_df_full_run['p_values'].apply(lambda x: isinstance(x, dict)).all():
                        best_idx_loc_mc = mc_results_df_full_run.apply(
                            lambda x: sum(val for val in x['p_values'].values() if pd.notna(val)), axis=1
                        ).idxmin()
                        best_row_mc = mc_results_df_full_run.loc[best_idx_loc_mc]
                        best_params_from_mc = best_row_mc['params'] # Params from MC are already structured
                        
                        if any(p < 0.05 for p in best_row_mc['p_values'].values() if pd.notna(p)):
                            print("\n✓ Found statistically significant parameter set from Monte Carlo for Full Run.")
                            initial_params_for_wfa = {
                                'long_risk': best_params_from_mc['long_risk'],
                                'long_reward': best_params_from_mc['long_reward'],
                                'position_size': best_params_from_mc['position_size'],
                                'max_positions_param': best_params_from_mc['max_positions'],
                                'adx_thresh': best_params_from_mc['adx_threshold'],
                                'persistence_days': best_params_from_mc['persistence_days'],
                                'max_position_duration': best_params_from_mc['max_position_duration'],
                                'threshold': best_params_from_mc['threshold']
                            }
                            print("\nProceeding to Walk-Forward Analysis for Full Run...")
                            wfa_summary_full_run = walk_forward_analysis(IS, OOS, initial_params_for_wfa)
                            if wfa_summary_full_run:
                                print("\nAnchored Walk-Forward Analysis for Full Run completed.")
                            else:
                                print("Anchored Walk-Forward Analysis for Full Run failed or produced no results.")
                        else:
                            print("⚠ No statistically significant parameters from MC. Using best from initial Opt for WFA.")
                    else:
                        print("Error in MC results format for Full Run.")
                else:
                    print("Monte Carlo analysis for Full Run did not yield results.")
            else:
                print("Initial optimization for Full Run did not yield results.")

        # --------------------------------------------------------------------------------------------------------------------------
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()
        return None
# ---------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()