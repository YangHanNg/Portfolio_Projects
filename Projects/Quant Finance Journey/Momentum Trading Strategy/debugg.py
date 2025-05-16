import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import yfinance as yf
from tabulate import tabulate  
import random
import pandas_ta as ta 
from datetime import datetime, timedelta
import multiprocessing as mp 
from collections import deque
from numba import jit
from tqdm import tqdm
import optuna
from functools import partial  # Add missing import for partial
import traceback
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from joblib import Parallel, delayed
#================== SCRIPT PARAMETERS =======================================================================================================================
# Controls
TYPE = 4 # 1. Full run # 2. Monte Carlo # 3. Optimization # 4. Test

# Optimization directions
OPTIMIZATION_DIRECTIONS = {
    'sharpe': 'maximize',
    'profit_factor': 'maximize',
    'returns': 'maximize',
    'max_drawdown': 'minimize',
}
# Optimization objectives
OBJECTIVE_WEIGHTS = {
    'sharpe': 0.30,        
    'profit_factor': 0.20, 
    'returns': 0.20,     
    'max_drawdown': 0.30   
}

# Optimization parameters
THRESHOLD = {
    'Entry': 0.5,
    'Exit': 0.3,
    'RSI_BUY': 32,
    'RSI_EXIT': 62,
}
ADX_THRESHOLD_DEFAULT = 25
DEFAULT_LONG_RISK = 0.04 
DEFAULT_LONG_REWARD = 0.12 
DEFAULT_POSITION_SIZE = 0.05
MAX_OPEN_POSITIONS = 100
PERSISTENCE_DAYS = 3
MAX_POSITION_DURATION = 15

# Risk & Reward
DEFAULT_SHORT_RISK = 0.01  
DEFAULT_SHORT_REWARD = 0.03  

# Base parameters
TICKER = ['PEP']
INITIAL_CAPITAL = 10000.0

# Moving average strategy parameters
FAST = 20
SLOW = 50
WEEKLY_MA_PERIOD = 50

# RSI strategy parameters
RSI_LENGTH = 14

# Bollinger Bands strategy parameters
BB_LEN = 20
ST_DEV = 2
#================== DATA RETRIEVAL & HANDLING ===============================================================================================================
def get_data(ticker):
    data_start_year = 2013
    print('\nDownloading data....')
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
    print(f"Data split: In-Sample from {data_oldest.year} to {in_sample_period.year}, Out-of-Sample from {out_of_sample_period.year} to {data_latest.year}")
    return in_sample_df, out_of_sample_df
# --------------------------------------------------------------------------------------------------------------------------
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
# --------------------------------------------------------------------------------------------------------------------------
def signals(df, adx_threshold, threshold):
    
    fast_ma_col = f"{FAST}_ma"
    slow_ma_col = f"{SLOW}_ma"
    signals_df = pd.DataFrame(index=df.index)

    params = ['Entry', 'Exit', 'RSI_BUY', 'RSI_EXIT']
    for param in params:
        if param in threshold:
            threshold[param] = float(threshold[param])
        else:
            print(f"Warning: {param} not found in threshold. Using default value.")
    
    rsi_buy_threshold = threshold.get('RSI_BUY', 40)
    rsi_exit_threshold = threshold.get('RSI_EXIT', 60)
    buy_threshold = threshold.get('Entry', 0.6)
    exit_threshold = threshold.get('Exit', 0.4)

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
    high_np = df['High'].values
    lower_np = df['Low'].values
    
    # Calculate ATR ratio for volatility normalization
    atr_ratio_np = np.full_like(close_np, 0.02, dtype=float) # Ensure float
    valid_close_mask = close_np != 0
    safe_close_np = np.where(valid_close_mask, close_np, 1e-9) # Avoid division by zero
    atr_ratio_np[valid_close_mask] = atr_np[valid_close_mask] / safe_close_np[valid_close_mask]
    atr_ratio_np = np.nan_to_num(atr_ratio_np, nan=0.02, posinf=0.02, neginf=0.02)

    # 2. Calculate Primary and Secondary Conditions
    
    # Momentum persistence check
    roll_window = 5
    higher_highs = np.zeros_like(high_np, dtype=bool)
    higher_lows = np.zeros_like(lower_np, dtype=bool)

    for i in range(roll_window, len(high_np)):
        # Check if current high is higher than previous N-day high
        higher_highs[i] = high_np[i] > np.max(high_np[i-roll_window:i])
        # Check if current low is higher than previous N-day low
        higher_lows[i] = lower_np[i] > np.max(lower_np[i-roll_window:i])

    conditions = {
        'primary_trend_long': (close_np > fast_ma_np) & (close_np > slow_ma_np),
        'trend_strength_ok': adx_np > adx_threshold,
        'rsi_buy_zone': rsi_np < rsi_buy_threshold,
        'rsi_exit_zone': rsi_np > rsi_exit_threshold,
        'rsi_extreme_exit': rsi_np > 80,
        'volume_ok': df['volume_confirmed'].values.astype(bool),
        'weekly_uptrend': df['weekly_uptrend'].values.astype(bool),
        'bb_buy': close_np < lower_band_np,
        'bb_exit': close_np > upper_band_np,
        'price_momentum_strong': higher_highs | higher_lows
    }
    
    # Convert all conditions to boolean arrays
    for key in conditions:
        conditions[key] = np.array(conditions[key], dtype=bool)

    # 3. Calculate Momentum Score - single score reflecting long momentum strength
    momentum_score = np.zeros_like(close_np, dtype=float)

    # Trend component (40%)
    trend_score = ((close_np > fast_ma_np).astype(float) * 0.2 +
                   (fast_ma_np > slow_ma_np).astype(float) * 0.2)
    
    # RSI component (25%)
    rsi_score = np.zeros_like(close_np, dtype=float)
    # RSI below 30 = strong buy signal, scales linearly up to 50
    rsi_score = np.where(rsi_np < 30, 0.25,
                         np.where(rsi_np < 50, 0.25 *(50 - rsi_np) / 20, 0))

    # Volume component (20%)
    volume_score = conditions['volume_ok'].astype(float) * 0.2

    # Price action component (10%)
    price_action_score = conditions['price_momentum_strong'].astype(float) * 0.15

    # Combined scores
    momentum_score = trend_score + rsi_score + volume_score + price_action_score

    # Normalize to 0-1 range and apply volatility adjustment
    momentum_score *= (1+np.log1p(atr_ratio_np)) # Higher volatility = amplified score
    momentum_score = np.clip(momentum_score, 0, 1)

    # 4. Calculate Momentum Decay
    momentum_decay = np.zeros_like(close_np, dtype=bool)
    buy_signal = np.zeros_like(close_np, dtype=bool)
    exit_signal = np.zeros_like(close_np, dtype=bool)
    immediate_exit = np.zeros_like(close_np, dtype=bool)
    decay_window = 3

    for i in range(decay_window, len(momentum_score)):
        if all(momentum_score[i-j] < momentum_score[i-j-1] for j in range(1, decay_window)):
            momentum_decay[i] = True

    # 5. Generate Buy and Exit Signals
    # Buy signal: Strong momentum + trend confirmation + not in decay
    buy_signal = (
        conditions['trend_strength_ok'] & 
        conditions['primary_trend_long'] & 
        conditions['volume_ok'] & 
        ~momentum_decay &
        (momentum_score > buy_threshold)  # Threshold for entry
    )
    
    # Exit signal: Momentum decay OR overbought OR trend broken
    exit_signal = (
        momentum_decay | 
        conditions['rsi_exit_zone'] |
        conditions['bb_exit'] |
        (~conditions['primary_trend_long'] & (momentum_score < exit_threshold))
    )

    # Immediate exit: Extreme RSI or dramatic trend breakdown
    immediate_exit = conditions['rsi_extreme_exit'] | (adx_np < adx_threshold/2)
    
    # 6. Store results in DataFrame
    signals_df = pd.DataFrame(index=df.index)
    signals_df['momentum_score'] = momentum_score
    signals_df['momentum_decay'] = momentum_decay
    signals_df['buy_signal'] = buy_signal
    signals_df['exit_signal'] = exit_signal
    signals_df['immediate_exit'] = immediate_exit
    
    # Calculate signal_changed flag
    last_buy_signal = buy_signal.astype(int)
    signal_changed_np = np.diff(last_buy_signal, prepend=0) != 0
    signals_df['signal_changed'] = signal_changed_np
    
    return signals_df
#================== MOMENTUM STRATEGY =======================================================================================================================
def momentum(df_with_indicators, long_risk=DEFAULT_LONG_RISK, long_reward=DEFAULT_LONG_REWARD, position_size=DEFAULT_POSITION_SIZE, risk_free_rate=0.04, 
             max_positions=MAX_OPEN_POSITIONS, adx_threshold=ADX_THRESHOLD_DEFAULT, persistence_days=PERSISTENCE_DAYS, max_position_duration=MAX_POSITION_DURATION, threshold=THRESHOLD):

    if df_with_indicators.empty and df_with_indicators.index.empty:
        print("Warning: Empty dataframe (df_with_indicators) provided to momentum.")
        return [], {}, pd.Series(dtype='float64'), pd.Series(dtype='float64')

    # 1. Generate signals using the indicator-laden DataFrame
    signals_df = signals(df_with_indicators, adx_threshold=adx_threshold, threshold=threshold)

    # 2. Initialize Trade Managers and Tracking
    trade_manager = TradeManager(INITIAL_CAPITAL, max_positions)

    # Use the index from df_with_indicators
    equity_curve = pd.Series(INITIAL_CAPITAL, index=df_with_indicators.index)
    returns_series = pd.Series(0.0, index=df_with_indicators.index)

    # 3. Main Processing Loop
    for i in range(1, len(df_with_indicators)):
        current_date = df_with_indicators.index[i]
        transaction_price = df_with_indicators['Open'].iloc[i]

        # Critical data checks for the current row
        previous_day_data_row = df_with_indicators.iloc[i-1]
        previous_day_atr = previous_day_data_row['ATR']
        previous_day_adx = previous_day_data_row['ADX']

        # Get signals for the previous day
        buy_signal = signals_df['buy_signal'].iloc[i-1]
        exit_signal = signals_df['exit_signal'].iloc[i-1]
        immediate_exit = signals_df['immediate_exit'].iloc[i-1]
        momentum_score = signals_df['momentum_score'].iloc[i-1]
        momentum_decay = signals_df['momentum_decay'].iloc[i-1]


        # Check for momentum persistence
        momentum_persistence = False
        if i >= persistence_days:
            momentum_persistence = all(signals_df['momentum_score'].iloc[i-persistence_days:i] > 0.6)

        # --- Exit Conditions (Priority Order) ---
        if trade_manager.position_count > 0:
            # 1. Check trailing stop first (highest priority)
            any_trailing_stop_hit = trade_manager.trailing_stops(transaction_price, current_date, previous_day_atr, previous_day_adx)
            if not any_trailing_stop_hit:
                # Check for immediate exit signal
                if immediate_exit:
                    trade_manager.process_exits(current_date, transaction_price, direction_to_exit='Long', Trim=0.0)
                # Normal exit with tiered approach
                elif exit_signal:
                    # Determine if we should exit partially or fully
                    if momentum_score > 0.4:
                        # Partial exit if momentum is strong
                        trade_manager.process_exits(current_date, transaction_price, direction_to_exit='Long', Trim=0.4)
                    else: 
                        # Momentum below threshold
                        trade_manager.process_exits(current_date, transaction_price, direction_to_exit='Long', Trim=0.0)
                
                elif momentum_persistence and trade_manager.position_count > 0:
                    # Take partial profits if we had strong momentum for a while
                    unrealized_pnls = trade_manager.unrealized_pnl(transaction_price)
                    portfolio_value = trade_manager.portfolio_value + unrealized_pnls
                    if unrealized_pnls > (portfolio_value * 0.05):
                        trade_manager.process_exits(current_date, transaction_price, direction_to_exit='Long', Trim=0.3)

            # Check position health
            position_health = trade_manager.position_health(transaction_price, previous_day_atr, current_date, momentum_score)

            # 2. Check trend validity  
            if trade_manager.position_count > 0 and not any_trailing_stop_hit:
                # Check if we have position with substantial profits
                if position_health['profit_factor'] > 1.5:
                    # Scale exit size based on momentum score
                    if position_health['strength'] in ['weak', 'very_weak']:
                        # Exit full position if momentum is weak
                        trade_manager.process_exits(current_date, transaction_price, direction_to_exit='Long', Trim=0.0)
                    elif position_health['strength'] == 'moderate':
                        # Partial exit (50%) if momentum is moderate
                        trade_manager.process_exits(current_date, transaction_price, direction_to_exit='Long', Trim=0.5)
                    elif position_health['strength'] in ['strong', 'very_strong'] and position_health['profit_factor'] > 3.0:
                        # Take some profits even with strong momentum
                        trade_manager.process_exits(current_date, transaction_price, direction_to_exit='Long', Trim=0.3)
                
                # Check for position that have reached their time limit
                for pos_idx, duration in position_health['position_duration'].items():
                    if duration > max_position_duration:
                        # Reduce exposure for time-based risk management
                        if pos_idx in trade_manager.active_trades.index:  # Ensure position still exists
                            # Exit partially for old positions regardless of performance
                            trade_manager.process_exits(current_date, transaction_price, direction_to_exit='Long', Trim=0.5)

        # --- Entry Conditions ---
        if buy_signal and trade_manager.position_count < max_positions:
            # Only enter if not in decay and ADX shows sufficient trend strength
            if not momentum_decay and previous_day_adx > adx_threshold:
                entry_params = {'price': transaction_price*(1 + 0.003),
                                'portfolio_value': trade_manager.portfolio_value,
                                'risk': long_risk, 'reward': long_reward,
                                'atr': previous_day_atr,
                                'adx': previous_day_adx,
                                'position_size': position_size,
                            }
                trade_manager.process_entry(current_date, entry_params, direction='Long')
        
        # Performance Tracking
        total_value = trade_manager.portfolio_value + trade_manager.unrealized_pnl(transaction_price)
        equity_curve.iloc[i] = total_value
        returns_series.iloc[i] = (total_value / equity_curve.iloc[i-1] - 1) if equity_curve.iloc[i-1] != 0 else 0.0

    # Calculate final statistics
    stats = trade_statistics(equity_curve, trade_manager.trade_log, trade_manager.wins, 
                             trade_manager.losses, risk_free_rate)

    return trade_manager.trade_log, stats, equity_curve, returns_series
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
    def position_health(self, current_price, current_atr, current_date, current_score):
        if self.active_trades.empty:
            return {'profit_factor': 0.0, 'score_strength': 'none', 'position_duration': {}, 'take_profit_levels': {}}
        current_price = np.float32(current_price)
        current_atr = np.float32(current_atr)
        current_score = np.float32(current_score)

        unrealized_pnls = self.unrealized_pnl(current_price)

        total_risk = 0.0
        position_durations = {}
        take_profit_levels = {}

        for idx, trade in self.active_trades.iterrows():
            if trade['direction']  == 'long':
                risk_per_trade = abs(trade['entry_price'] - trade['stop_loss']) * trade['share_amount']
                total_risk += risk_per_trade

                if isinstance(trade['entry_date'], pd.Timestamp):
                    position_durations[trade['entry_date']] = (current_date - trade['entry_date']).days

                current_profit = (current_price - trade['entry_price']) * trade['share_amount']
                if current_profit > 0:
                    r_multiple = current_profit / risk_per_trade if risk_per_trade > 0 else 0

                    take_profit_levels[idx] = {
                        '1R': trade['entry_price'] + 1 * abs(trade['entry_price'] - trade['stop_loss']),
                        '2R': trade['entry_price'] + 2 * abs(trade['entry_price'] - trade['stop_loss']),
                        '3R': trade['entry_price'] + 3 * abs(trade['entry_price'] - trade['stop_loss']),
                        'current_r': r_multiple
                    }
        profit_factor = unrealized_pnls / total_risk if total_risk > 0 else 0.0

        score_strength = 'none'
        if current_score > 0.8:
            score_strength = 'very_strong'
        elif current_score > 0.6:
            score_strength = 'strong'
        elif current_score > 0.4:
            score_strength = 'moderate'
        elif current_score > 0.2:
            score_strength = 'weak'
        else:
            score_strength = 'very_weak'
        return {
            'profit_factor': profit_factor,
            'score_strength': score_strength,
            'position_duration': position_durations,
            'take_profit_levels': take_profit_levels
        }
    # ----------------------------------------------------------------------------------------------------------
    def trailing_stops(self, current_price, current_date, current_atr, adx_value):
        if self.active_trades.empty:
            return False

        current_price_f32 = np.float32(current_price)
        current_atr_f32 = np.float32(current_atr)
        atr_multiplier = np.float32(2.5 if adx_value > 30 else 1.5)

        # Initialize stops_hit flags
        stops_hit_flags = pd.Series([False] * len(self.active_trades), index=self.active_trades.index)

        # Process Long Positions (since we only have longs now)
        # Update highest prices for all active trades
        self.active_trades['highest_close_since_entry'] = np.maximum(
            self.active_trades['highest_close_since_entry'],
            current_price_f32
        )
        
        # Calculate new stop-loss levels
        new_stops = self.active_trades['highest_close_since_entry'] - (atr_multiplier * current_atr_f32)
        
        # Update stop-loss (only if new stop is higher)
        self.active_trades['stop_loss'] = np.maximum(
            self.active_trades['stop_loss'],
            new_stops
        )
        
        # Check if current price hits the stop-loss
        stops_hit_flags = current_price_f32 <= self.active_trades['stop_loss']

        # Process Exits for Hit Stops
        if stops_hit_flags.any():
            indices_to_remove = []
            trades_that_hit_stop = self.active_trades[stops_hit_flags]

            for idx, trade in trades_that_hit_stop.iterrows():
                pnl = self.exit_pnl(trade, current_date, current_price_f32, 
                                trade['remaining_shares'], 'Trailing Stop')
                self.portfolio_value += pnl
                self.allocated_capital -= (current_price_f32 * trade['remaining_shares'])
                self.position_count -= 1
                indices_to_remove.append(idx)
            
            if indices_to_remove:
                self.active_trades = self.active_trades.drop(index=indices_to_remove).reset_index(drop=True)
            return True # Indicates one or more trailing stops were hit and processed

        return False # No trailing stops were hit
    # ----------------------------------------------------------------------------------------------------------
    def process_exits(self, current_date, current_price, direction_to_exit, Trim=0.0): 

        if self.active_trades.empty:
            return 0.0

        # Convert inputs to proper types
        current_price = np.float32(current_price)
        total_pnl = 0.0
        
        # Initialize removal mask
        indices_to_remove = []
        
        # Get relevant trades (long only)
        long_trades = self.active_trades[self.active_trades['direction'] == 'Long']
        
        for idx, trade in long_trades.iterrows():
            current_shares = trade['remaining_shares']
            
            if current_shares <= 0:
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

            # Priority 3: Signal-based Exits (Full or Partial)
            if Trim > 0.0:
                # Partial exit
                shares_to_exit = int(current_shares * Trim)
                if shares_to_exit > 0:
                    pnl = self.exit_pnl(
                        trade, 
                        current_date, 
                        current_price,
                        shares_to_exit, 
                        'Partial Signal Exit'
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
                    'Full Signal Exit'
                )
                total_pnl += pnl
                self.portfolio_value += pnl
                self.allocated_capital -= (current_price * current_shares)
                self.position_count -= 1
                indices_to_remove.append(idx)
        
        # Remove closed positions
        if indices_to_remove:
            self.active_trades = self.active_trades.drop(index=indices_to_remove).reset_index(drop=True)
            
        return total_pnl
    # ----------------------------------------------------------------------------------------------------------
    def process_entry(self, current_date, entry_params, direction=None):
        
        is_long = direction == 'Long'
        direction_mult = 1 if is_long else -1
        
        # 1. Calculate Initial Stop (ATR + ADX adjusted)
        adx_normalized = np.clip((entry_params['adx'] - 20) / 30, 0, 1)  # ADX=20→0, ADX=50→1
        atr_multiplier = 2 + adx_normalized * 1.0 
        risk_based_stop = entry_params['price'] * entry_params['risk']
        atr_based_stop = entry_params['atr'] * atr_multiplier
        stop_distance = max(atr_based_stop, risk_based_stop)
        initial_stop = entry_params['price'] - (stop_distance * direction_mult)
        
        # 2. Position Sizing Based on Risk
        risk_per_share = abs(entry_params['price'] - initial_stop)
        max_risk_amount = entry_params['portfolio_value'] * entry_params['risk']
        shares_by_risk = int(max_risk_amount / max(risk_per_share, 1e-9))  # Avoid division by zero

        # Calculate shares based on position size limit
        max_position_value = entry_params['portfolio_value'] * entry_params['position_size']
        shares_by_size = int(max_position_value / entry_params['price'])

        shares = min(shares_by_risk, shares_by_size)

        # Calculate dollar amount for this position
        position_dollar_amount = shares * entry_params['price']
        actual_position_size = position_dollar_amount / entry_params['portfolio_value']


        max_total_exposure = 0.95
        current_exposure = 0.0 
        if not self.active_trades.empty:
            current_positions_value = (self.active_trades['share_amount'] * entry_params['price']).sum()
            current_exposure = current_positions_value / entry_params['portfolio_value']

        # Calculate new position exposure
        position_dollar_amount = shares * entry_params['price']
        new_exposure = position_dollar_amount / entry_params['portfolio_value']
        total_exposure = current_exposure + new_exposure
        
        # 3. Check total portfolio exposure (add here)
        if total_exposure > max_total_exposure:
            # Try to adjust position size to fit within exposure limits
            available_exposure = max_total_exposure - current_exposure
            if available_exposure > 0:
                adjusted_shares = int((available_exposure * entry_params['portfolio_value']) / entry_params['price'])
                if adjusted_shares > 0:
                    shares = min(shares, adjusted_shares)
                    position_dollar_amount = shares * entry_params['price']
                    actual_position_size = position_dollar_amount / entry_params['portfolio_value']
                else:
                    return False
            else:
                return False

        # 4. Available Capital Check 
        available_capital = self.portfolio_value - self.allocated_capital
        if position_dollar_amount > available_capital or shares <= 0:
            return False
        
        # Calculate commission before final check
        commission = self.calculate_commission(shares, entry_params['price'])

        # Final minimum position check
        min_position_value = entry_params['portfolio_value'] * 0.001  # 0.1% minimum position size
        if position_dollar_amount < min_position_value:
            return False
        
    
        # 4. Calculate Take-Profit Using Risk/Reward Ratio
        if 'reward' in entry_params:
            # Reward distance = (risk_per_share) * (reward/risk ratio)
            # Example: If risk=0.01 (1%) and reward=0.03 (3%), reward_distance = risk_per_share * 3
            reward_distance = risk_per_share * (entry_params['reward'] / entry_params['risk'])
            take_profit = entry_params['price'] + (reward_distance * direction_mult)
        else:
            take_profit = None

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
        # Fixed commission model
        fixed_fee = 5.00  # $5 per trade
        
        # Per-share commission model
        per_share_fee = 0.005 * shares  # 0.5 cents per share
        
        # Percentage-based model
        percentage_fee = shares * price * 0.001  # 0.1% of trade value
        
        # Tiered model example
        if shares * price < 5000:
            tiered_fee = 5.00
        elif shares * price < 10000:
            tiered_fee = 7.50
        else:
            tiered_fee = 10.00
        
        # Choose your model
        commission = per_share_fee  # or per_share_fee, percentage_fee, tiered_fee
        
        # Add minimum commission if needed
        return max(commission, 1.00)  # Minimum $1.00 commission
    # ----------------------------------------------------------------------------------------------------------
    def exit_pnl(self, trade_series, exit_date, exit_price, shares_to_exit, reason):
        
        entry_price = trade_series['entry_price']
        entry_date = trade_series['entry_date']
        trade_direction = trade_series['direction']
        position_id = trade_series['position_id']
        original_shares = trade_series['share_amount']

        # Calculate if this is a complete exit
        is_complete_exit = (shares_to_exit >= trade_series['remaining_shares'])

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
        self.portfolio_value -= self.calculate_commission(shares_to_exit, exit_price)  # Deduct commission for exit
        self.trade_log.append({
            'Entry Date': entry_date,
            'Exit Date': exit_date,
            'Direction': trade_direction,
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Shares': shares_to_exit,
            'Original Shares': original_shares,
            'PnL': pnl,
            'Duration': duration,
            'Exit Reason': reason,
            'Position ID': position_id,
            'Is Complete Exit': is_complete_exit,
            'Remaining Shares': trade_series['remaining_shares'] - shares_to_exit if not is_complete_exit else 0
        })
        return pnl
    # ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
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
        'Return (%)': net_profit_pct,
        'Profit Factor': profit_factor,
        'Expectancy (%)': expectancy_pct,
        'Max Drawdown (%)': max_drawdown,
        'Annualized Return (%)': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Avg Win/Loss Ratio': avg_win_loss_ratio
    }
# --------------------------------------------------------------------------------------------------------------------------
def test(df_input, long_risk=DEFAULT_LONG_RISK, long_reward=DEFAULT_LONG_REWARD, position_size=DEFAULT_POSITION_SIZE, max_positions_param=MAX_OPEN_POSITIONS,
         adx_thresh=ADX_THRESHOLD_DEFAULT, persistence_days=PERSISTENCE_DAYS, max_position_duration=MAX_POSITION_DURATION, threshold=THRESHOLD): 
    
    df = df_input.copy()
    
    trade_log, trade_stats, portfolio_equity, trade_returns = momentum(
        df_input, long_risk=long_risk, long_reward=long_reward, position_size=position_size, risk_free_rate=0.04, max_positions=max_positions_param,
        adx_threshold=adx_thresh, persistence_days=persistence_days, max_position_duration=max_position_duration, threshold=threshold)

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
    # print(f"Short Risk: {short_risk*100:.1f}% | Short Reward: {short_reward*100:.1f}%")
    print(f"Position Size: {position_size*100:.1f}% | Max Open Positions: {max_positions_param}")
    print(f"ADX Threshold: {adx_thresh:.1f} | Persistence: {persistence_days} days | Max Duration: {max_position_duration} days")
    print(f"Signal Weights: {threshold}")
    
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
        ["Avg. Win/Loss Ratio", f"{trade_stats['Avg Win/Loss Ratio']:.2f}"],
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

def main():
    IS_df, OOS_df = get_data(TICKER)
    df_prepared = prepare_data(IS_df)
    test(df_prepared, long_risk=DEFAULT_LONG_RISK, long_reward=DEFAULT_LONG_REWARD, position_size=DEFAULT_POSITION_SIZE, max_positions_param=MAX_OPEN_POSITIONS,
                adx_thresh=ADX_THRESHOLD_DEFAULT, persistence_days=PERSISTENCE_DAYS, max_position_duration=MAX_POSITION_DURATION, threshold=THRESHOLD)
    return None

if __name__ == "__main__":
    main()     
    