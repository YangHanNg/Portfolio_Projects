import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import yfinance as yf
from tabulate import tabulate  
import pandas_ta as ta 
import multiprocessing as mp 
from collections import deque
from numba import jit
import time
from tqdm import tqdm
import optuna
from functools import partial  # Add missing import for partial
import traceback
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf
from joblib import Parallel, delayed
#================== SCRIPT PARAMETERS =======================================================================================================================
# Controls
TYPE = 2 # 1. Full run # 2. Monte Carlo # 3. Optimization # 4. Test

# Optimization directions
OPTIMIZATION_DIRECTIONS = {
    'sharpe': 'maximize',
    'profit_factor': 'maximize',
    'avg_win_loss_ratio': 'maximize',
    'max_drawdown': 'minimize',
}
# Optimization objectives
OBJECTIVE_WEIGHTS = {
    'sharpe': 0.30,        
    'profit_factor': 0.20, 
    'avg_win_loss_ratio': 0.30,     
    'max_drawdown': 0.20   
}

# Optimization parameters
THRESHOLD = {
    'Entry': 0.7,
    'Exit': 0.4,
    'RSI_BUY': 35,
    'RSI_EXIT': 60,
}
ADX_THRESHOLD_DEFAULT = 30
DEFAULT_LONG_RISK = 0.03 
DEFAULT_LONG_REWARD = 0.08 
DEFAULT_POSITION_SIZE = 0.05
MAX_OPEN_POSITIONS = 100
PERSISTENCE_DAYS = 3
MAX_POSITION_DURATION = 10

# Risk & Reward
DEFAULT_SHORT_RISK = 0.01  
DEFAULT_SHORT_REWARD = 0.03  

# Base parameters
TICKER = ['PEP']
INITIAL_CAPITAL = 100000.0

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
        entry_commission = trade_series['commission']

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
#================== OPTIMIZING STRATEGY =======================================================================================================================
def parameter_test(df, long_risk, long_reward, position_size, tech_params, trial=None):
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
        trade_log, stats, equity_curve, returns_series = momentum(
            df,
            long_risk=long_risk,
            long_reward=long_reward,
            position_size=position_size,
            max_positions=tech_params.get('MAX_OPEN_POSITIONS', MAX_OPEN_POSITIONS),
            adx_threshold=tech_params.get('ADX_THRESHOLD', ADX_THRESHOLD_DEFAULT),
            persistence_days=tech_params.get('PERSISTENCE_DAYS', PERSISTENCE_DAYS),
            max_position_duration=tech_params.get('MAX_POSITION_DURATION', MAX_POSITION_DURATION),
            threshold=tech_params.get('THRESHOLD', THRESHOLD),
        )
        
        # 5. Process metrics
        required_metrics_keys = ['Sharpe Ratio', 'Profit Factor', 'Return (%)', 'Max Drawdown (%)']
        if stats and all(metric in stats for metric in required_metrics_keys):
            metrics = [
                stats['Sharpe Ratio'],
                stats['Profit Factor'],
                stats['Avg Win/Loss Ratio'],
                stats['Max Drawdown (%)']
            ]
            
            if trial:
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
        'long_risk': trial.suggest_float('long_risk', 0.01, 0.10, step=0.01),
        'long_reward': trial.suggest_float('long_reward', 0.02, 0.20, step=0.01),
        'position_size': trial.suggest_float('position_size', 0.05, 0.50, step=0.05),

        # Technical parameters
        'max_open_positions': trial.suggest_int('max_open_positions', 5, 50,),
        'adx_threshold': trial.suggest_float('adx_threshold', 20.0, 30.0, step=1.0),
        'persistence_days': trial.suggest_int('persistence_days', 3,7),
        'max_position_duration': trial.suggest_int('max_position_duration', 5, 30),

        # Thresholds
        'entry': trial.suggest_float('entry', 0.5, 0.7, step=0.05),
        'exit': trial.suggest_float('exit', 0.3, 0.5, step=0.05),
        'rsi_buy': trial.suggest_float('rsi_buy', 30, 35, step=0.5),
        'rsi_exit': trial.suggest_float('rsi_exit', 60, 65, step=0.5),
    }
    
    # 1. Risk/Reward Validation
    if (params['long_risk'] >= params['long_reward']):
        trial.set_user_attr("invalid_reason", "Risk >= Reward")
        return bad_metrics_template

    if (params['long_reward']/params['long_risk'] > 4.0):
        trial.set_user_attr("invalid_reason", "High Reward/Risk Ratio")
        return bad_metrics_template

    # 2. Position Sizing Validation
    max_position_risk = params['position_size'] * params['max_open_positions']
    if max_position_risk > 2.5:  # >100% exposure
        trial.set_user_attr("invalid_reason", "Excessive position risk")
        return bad_metrics_template
    
    # Set up signal weights
    threshold = THRESHOLD.copy()
    threshold['Entry'] = params['entry']
    threshold['Exit'] = params['exit']
    threshold['RSI_BUY'] = params['rsi_buy']
    threshold['RSI_EXIT'] = params['rsi_exit']

    # Set up combined tech params
    tech_params = { 
        'MAX_OPEN_POSITIONS': params['max_open_positions'],
        'ADX_THRESHOLD': params['adx_threshold'],
        'PERSISTENCE_DAYS': params['persistence_days'],
        'MAX_POSITION_DURATION': params['max_position_duration'],
        'THRESHOLD': threshold,
    }
    # Evaluate this parameter set
    return parameter_test(
        base_df.copy(),
        params['long_risk'],
        params['long_reward'],
        params['position_size'],
        tech_params,
        trial
    )
# --------------------------------------------------------------------------------------------------------------
def optimize(prepared_data):
    # Optimizing parameters for Optuna
    target_metrics = list(OPTIMIZATION_DIRECTIONS.keys())
    opt_directions = [OPTIMIZATION_DIRECTIONS[metric] for metric in target_metrics]
    n_trials=300
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
        pruner=optuna.pruners.PatientPruner(base_pruner, patience=3),
        sampler=optuna.samplers.NSGAIIISampler(seed=42, population_size=50)  # Use NSGA-III for multi-objective
    )

    # Define the objective function
    objective_func = partial(objectives, base_df=data)
        
    # Run optimization with progress bar
    completed_trials=0
    print("\nOptimizing Parameters...")
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
    return pareto_front[:15]
# --------------------------------------------------------------------------------------------------------------------------
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
        ["Starting Capital [$]", f"{equity_curve.iloc[0]:,.2f}"],
        ["Ending Capital [$]", f"{equity_curve.iloc[-1]:,.2f}"],
        ["Start", f"{df.index[0].strftime('%Y-%m-%d')}"],
        ["End", f"{df.index[-1].strftime('%Y-%m-%d')}"],
        ["Duration [days]", f"{exposure_time}"],
        ["Equity Final [$]", f"{equity_curve.iloc[-1]:,.2f}"],
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
    
    print(f"\n=== TRADE SUMMARY ===") 
    trade_metrics = [
        ["Total Trades", f"{stats['Total Trades']:.0f}"], 
        ["Win Rate [%]", f"{stats['Win Rate']:.2f}"],
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
#================== STRATEGY SIGNIFICANCE TESTING ==============================================================================================================7
def stationary_bootstrap(data, block_size = 10.0, num_samples= 1000, sample_length = None, seed = None):
    if seed is not None:
        np.random.seed(seed)
    
    n = len(data)
    if sample_length is None:
        sample_length = n
    
    # Probability parameter for geometric distribution
    p = 1.0 / block_size
    
    # Store bootstrapped samples
    bootstrap_samples = []
    
    for _ in range(num_samples):
        # Initialize indices array for this bootstrap sample
        indices = np.zeros(sample_length, dtype=int)
        
        # Generate the first index randomly
        t = np.random.randint(0, n)
        
        # Fill the bootstrap indices
        for i in range(sample_length):
            indices[i] = t
            
            # With probability p, start a new block
            if np.random.random() < p:
                t = np.random.randint(0, n)
            else:
                # Continue the block, handling the case where we reach the end
                t = (t + 1) % n
        
        # Create the bootstrapped sample with proper index
        bootstrap_sample = data.iloc[indices].copy()
        
        # Reset the index to maintain the same date structure as original
        # This is important for strategy testing - maintaining continuous time
        bootstrap_sample.index = data.index[:len(bootstrap_sample)]
        
        bootstrap_samples.append(bootstrap_sample)
    
    return bootstrap_samples
# --------------------------------------------------------------------------------------------------------------------------  
def monte_carlo(prepared_data, pareto_front, num_simulations=1000):
    
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
        block_size=10.0,
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
                observed_metrics[internal_key] = np.nan # Or some default bad value

        # Initialize arrays for simulation metrics using internal keys
        sim_metrics = {internal_key: [] for internal_key in metric_key_map.keys()}
    
        num_bootstrap_samples = len(bootstrap_samples)
        for sample_idx, sample in enumerate(tqdm(bootstrap_samples, total=num_bootstrap_samples, desc=f"Sims for Set {param_idx+1}", leave=False, position=(param_idx % (mp.cpu_count() if mp.cpu_count() > 0 else 1)) )):
            _, sim_stats_run, _, _ = momentum(sample, **params)
            
            for internal_key, original_key in metric_key_map.items():
                if original_key in sim_stats_run:
                    sim_metrics[internal_key].append(sim_stats_run[original_key])
                else:
                    # This case should ideally not happen
                    # print(f"Warning: Simulated metric {original_key} not found in sim_stats for param set {param_idx + 1}, sample {sample_idx}.") # Optional: reduce verbosity
                    sim_metrics[internal_key].append(np.nan) # Append NaN if key is missing
        
        # Calculate p-values and simulation statistics
        results = {
            'parameter_set': param_idx + 1,
            'params': params,
            'p_values': {},
            'percentiles': {},
            'observed_metrics': observed_metrics,
            'simulation_metrics': {}
        }
        
        for internal_key in sim_metrics: # internal_key is 'sharpe', 'profit_factor', etc.
            sim_array_raw = np.array(sim_metrics[internal_key], dtype=float)
            # Filter out NaNs that might have been appended if keys were missing
            sim_array = sim_array_raw[~np.isnan(sim_array_raw)]

            observed_value = observed_metrics.get(internal_key, np.nan)

            if np.isnan(observed_value) or len(sim_array) == 0:
                results['p_values'][internal_key] = np.nan
                results['percentiles'][internal_key] = np.nan
                results['simulation_metrics'][internal_key] = {
                    'mean': np.nan, 'std': np.nan, 'skew': np.nan, 'kurtosis': np.nan
                }
                continue
            
            # Calculate p-value (proportion of simulations better than observed)
            if internal_key in ['sharpe', 'profit_factor', 'avg_win_loss_ratio']: # These are the internal keys
                p_value = np.mean(sim_array >= observed_value)
            else:  # max_drawdown - lower is better
                p_value = np.mean(sim_array <= observed_value)
            
            # Calculate percentile of observed value
            percentile = stats.percentileofscore(sim_array, observed_value)
            
            results['p_values'][internal_key] = p_value
            results['percentiles'][internal_key] = percentile
            results['simulation_metrics'][internal_key] = {
                'mean': np.mean(sim_array),
                'std': np.std(sim_array),
                'skew': stats.skew(sim_array),
                'kurtosis': stats.kurtosis(sim_array)
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
    
    # Print summary statistics
    print("\n=== Monte Carlo Analysis Summary ===")
    print(f"Total Parameter Sets Analyzed: {len(mc_results)}")
    print(f"Significant Results (p < 0.05):")

    for metric in ['sharpe', 'profit_factor', 'avg_win_loss_ratio', 'max_drawdown']:
        sig_count = sum(1 for result in mc_results if result['p_values'][metric] < 0.05)
        print(f"{metric}: {sig_count}/{len(mc_results)}")
    
    print("\nTop 5 Parameter Sets by Statistical Significance:")
    headers = ["Set", "Metric", "Observed", "Sim Mean", "Sim Std", "p-value", "Percentile"]
    rows = []
    
    # Sort by combined significance
    sorted_results = sorted(mc_results, 
                          key=lambda x: sum(x['p_values'].values()))[:5]
    
    for result in sorted_results:
        for metric in ['sharpe', 'profit_factor', 'avg_win_loss_ratio', 'max_drawdown']:
            rows.append([
                f"{result['parameter_set']}",
                metric,
                f"{result['observed_metrics'][metric]:.2f}",
                f"{result['simulation_metrics'][metric]['mean']:.2f}",
                f"{result['simulation_metrics'][metric]['std']:.2f}",
                f"{result['p_values'][metric]:.3f}",
                f"{result['percentiles'][metric]:.1f}"
            ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    return results_df
#================== STRATEGY ROBUSTNESS TESTING ==================================================================================================================
def walk_forward_analysis(prepared_data, parameters, train_months=24, test_months=6, min_train=12, n_jobs=-1):
    
    def run_strategy_window(data_window):
        """Run strategy on a specific data window"""
        performance_log, stats, equity_curve, returns_series = momentum(
            data_window,
            long_risk=parameters['long_risk'],
            long_reward=parameters['long_reward'],
            position_size=parameters['position_size'],
            max_positions=parameters['tech_params']['MAX_OPEN_POSITIONS'],
            adx_threshold_sig=parameters['tech_params']['ADX_THRESHOLD'],
            persistence_days=parameters['tech_params']['PERSISTENCE_DAYS'],
            max_position_duration=parameters['tech_params']['MAX_POSITION_DURATION'],
            threshold=parameters['tech_params']['THRESHOLD']
        )
        return {'stats': stats, 'returns_series': returns_series}

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
#================== MAIN PROGRAM EXECUTION =======================================================================================================================
def main():
    try:
        if isinstance(TICKER, list):
            ticker_str = TICKER[0]
        else:
            ticker_str = TICKER
        IS, OOS = get_data(ticker_str)
        df_prepared = prepare_data(IS)
        if TYPE == 4:
            test(df_prepared, long_risk=DEFAULT_LONG_RISK, long_reward=DEFAULT_LONG_REWARD, position_size=DEFAULT_POSITION_SIZE, max_positions_param=MAX_OPEN_POSITIONS,
                adx_thresh=ADX_THRESHOLD_DEFAULT, persistence_days=PERSISTENCE_DAYS, max_position_duration=MAX_POSITION_DURATION, threshold=THRESHOLD)
        elif TYPE == 3:
            # Run optimization on in-sample data
            pareto_front = optimize(df_prepared)
            if pareto_front:
                visualize(pareto_front, df_prepared)
            else:
                print("Optimization did not yield any results.")
        elif TYPE == 2:  # Monte Carlo Testing
            pareto_front = optimize(df_prepared)[:3]
            if pareto_front and len(pareto_front) > 0:
                mc_results_df = monte_carlo(df_prepared, pareto_front)
                if not mc_results_df.empty:
                    # Find best parameter set (lowest combined p-value)
                    # Ensure 'p_values' column exists and contains dictionaries
                    if 'p_values' in mc_results_df.columns and mc_results_df['p_values'].apply(lambda x: isinstance(x, dict)).all():
                        best_idx_loc = mc_results_df.apply(
                            lambda x: sum(val for val in x['p_values'].values() if pd.notna(val)), axis=1
                        ).idxmin()
                        best_row = mc_results_df.loc[best_idx_loc]
                        best_params = best_row['params']
                        
                        if any(p < 0.05 for p in best_row['p_values'].values() if pd.notna(p)):
                            print("\n✓ Found statistically significant parameter set from Monte Carlo.")
                            # Reconstruct the 'threshold' sub-dictionary for the test function
                            test_params = {
                                'long_risk': best_params['long_risk'],
                                'long_reward': best_params['long_reward'],
                                'position_size': best_params['position_size'],
                                'max_positions_param': best_params['max_positions'],
                                'adx_thresh': best_params['adx_threshold'],
                                'persistence_days': best_params['persistence_days'],
                                'max_position_duration': best_params['max_position_duration'],
                                'threshold': best_params['threshold'] # Pass the threshold dict directly
                            }
                            test(df_prepared, **test_params)
                        else:
                            print("⚠ No statistically significant parameter sets found after Monte Carlo.")
                    else:
                        print("Error: 'p_values' column is missing or not in the expected format in mc_results_df.")
                else:
                    print("Monte Carlo analysis did not yield any results.")
            else:
                print("Optimization did not yield any Pareto front for Monte Carlo.")
        elif TYPE == 1:
            return None
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()
        return None
# -------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()