import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Momentum Strategy Example:

#Entry Signal:
# - 20-day MA > 50-day MA (bullish crossover)
# - Price is above both MAs
# - MACD crossover (MACD > signal line)
# - RSI climbing from 30 to 50
# - Price breaking upper Bollinger Band
# - MFI > 50 (volume supports move)

#Exit Signal:
# - 20-day MA crosses below 50-day MA
# - MACD turns down (bearish crossover)
# - RSI > 70 (overbought)
# - Price hits resistance or reverts inside bands
# - MFI divergence (price up, MFI down)

# Add backtesting over date ranges or multiple assets
# Implement position sizing or stop-loss / take-profit
# Use vectorized signals to reduce false triggers
# Track win/loss ratio, max drawdown, etc

# --------------------------------------------------------------------------------------------------------------------------

# Base parameters
TICKER = ['SPY']
DEFAULT_LOOKBACK = 8000

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
DEFAULT_RISK = 0.1  # 2% risk per trade
DEFAULT_REWARD = 0.5  # 4% target profit per trade
DEFAULT_POSITION_SIZE = 1  # 10% of portfolio per trade
MAX_OPEN_POSITIONS = 1 # Maximum number of concurrent open positions

# --------------------------------------------------------------------------------------------------------------------------

def get_data(tickers, start_date=None, end_date=None, lookback=DEFAULT_LOOKBACK):
    if isinstance(tickers, str):
        tickers = [tickers]
    
    # Set default date range if not provided
    if end_date is None:
        end_date = datetime.now()
    else:
        end_date = pd.to_datetime(end_date)
        
    if start_date is None:
        start_date = end_date - timedelta(days=lookback)
    else:
        start_date = pd.to_datetime(start_date)
    
    data_dict = {}
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            # Ensure column names are simplified
            df.columns = df.columns.get_level_values(0) if hasattr(df.columns, 'get_level_values') else df.columns
            # Add ticker column for identification
            df['Ticker'] = ticker
            data_dict[ticker] = df
            print(f"Downloaded {len(df)} bars for {ticker}")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    
    return df, data_dict

# --------------------------------------------------------------------------------------------------------------------------
# Functions for the strategies

def adding_indicators(df):
    
    # Moving Averages Crossover
    df[f'{FAST}_ma'] = df['Close'].rolling(FAST).mean()
    df[f'{SLOW}_ma'] = df['Close'].rolling(SLOW).mean()

    # Moving Average Convergence Divergence (MACD) - Vectorized
    df['MACD_fast'] = df['Close'].ewm(span=MACD_FAST).mean()
    df['MACD_slow'] = df['Close'].ewm(span=MACD_SLOW).mean()
    df['MACD'] = df['MACD_fast'] - df['MACD_slow']
    df['Signal'] = df['MACD'].ewm(span=MACD_SPAN).mean()
    df['MACD_hist'] = df['MACD'] - df['Signal'] 

    # Relative Strength Index (RSI) - Vectorized
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use exponential moving average for more stable RSI
    avg_gain = gain.ewm(com=RSI_LENGTH-1, min_periods=RSI_LENGTH).mean()
    avg_loss = loss.ewm(com=RSI_LENGTH-1, min_periods=RSI_LENGTH).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # Add small constant to avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))

    # Money Flow Index (MFI) - Vectorized
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    # Create boolean masks for positive and negative money flow
    positive_flow_mask = typical_price > typical_price.shift(1)
    negative_flow_mask = typical_price < typical_price.shift(1)
    
    # Apply masks to calculate positive and negative money flow
    positive_mf = pd.Series(0.0, index=money_flow.index, dtype='float64')
    negative_mf = pd.Series(0.0, index=money_flow.index, dtype='float64')
    
    positive_mf.loc[positive_flow_mask] = money_flow.loc[positive_flow_mask]
    negative_mf.loc[negative_flow_mask] = money_flow.loc[negative_flow_mask]
    
    # Calculate MFI
    positive_mf_sum = positive_mf.rolling(window=MFI_LENGTH).sum()
    negative_mf_sum = negative_mf.rolling(window=MFI_LENGTH).sum()
    mfi_ratio = positive_mf_sum / (negative_mf_sum + 1e-10)  # Add small constant
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))

    # Bollinger Bands - Vectorized
    df['BB_SMA'] = df['Close'].rolling(BB_LEN).mean()
    df['BB_STD'] = df['Close'].rolling(BB_LEN).std()
    df['Upper_Band'] = df['BB_SMA'] + (DEVS * df['BB_STD'])
    df['Lower_Band'] = df['BB_SMA'] - (DEVS * df['BB_STD'])
    
    # Calculate ATR for dynamic position sizing
    df['TR'] = np.maximum(
        np.maximum(
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Close'].shift(1))
        ),
        np.abs(df['Low'] - df['Close'].shift(1))
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Confirm signal generation with reduced false triggers
    # Use signal confirmation window to reduce noise
    df['MA_Bullish'] = (df[f'{FAST}_ma'] > df[f'{SLOW}_ma']) & (df[f'{FAST}_ma'].shift(1) <= df[f'{SLOW}_ma'].shift(1))
    df['MA_Bearish'] = (df[f'{FAST}_ma'] < df[f'{SLOW}_ma']) & (df[f'{FAST}_ma'].shift(1) >= df[f'{SLOW}_ma'].shift(1))
    
    df['MACD_Bullish'] = (df['MACD'] > df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1))
    df['MACD_Bearish'] = (df['MACD'] < df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1))
    
    df['RSI_Bullish'] = (df['RSI'] > 50) & (df['RSI'] < 70) & (df['RSI'].shift(1) <= 50)
    df['RSI_Bearish'] = (df['RSI'] < 50) & (df['RSI'] > 30) & (df['RSI'].shift(1) >= 50)
    
    df['BB_Bullish'] = (df['Close'] < df['Lower_Band']) & (df['Close'].shift(1) >= df['Lower_Band'].shift(1))
    df['BB_Bearish'] = (df['Close'] > df['Upper_Band']) & (df['Close'].shift(1) <= df['Upper_Band'].shift(1))
    
    # Combined signals with confirmation window to reduce false triggers
    df['Raw_Buy'] = (
        df['MA_Bullish'] | ((df['Close'] > df[f'{FAST}_ma']) | 
         (df['Close'] > df[f'{SLOW}_ma']) |
         (df['RSI_Bullish'] | (df['RSI'] > 50)) |
         (df['MACD_Bullish'] | (df['MACD_hist'].diff() > 0)) |
         (df['BB_Bullish'] | (df['Close'] < df['Lower_Band'])) |
         (df['MFI'] > 50))
    )

    df['Raw_Sell'] = (
        df['MA_Bearish'] |
        ((df[f'{FAST}_ma'] < df[f'{SLOW}_ma']) |
         (df['RSI'] > 70) |
         df['MACD_Bearish'] |
         df['BB_Bearish'])
    )

    # NEW: Track consecutive sell signals for 5-day requirement
    df['Consec_Sell_Days'] = 0
    # Set the first row to 0 or 1 based on Raw_Sell
    if len(df) > 0:
        df.loc[df.index[0], 'Consec_Sell_Days'] = 1 if df['Raw_Sell'].iloc[0] else 0
        
    # Use iterative approach for the rest of the rows
    for i in range(1, len(df)):
        if df['Raw_Sell'].iloc[i]:
            df.loc[df.index[i], 'Consec_Sell_Days'] = df['Consec_Sell_Days'].iloc[i-1] + 1
        else:
            df.loc[df.index[i], 'Consec_Sell_Days'] = 0
    
    # Modified Buy/Sell Signals according to new requirements
    df['Buy_Signal'] = df['Raw_Buy'] & df['Raw_Buy'].shift(1)  # Keep the same 2-day confirmation for buys
    df['Sell_Signal'] = df['Consec_Sell_Days'] >= 5  # Sell only after 5 consecutive days of sell signals
    
    df['Buy_Signal'] = df['Buy_Signal'].shift(1)
    df['Sell_Signal'] = df['Sell_Signal'].shift(1)
    
    return df.dropna()

# --------------------------------------------------------------------------------------------------------------------------

def set_momentum_strat(df, risk=DEFAULT_RISK, reward=DEFAULT_REWARD, position_size=DEFAULT_POSITION_SIZE, 
                      max_positions=MAX_OPEN_POSITIONS, risk_free_rate=0.04):
    
    # Initialize tracking variables
    initial_capital = 10000  # Starting with $100,000
    available_capital = initial_capital
    portfolio_value = initial_capital
    
    # Create a Series to track equity curve (portfolio value over time)
    equity = pd.Series(index=df.index, dtype=float)
    trade_returns = pd.Series(0.0, index=df.index)
    active_long_trades = []  # List to track active long trades in order: [(entry_date, trade_info), ...]
    active_short_trades = []  # List to track active short trades in order: [(entry_date, trade_info), ...]
    long_positions = 0  # Track number of open long positions
    short_positions = 0  # Track number of open short positions
    
    # Trade statistics
    trade_log = []  # Detailed record of each trade
    wins = []
    losses = []
    lengths = []
    
    # Loop through each bar (day)
    for i in range(len(df)):
        current_date = df.index[i]
        current_price = df['Close'].iloc[i]
        
        # First check if we need to exit short positions on buy signal
        if df['Buy_Signal'].iloc[i] and active_short_trades:
            # Exit all short positions when buy signal appears
            for entry_date, trade_info in active_short_trades:
                signal, entry_price, stop_loss, take_profit, trade_position_size, share_amount = trade_info
                
                # Calculate trade return for short
                trade_return = (entry_price - current_price) / entry_price * trade_position_size * portfolio_value
                
                # Update portfolio value
                portfolio_value += trade_return
                available_capital += (trade_position_size * initial_capital) + trade_return
                
                # Record trade result
                trade_pct = trade_return / (trade_position_size * portfolio_value) * 100
                
                if trade_return > 0:
                    wins.append(trade_pct)
                else:
                    losses.append(trade_pct)
                
                # Calculate trade length
                trade_length = (df.index.get_loc(current_date) - df.index.get_loc(entry_date))
                lengths.append(trade_length)
                
                # Log detailed trade information
                trade_log.append({
                    'Entry Date': entry_date,
                    'Exit Date': current_date,
                    'Direction': 'Short',
                    'Entry Price': entry_price,
                    'Exit Price': current_price,
                    'Position Size': trade_position_size * 100,
                    'Shares': share_amount,
                    'P&L ($)': trade_return,
                    'P&L (%)': trade_pct,
                    'Duration': trade_length,
                    'Exit Reason': 'Buy Signal'
                })
                
                # Update equity and trade_returns
                trade_returns.loc[current_date] += trade_pct
            
            # Clear all short positions
            active_short_trades = []
            short_positions = 0
        
        # Check for long position exits on sell signals
        if df['Sell_Signal'].iloc[i] and active_long_trades:
            # First close all long positions from oldest to newest
            trades_to_remove = []
            
            # Process trades from oldest to newest
            for trade_idx, (entry_date, trade_info) in enumerate(active_long_trades):
                signal, entry_price, stop_loss, trade_position_size, share_amount = trade_info
                
                # Calculate trade return
                trade_return = (current_price - entry_price) / entry_price * trade_position_size * portfolio_value
                
                # Update portfolio value
                portfolio_value += trade_return
                available_capital += (trade_position_size * initial_capital) + trade_return
                long_positions -= 1
                
                # Record trade result
                trade_pct = trade_return / (trade_position_size * portfolio_value) * 100
                
                if trade_return > 0:
                    wins.append(trade_pct)
                else:
                    losses.append(trade_pct)
                
                # Calculate trade length
                trade_length = (df.index.get_loc(current_date) - df.index.get_loc(entry_date))
                lengths.append(trade_length)
                
                # Log detailed trade information
                trade_log.append({
                    'Entry Date': entry_date,
                    'Exit Date': current_date,
                    'Direction': 'Long',
                    'Entry Price': entry_price,
                    'Exit Price': current_price,
                    'Position Size': trade_position_size * 100,
                    'Shares': share_amount,
                    'P&L ($)': trade_return,
                    'P&L (%)': trade_pct,
                    'Duration': trade_length,
                    'Exit Reason': 'Sell Signal'
                })
                
                # Update equity and trade_returns
                trade_returns.loc[current_date] += trade_pct
                
                # Mark this trade for removal
                trades_to_remove.append(trade_idx)
            
            # Remove exited trades
            for idx in sorted(trades_to_remove, reverse=True):
                active_long_trades.pop(idx)
            
            # After exiting all long positions, consider entering short positions
            if long_positions == 0 and short_positions < max_positions and available_capital > 0:
                # Calculate dynamic position size based on ATR volatility
                atr_position_adjust = 1.0
                if pd.notna(df['ATR'].iloc[i]) and df['ATR'].iloc[i] > 0:
                    atr_ratio = df['ATR'].iloc[i] / current_price
                    atr_position_adjust = 0.02 / atr_ratio
                    atr_position_adjust = max(0.5, min(atr_position_adjust, 1.5))
                
                # Calculate position size for this short trade
                trade_position_size = position_size * atr_position_adjust
                
                # Ensure we don't exceed available capital
                if trade_position_size * portfolio_value > available_capital:
                    trade_position_size = available_capital / portfolio_value
                
                # Calculate share amount
                share_amount = (trade_position_size * portfolio_value) / current_price
                
                # Short position setup
                signal = -1
                entry_price = current_price
                stop_loss = entry_price * (1 + risk)  # Stop loss for shorts moves up
                take_profit = entry_price * (1 - reward)  # Take profit for shorts moves down
                
                # Enter short trade
                active_short_trades.append((current_date, (signal, entry_price, stop_loss, take_profit, trade_position_size, share_amount)))
                available_capital -= trade_position_size * portfolio_value
                short_positions += 1
                
                # Log the entry to the trade log
                trade_log.append({
                    'Entry Date': current_date,
                    'Exit Date': None,
                    'Direction': 'Short',
                    'Entry Price': entry_price,
                    'Exit Price': None,
                    'Position Size': trade_position_size * 100,
                    'Shares': share_amount,
                    'P&L ($)': 0,
                    'P&L (%)': 0,
                    'Duration': 0,
                    'Exit Reason': 'Pending'
                })
        
        # Check for stop loss hits on long positions
        trades_to_remove = []
        for trade_idx, (entry_date, trade_info) in enumerate(active_long_trades):
            signal, entry_price, stop_loss, trade_position_size, share_amount = trade_info
            
            # Exit if stop loss is hit
            if current_price <= stop_loss:
                trade_return = (current_price - entry_price) / entry_price * trade_position_size * portfolio_value
                
                # Update portfolio value
                portfolio_value += trade_return
                available_capital += (trade_position_size * initial_capital) + trade_return
                long_positions -= 1
                
                # Record trade result
                trade_pct = trade_return / (trade_position_size * portfolio_value) * 100
                losses.append(trade_pct)
                
                # Calculate trade length
                trade_length = (df.index.get_loc(current_date) - df.index.get_loc(entry_date))
                lengths.append(trade_length)
                
                # Log detailed trade information
                trade_log.append({
                    'Entry Date': entry_date,
                    'Exit Date': current_date,
                    'Direction': 'Long',
                    'Entry Price': entry_price,
                    'Exit Price': current_price,
                    'Position Size': trade_position_size * 100,
                    'Shares': share_amount,
                    'P&L ($)': trade_return,
                    'P&L (%)': trade_pct,
                    'Duration': trade_length,
                    'Exit Reason': 'Stop Loss'
                })
                
                # Update equity and trade_returns
                trade_returns.loc[current_date] += trade_pct
                
                # Mark this trade for removal
                trades_to_remove.append(trade_idx)
        
        # Remove long trades that hit stop loss
        for idx in sorted(trades_to_remove, reverse=True):
            active_long_trades.pop(idx)
        
        # Check for take profit or stop loss on short positions
        trades_to_remove = []
        for trade_idx, (entry_date, trade_info) in enumerate(active_short_trades):
            signal, entry_price, stop_loss, take_profit, trade_position_size, share_amount = trade_info
            
            exit_trade = False
            exit_reason = ""
            trade_return = 0
            
            # Exit short if take profit is hit
            if current_price <= take_profit:
                trade_return = (entry_price - current_price) / entry_price * trade_position_size * portfolio_value
                exit_trade = True
                exit_reason = "Take Profit"
            # Exit short if stop loss is hit
            elif current_price >= stop_loss:
                trade_return = (entry_price - current_price) / entry_price * trade_position_size * portfolio_value
                exit_trade = True
                exit_reason = "Stop Loss"
            
            if exit_trade:
                # Update portfolio value
                portfolio_value += trade_return
                available_capital += (trade_position_size * initial_capital) + trade_return
                short_positions -= 1
                
                # Record trade result
                trade_pct = trade_return / (trade_position_size * portfolio_value) * 100
                
                if trade_return > 0:
                    wins.append(trade_pct)
                else:
                    losses.append(trade_pct)
                
                # Calculate trade length
                trade_length = (df.index.get_loc(current_date) - df.index.get_loc(entry_date))
                lengths.append(trade_length)
                
                # Log detailed trade information
                trade_log.append({
                    'Entry Date': entry_date,
                    'Exit Date': current_date,
                    'Direction': 'Short',
                    'Entry Price': entry_price,
                    'Exit Price': current_price,
                    'Position Size': trade_position_size * 100,
                    'Shares': share_amount,
                    'P&L ($)': trade_return,
                    'P&L (%)': trade_pct,
                    'Duration': trade_length,
                    'Exit Reason': exit_reason
                })
                
                # Update equity and trade_returns
                trade_returns.loc[current_date] += trade_pct
                
                # Mark this trade for removal
                trades_to_remove.append(trade_idx)
        
        # Remove short trades that hit take profit or stop loss
        for idx in sorted(trades_to_remove, reverse=True):
            active_short_trades.pop(idx)
        
        # Enter new long trades if buy signal is present and we have capacity
        if df['Buy_Signal'].iloc[i] and long_positions < max_positions and available_capital > 0:
            # Calculate dynamic position size based on ATR volatility
            atr_position_adjust = 1.0
            if pd.notna(df['ATR'].iloc[i]) and df['ATR'].iloc[i] > 0:
                atr_ratio = df['ATR'].iloc[i] / current_price
                atr_position_adjust = 0.02 / atr_ratio  # Target 2% volatility
                atr_position_adjust = max(0.5, min(atr_position_adjust, 1.5))
            
            # Calculate position size for this trade
            trade_position_size = position_size * atr_position_adjust
            
            # Ensure we don't exceed available capital
            if trade_position_size * portfolio_value > available_capital:
                trade_position_size = available_capital / portfolio_value
            
            # Calculate share amount
            share_amount = (trade_position_size * portfolio_value) / current_price
            
            # Long position setup
            signal = 1
            entry_price = current_price
            stop_loss = entry_price * (1 - risk)
            
            # Enter trade if we have capacity
            active_long_trades.append((current_date, (signal, entry_price, stop_loss, trade_position_size, share_amount)))
            available_capital -= trade_position_size * portfolio_value
            long_positions += 1
            
            # Log the entry to the trade log
            trade_log.append({
                'Entry Date': current_date,
                'Exit Date': None,
                'Direction': 'Long',
                'Entry Price': entry_price,
                'Exit Price': None,
                'Position Size': trade_position_size * 100,
                'Shares': share_amount,
                'P&L ($)': 0,
                'P&L (%)': 0,
                'Duration': 0,
                'Exit Reason': 'Pending'
            })
        
        # Update equity curve at the end of each day
        equity.iloc[i] = portfolio_value
    
    # Handle any open trades at the end of the simulation
    final_date = df.index[-1]
    final_price = df['Close'].iloc[-1]
    
    # Close any remaining long positions
    for entry_date, trade_info in active_long_trades:
        signal, entry_price, stop_loss, trade_position_size, share_amount = trade_info
        trade_return = (final_price - entry_price) / entry_price * trade_position_size * portfolio_value
        trade_pct = trade_return / (trade_position_size * portfolio_value) * 100
        
        if trade_return > 0:
            wins.append(trade_pct)
        else:
            losses.append(trade_pct)
        
        trade_length = (df.index.get_loc(final_date) - df.index.get_loc(entry_date))
        lengths.append(trade_length)
        
        trade_log.append({
            'Entry Date': entry_date,
            'Exit Date': final_date,
            'Direction': 'Long',
            'Entry Price': entry_price,
            'Exit Price': final_price,
            'Position Size': trade_position_size * 100,
            'Shares': share_amount,
            'P&L ($)': trade_return,
            'P&L (%)': trade_pct,
            'Duration': trade_length,
            'Exit Reason': 'End of Simulation'
        })
    
    # Close any remaining short positions
    for entry_date, trade_info in active_short_trades:
        signal, entry_price, stop_loss, take_profit, trade_position_size, share_amount = trade_info
        trade_return = (entry_price - final_price) / entry_price * trade_position_size * portfolio_value
        trade_pct = trade_return / (trade_position_size * portfolio_value) * 100
        
        if trade_return > 0:
            wins.append(trade_pct)
        else:
            losses.append(trade_pct)
        
        trade_length = (df.index.get_loc(final_date) - df.index.get_loc(entry_date))
        lengths.append(trade_length)
        
        trade_log.append({
            'Entry Date': entry_date,
            'Exit Date': final_date,
            'Direction': 'Short',
            'Entry Price': entry_price,
            'Exit Price': final_price,
            'Position Size': trade_position_size * 100,
            'Shares': share_amount,
            'P&L ($)': trade_return,
            'P&L (%)': trade_pct,
            'Duration': trade_length,
            'Exit Reason': 'End of Simulation'
        })
    
    # Clean up the trade log - remove pending entries
    final_trade_log = []
    for trade in trade_log:
        if trade['Exit Date'] is not None:
            final_trade_log.append(trade)
    
    # Create trade statistics summary
    trade_stats = {}
    
    # Basic trade metrics
    trade_stats['Total Trades'] = len(wins) + len(losses)
    trade_stats['Win Rate'] = len(wins) / trade_stats['Total Trades'] * 100 if trade_stats['Total Trades'] > 0 else 0
    trade_stats['Profit Factor'] = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else float('inf')
    
    # Return metrics
    trade_stats['Net Profit ($)'] = portfolio_value - initial_capital
    trade_stats['Net Profit (%)'] = (portfolio_value / initial_capital - 1) * 100
    trade_stats['Annualized Return (%)'] = ((portfolio_value / initial_capital) ** (252 / len(df)) - 1) * 100
    
    # Risk metrics
    trade_stats['Max Drawdown (%)'] = ((equity / equity.cummax()) - 1).min() * 100
    trade_stats['Avg Win (%)'] = np.mean(wins) if wins else 0
    trade_stats['Avg Loss (%)'] = np.mean(losses) if losses else 0
    trade_stats['Best Trade (%)'] = max(wins) if wins else 0
    trade_stats['Worst Trade (%)'] = min(losses) if losses else 0
    
    # Risk-adjusted metrics
    all_returns = np.array(wins + losses)
    daily_returns = equity.pct_change().dropna()
    
    # Sharpe ratio calculation
    daily_risk_free = risk_free_rate / 252
    sharpe_ratio = np.sqrt(252) * (daily_returns.mean() - daily_risk_free) / daily_returns.std() if len(daily_returns) > 1 else 0
    trade_stats['Sharpe Ratio'] = sharpe_ratio
    
    # Sortino ratio (downside risk only)
    negative_returns = daily_returns[daily_returns < 0]
    sortino_ratio = np.sqrt(252) * (daily_returns.mean() - daily_risk_free) / negative_returns.std() if len(negative_returns) > 1 else 0
    trade_stats['Sortino Ratio'] = sortino_ratio
    
    # Time metrics
    trade_stats['Avg Trade Duration'] = np.mean(lengths) if lengths else 0
    trade_stats['Max Trade Duration'] = max(lengths) if lengths else 0
    
    # Create dataframe from final trades log if we have any completed trades
    trade_log_df = pd.DataFrame(final_trade_log) if final_trade_log else pd.DataFrame(columns=[
        'Entry Date', 'Exit Date', 'Direction', 'Entry Price', 'Exit Price', 
        'Position Size', 'Shares', 'P&L ($)', 'P&L (%)', 'Duration', 'Exit Reason'
    ])
    
    # Add long vs short statistics
    if not trade_log_df.empty and 'Direction' in trade_log_df.columns:
        long_trades = trade_log_df[trade_log_df['Direction'] == 'Long']
        short_trades = trade_log_df[trade_log_df['Direction'] == 'Short']
        
        trade_stats['Long Trades'] = len(long_trades)
        trade_stats['Short Trades'] = len(short_trades)
        trade_stats['Long Win Rate'] = long_trades[long_trades['P&L (%)'] > 0].shape[0] / len(long_trades) * 100 if len(long_trades) > 0 else 0
        trade_stats['Short Win Rate'] = short_trades[short_trades['P&L (%)'] > 0].shape[0] / len(short_trades) * 100 if len(short_trades) > 0 else 0
        
        # Calculate average profitability by direction
        if len(long_trades) > 0:
            trade_stats['Avg Long Profit (%)'] = long_trades['P&L (%)'].mean()
        if len(short_trades) > 0:
            trade_stats['Avg Short Profit (%)'] = short_trades['P&L (%)'].mean()
    
    # Expectancy and system quality
    expectancy = (trade_stats['Win Rate']/100 * trade_stats['Avg Win (%)']) - ((100-trade_stats['Win Rate'])/100 * abs(trade_stats['Avg Loss (%)']))
    trade_stats['Expectancy (%)'] = expectancy
    
    return equity, trade_stats, trade_log_df

# --------------------------------------------------------------------------------------------------------------------------

def test_strategy(df, ticker, risk=DEFAULT_RISK, reward=DEFAULT_REWARD, position_size=DEFAULT_POSITION_SIZE):
    
    # Add technical indicators
    df = adding_indicators(df)
    
    # Run backtest with enhanced trade tracking
    portfolio_equity, trade_stats, trade_log = set_momentum_strat(df, risk=DEFAULT_RISK, reward=DEFAULT_REWARD, position_size=DEFAULT_POSITION_SIZE, 
                          max_positions=MAX_OPEN_POSITIONS, risk_free_rate=0.04)
    
    df = df.copy()
    # Add asset returns to dataframe for comparison
    df.loc[:, 'Asset_Returns'] = df['Close'].pct_change().fillna(0).cumsum()
    
    # Convert equity curve to returns for comparison
    df.loc[:, 'Strategy_Returns'] = (portfolio_equity / portfolio_equity.iloc[0] - 1)
    
    # Print summary stats
    print(f"\n=== {ticker} STRATEGY SUMMARY ===")
    print(f"Risk: {risk*100:.1f}% | Reward: {reward*100:.1f}% | Position Size: {position_size*100:.1f}%")
    print(f"Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} ({len(df)} trading days)")
    print(f"Starting Capital: ${portfolio_equity.iloc[0]:,.2f}")
    print(f"Ending Capital: ${portfolio_equity.iloc[-1]:,.2f}")
    print(f"Total Return: {trade_stats['Net Profit (%)']:.2f}%")
    print(f"Annual Return: {trade_stats['Annualized Return (%)']:.2f}%")
    print(f"Total Trades: {trade_stats['Total Trades']}")
    print(f"Win Rate: {trade_stats['Win Rate']:.2f}%")
    print(f"Profit Factor: {trade_stats['Profit Factor']:.2f}")
    print(f"Max Drawdown: {trade_stats['Max Drawdown (%)']:.2f}%")
    print(f"Sharpe Ratio: {trade_stats['Sharpe Ratio']:.2f}")
    print(f"Sortino Ratio: {trade_stats['Sortino Ratio']:.2f}")
    print(f"Expectancy: {trade_stats['Expectancy (%)']:.2f}%")
    
    # Create visualizations
    create_backtest_charts(df, portfolio_equity, trade_stats, trade_log, ticker)
    
    return {
        'ticker': ticker,
        'df': df,
        'equity': portfolio_equity,
        'trade_stats': trade_stats,
        'trade_log': trade_log
    }

    return

def create_backtest_charts(df, equity, trade_stats, trade_log, ticker):
    """
    Create comprehensive charts for backtest analysis.
    """
    # Set style
    sns.set(style="whitegrid", rc={"figure.figsize": (14, 8)})
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'lines.linewidth': 2.5,
    })
    
    # Define formatter functions
    def dollar_formatter(x, pos):
        return f'${x:,.0f}'
    def pct_formatter(x, pos):
        return f'{x:.1f}%'
    
    # Figure 1: Price chart with signals and indicators
    plt.figure(figsize=(16, 10))
    plt.subplot(3, 1, 1)
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df[f'{FAST}_ma'], label=f'{FAST}-day MA', linestyle='--')
    plt.plot(df[f'{SLOW}_ma'], label=f'{SLOW}-day MA', linestyle='--')
    plt.plot(df['Upper_Band'], label='Upper Bollinger', linestyle=':', color='gray')
    plt.plot(df['Lower_Band'], label='Lower Bollinger', linestyle=':', color='gray')
    # Signal markers
    buy_signals = df[df['Buy_Signal']]
    sell_signals = df[df['Sell_Signal']]
    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', s=50, alpha=0.3)
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', s=50, alpha=0.3)
    plt.title(f'{ticker} Price Chart with Signals')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)  # Add minor gridlines
    # RSI subplot
    plt.subplot(3, 1, 2)
    plt.plot(df['RSI'], color='purple', label='RSI')
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    plt.axhline(y=50, color='k', linestyle='--', alpha=0.3)
    plt.title('RSI')
    plt.ylabel('RSI Value')
    plt.legend()
    plt.grid(True)  # Add minor gridlines
    # MACD subplot
    plt.subplot(3, 1, 3)
    plt.plot(df['MACD'], color='blue', label='MACD')
    plt.plot(df['Signal'], color='red', label='Signal')
    plt.bar(df.index, df['MACD_hist'], color=np.where(df['MACD_hist'] > 0, 'green', 'red'), alpha=0.5, label='Histogram')
    plt.title('MACD')
    plt.ylabel('MACD Value')
    plt.legend()
    plt.grid(True)  # Add minor gridlines
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Equity curve with drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    # Equity curve
    ax1.plot(equity, color='blue', linewidth=2)
    ax1.set_title(f'{ticker} Portfolio Equity Curve')
    ax1.set_ylabel('Portfolio Value')
    ax1.grid(True)  # Add minor gridlines
    ax1.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
    # Calculate drawdown for visualization
    equity_peak = equity.cummax()
    drawdown = (equity / equity_peak - 1) * 100
    # Drawdown subplot
    ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax2.set_title('Drawdown (%)')
    ax2.set_ylabel('Drawdown %')
    ax2.set_ylim([drawdown.min() * 1.1, 0])  # Set y-axis limit slightly below minimum drawdown
    ax2.grid(True)  # Add minor gridlines
    ax2.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    plt.tight_layout()
    plt.show()
    
    # Figure 3: Comparative returns
    plt.figure(figsize=(16, 8))
    plt.plot(df['Asset_Returns'] * 100, label=f'{ticker} Buy & Hold', color='blue')
    plt.plot(df['Strategy_Returns'] * 100, label='Strategy', color='green')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    plt.title(f'Strategy vs {ticker} Buy & Hold Returns (%)')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True)  # Add minor gridlines
    #plt.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    plt.tight_layout()
    plt.show()
    
    # Figure 4: Trade analysis
    if not trade_log.empty:
        plt.figure(figsize=(16, 12))
        
        # Subplot 1: Trade P&L distribution
        plt.subplot(2, 2, 1)
        sns.histplot(trade_log['P&L (%)'], bins=20, kde=True)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title('Trade P&L Distribution (%)')
        plt.xlabel('P&L (%)')
        plt.ylabel('Frequency')
        
        # Subplot 2: Trade duration vs P&L
        plt.subplot(2, 2, 2)
        sns.scatterplot(data=trade_log, x='Duration', y='P&L (%)', hue='Direction', size='Position Size',
                        sizes=(50, 200), alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        plt.title('Trade Duration vs P&L')
        plt.xlabel('Duration (Days)')
        plt.ylabel('P&L (%)')
        
        # Subplot 3: Cumulative P&L by trade
        plt.subplot(2, 2, 3)
        trade_log['Cumulative P&L ($)'] = trade_log['P&L ($)'].cumsum()
        plt.plot(range(len(trade_log)), trade_log['Cumulative P&L ($)'], marker='o', markersize=3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        plt.title('Cumulative P&L by Trade')
        plt.xlabel('Trade Number')
        plt.ylabel('Cumulative P&L ($)')
        
        # Subplot 4: Additional trade analysis (e.g., win/loss ratio by direction)
        plt.subplot(2, 2, 4)
        win_loss_ratio = trade_log[trade_log['P&L ($)'] > 0]['P&L (%)'].mean() / \
                         trade_log[trade_log['P&L ($)'] < 0]['P&L (%)'].mean()
        plt.axhline(y=1, color='green', linestyle='--')
        plt.axhline(y=-1, color='red', linestyle='--')
        sns.scatterplot(data=trade_log, x='Direction', y=win_loss_ratio)
        plt.title('Win/Loss Ratio by Direction')
        plt.xlabel('Direction (Buy/Sell)')
        plt.ylabel('Win/Loss Ratio')
    else:
        print("Trade log is empty. No trade analysis will be shown.")
        
    return

# --------------------------------------------------------------------------------------------------------------------------

def main():
    df, data_dict = get_data(TICKER)
    df = test_strategy(df, TICKER)
    return df

main()