import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

# --------------------------------------------------------------------------------------------------------------------------

# Base parameters
TICKER = 'SPY'
LOOKBACK = 300

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

# --------------------------------------------------------------------------------------------------------------------------

def get_data():
    df = yf.download(TICKER)
    df.columns = df.columns.get_level_values(0)
    return df.iloc[-LOOKBACK:, :]

# --------------------------------------------------------------------------------------------------------------------------
# Functions for the strategies

def adding_indicators(df):
    # Moving Averages
    df[f'{FAST}_ma'] = df['Close'].rolling(FAST).mean()
    df[f'{SLOW}_ma'] = df['Close'].rolling(SLOW).mean()
    df['MACross_Strategy'] = np.where(df[f'{FAST}_ma'] > df[f'{SLOW}_ma'], 1, -1)

    # MACD
    df['MACD_fast'] = df['Close'].ewm(span=MACD_FAST).mean()
    df['MACD_slow'] = df['Close'].ewm(span=MACD_SLOW).mean()
    df['MACD'] = df['MACD_fast'] - df['MACD_slow']
    df['Signal'] = df['MACD'].ewm(span=MACD_SPAN).mean()
    df['MACD_Strategy'] = np.where(df['MACD'] > df['Signal'], 1, -1)

    # RSI
    price_diff = df['Close'].diff()
    gain = price_diff.where(price_diff > 0, 0)
    loss = -price_diff.where(price_diff < 0, 0)
    avg_gain = gain.rolling(window=RSI_LENGTH).mean()
    avg_loss = loss.rolling(window=RSI_LENGTH).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_Strategy'] = np.where(df['RSI'] < RSI_OVERSOLD, 1,
                                  np.where(df['RSI'] > RSI_OVERBOUGHT, -1, 0))

    # MFI
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = np.where(typical_price.diff() > 0, money_flow, 0)
    negative_flow = np.where(typical_price.diff() < 0, money_flow, 0)
    pos_flow_sum = pd.Series(positive_flow).rolling(window=MFI_LENGTH).sum()
    neg_flow_sum = pd.Series(negative_flow).rolling(window=MFI_LENGTH).sum()
    mfr = pos_flow_sum / (neg_flow_sum + 1e-10)
    df['MFI'] = 100 - (100 / (1 + mfr))
    df['MFI_Strategy'] = np.where(df['MFI'] < MFI_OVERSOLD, 1,
                                  np.where(df['MFI'] > MFI_OVERBOUGHT, -1, 0))

    # Bollinger Bands
    df['BB_SMA'] = df['Close'].rolling(BB_LEN).mean()
    df['BB_STD'] = df['Close'].rolling(BB_LEN).std()
    df['Upper_Band'] = df['BB_SMA'] + (DEVS * df['BB_STD'])
    df['Lower_Band'] = df['BB_SMA'] - (DEVS * df['BB_STD'])
    df['BB_Strategy'] = np.where(df['Close'] > df['Upper_Band'], 1,
                                 np.where(df['Close'] < df['Lower_Band'], -1, 0))

    # Combine all strategies (entry when all aligned bullish)
    df['Strategy'] = np.where((df['MACross_Strategy'] == 1) &
                              (df['MACD_Strategy'] == 1) &
                              (df['RSI_Strategy'] == 1) &
                              (df['MFI_Strategy'] == 1) &
                              (df['BB_Strategy'] == 1), 1, -1)

    df['Strategy'] = df['Strategy'].shift(1)  # shift to avoid lookahead bias

    return df.dropna()

# --------------------------------------------------------------------------------------------------------------------------

def test_strategy(df):
    df['returns'] = df['Close'].pct_change()
    df['strategy_returns'] = df['returns'] * df['Strategy']

    df['asset_cumulative_returns'] = (1 + df['returns']).cumprod() - 1
    df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod() - 1

    if df['strategy_returns'].std() != 0:
        sharpe_ratio = np.sqrt(252) * (df['strategy_returns'].mean() / df['strategy_returns'].std())
    else:
        sharpe_ratio = 0

    print(f'Sharpe Ratio: {sharpe_ratio:.2f}')

    # Plot
    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df.index, df['asset_cumulative_returns']*100, label='Asset Cumulative Returns', color='#2E86C1')
    ax.plot(df.index, df['cumulative_strategy_returns']*100, label='Strategy Cumulative Returns', color='#28B463')

    # Enhance the plot
    ax.set_title(f'{TICKER} Cumulative Returns vs Strategy Returns', fontsize=14, pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Returns (%)', fontsize=12)
    ax.legend(fontsize=10, frameon=True)
    
    # Add both major and minor gridlines
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':')
    ax.minorticks_on()  # Enable minor ticks
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}%'.format(y)))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return df

# --------------------------------------------------------------------------------------------------------------------------

def main():
    df = get_data()
    df = adding_indicators(df)
    df = test_strategy(df)
    return df

# Run the strategy
main()