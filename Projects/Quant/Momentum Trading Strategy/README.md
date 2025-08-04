# Momentum Trading Strategy: Quantitative Analysis and Optimization
### Project Outline

This project implements a quantitative momentum trading strategy for financial assets, exemplified using SPY (S&P 500 ETF). It focuses on identifying and capitalizing on prevailing market trends by employing a variety of technical indicators and statistical methods. The core of the project involves data acquisition, signal generation, trade execution logic, strategy optimization, and rigorous backtesting, including Monte Carlo simulations and Walk-Forward Analysis to assess robustness and significance.

---
<br>

## Project Scope and Strategy Framework

The `fourelement.py` implementation represents a sophisticated momentum trading strategy. This strategy synthesizes five key elements of market dynamics: price trend, RSI-based momentum, ADX-based trend strength, volume acceleration, and a VIX-based market sentiment filter. This multi-element approach aims to provide a robust signaling mechanism.

### Strategy Evolution Roadmap
Future implementations will explore various momentum approaches including:
- Price-based momentum strategies (time-series and cross-sectional)
- Volume-weighted momentum indicators
- Sector rotation based on relative strength
- Multi-timeframe momentum convergence
- Adaptive momentum strategies using machine learning

### Validation Framework
Central to this project's methodology is the rigorous validation of each strategy through:

1.  **Monte Carlo Simulation**
    *   Employs stationary bootstrap (`stationary_bootstrap` function) for synthetic price series generation, with optimal block length determined by `determine_optimal_block_length`.
    *   Preserves temporal dependencies and statistical properties.
    *   Assesses strategy significance against random chance.
    *   Quantifies confidence intervals for performance metrics.

2.  **Walk-Forward Analysis (WFA)**
    *   Tests strategy robustness across different market regimes (`walk_forward_analysis` function).
    *   Validates parameter stability through time.
    *   Minimizes overfitting risk through out-of-sample testing (`OOS_WINDOW`).
    *   Provides realistic performance expectations.

This systematic approach to strategy validation ensures that any momentum strategy developed within this framework meets rigorous statistical standards before consideration for live trading.

---
<br>

## Execution

The project is structured into several key components, from data preparation to advanced strategy validation:

### 1. Data Acquisition and Preparation
   - **Data Source**: Historical daily price data (Open, High, Low, Close, Volume, and VIX data) for a specified ticker (e.g., `TICKER = 'SPY'`) is downloaded (handled by `get_data` function).
   - **Indicator Calculation**: The `prepare_data` function enriches raw data with technical indicators. These include:
     - Moving Averages (e.g., `FAST_ma`, `SLOW_ma`, `Weekly_MA{WEEKLY_MA_PERIOD}`)
     - Relative Strength Index (`RSI`)
     - Bollinger Bands (`Upper_Band`, `Lower_Band`)
     - Average True Range (`ATR`)
     - Average Directional Index (`ADX`)
     - Volume Moving Average (`Volume_MA20`)
     - Raw components for the five-element score (e.g., `price_roc_raw`, `ma_dist_raw`, `adx_slope_raw`, `vol_accel_raw`, `vix_factor_raw`).
   - **Data Splitting**: Data is managed for in-sample (`IS_HISTORY_DAYS`, `FINAL_IS_YEARS`) and out-of-sample (`FINAL_OOS_YEARS`) periods, especially for WFA.

---

### 2. Core Momentum Strategy Logic
   - **Signal Generation** (primarily in the `signals` function):
     - A composite momentum score is derived from five weighted elements defined in `DEFAULT_SIGNAL_PROCESSING_PARAMS['weights']`:
       1.  `price_trend`: Based on price rate-of-change and distance from moving average.
       2.  `rsi_zone`: Based on RSI levels.
       3.  `adx_slope`: Based on the slope of the ADX.
       4.  `vol_accel`: Based on volume acceleration.
       5.  `vix_factor`: Based on VIX levels relative to its moving average.
     - Raw indicator values are ranked over `RANKING_LOOKBACK_WINDOW` to normalize and capture relative strength.
     - Entry and exit signals are generated when this composite score crosses dynamic thresholds defined in `DEFAULT_SIGNAL_PROCESSING_PARAMS['thresholds']` (`buy_score`, `exit_score`, `immediate_exit_score`).
     - Additional conditions like ADX being above `ADX_THRESHOLD_DEFAULT` and VIX being below `VIX_ENTRY_THRESHOLD` act as filters.
   - **Trade Management** (handled by the `momentum` function and `TradeManager` class):
     - **Initial Capital**: Starts with `INITIAL_CAPITAL`.
     - **Position Management**: Manages up to `MAX_OPEN_POSITIONS`. Each position's duration is limited by `MAX_POSITION_DURATION`.
     - **Risk Management**: Stop-loss per trade is based on `DEFAULT_LONG_RISK` (percentage of capital or ATR-based, applied within the `momentum` logic). No explicit take-profit parameter; exits are score-based, duration-based, or stop-loss.
   - **Volatility Consideration**: The strategy uses ATR for general volatility assessment and VIX (`vix_factor` in score, `VIX_ENTRY_THRESHOLD` filter) to adapt to market sentiment. A `vol_adjustment` factor, derived from ranked ATR percentage, can further modulate signals or risk.

---

### 3. Strategy Optimization
   - **Objective Functions**: The `objectives` function evaluates a trial's performance during optimization. Parameters from `DEFAULT_SIGNAL_PROCESSING_PARAMS` (weights, thresholds) and other key settings (e.g., `ADX_THRESHOLD_DEFAULT`, `DEFAULT_LONG_RISK`, indicator lookbacks) are typically optimized.
   - **Optimization Metrics**: The `OPTIMIZATION_DIRECTIONS` and `OBJECTIVE_WEIGHTS` dictionaries define goals, such as maximizing Profit Factor, Average Win/Loss Ratio, Expectancy, and minimizing Maximum Drawdown.
   - **Methodology**: The `optimize` function uses an iterative process (e.g., Optuna, suggested by `trial` object) to find the best parameter set over in-sample data (`IS_HISTORY_DAYS`). `TRIALS` defines the number of optimization attempts.

---

### 4. Backtesting and Performance Analysis
   - **Simulation**: The `momentum` function serves as the backtesting engine, simulating trades and calculating equity over historical data.
   - **Key Performance Indicators (KPIs)**: The `trade_statistics` function calculates metrics:
     - Total Return, Annualized Return
     - Sharpe Ratio (using `RISK_FREE_RATE_ANNUAL`), Sortino Ratio
     - Maximum Drawdown
     - Win Rate, Loss Rate, Average Win, Average Loss, Profit Factor, Expectancy
     - Number of Trades
   - **Commission**: Trading commissions (`COMMISION = True`) can be included.

---

### 5. Robustness and Significance Testing
   - **Monte Carlo Simulation** (`monte_carlo` function):
     - Uses `stationary_bootstrap` with `BLOCK_SIZE` for synthetic series.
     - Runs the strategy (using parameters, potentially from `pareto_front` of optimization) on `num_simulations` synthetic series.
     - Assesses performance distribution and significance.
   - **Walk-Forward Analysis (WFA)** (`walk_forward_analysis` function):
     - Optimizes parameters on a rolling in-sample window (`WFA_HISTORY_DAYS` initially, then `OPTIMIZATION_FREQUENCY` defines re-optimization period).
     - Tests on subsequent out-of-sample periods (`OOS_WINDOW`).
     - Assesses real-world applicability and parameter stability.

---
<br>

## Key Technical Indicators and Concepts

| Indicator/Concept             | Description                                                                                                | Role in Strategy                                                                                                |
|-------------------------------|------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| **Moving Averages (MA)**      | Smooth price data to identify trend.                                                                       | Core trend input for 'price_trend' element; `FAST`, `SLOW`, `WEEKLY_MA_PERIOD`.                                 |
| **Relative Strength Index (RSI)** | Measures speed/change of price movements.                                                                  | Input for 'rsi_zone' element; `RSI_LENGTH`.                                                                     |
| **Average Directional Index (ADX)** | Measures trend strength.                                                                                   | Input for 'adx_slope' element and direct filter (`ADX_THRESHOLD_DEFAULT`); `ADX_LENGTH`.                        |
| **Average True Range (ATR)**  | Measures market volatility.                                                                                | Input for 'vol_adjustment' factor, potentially for dynamic risk sizing; `ATR_LENGTH`.                           |
| **Bollinger Bands**           | Measure volatility and potential breakouts.                                                                | Used in `prepare_data`; `BB_LEN`, `ST_DEV`. Not directly in the five-element score but available for analysis. |
| **Volume Analysis**           | Volume MA and acceleration.                                                                                | Input for 'vol_accel' element.                                                                                  |
| **Composite Momentum Score**  | Weighted sum of ranked elements (price trend, RSI, ADX slope, volume accel, VIX factor).                   | Primary driver for buy/sell decisions based on `buy_score`, `exit_score` thresholds.                            |
| **VIX (Volatility Index)**    | Measures market expectation of 30-day volatility.                                                          | Input for 'vix_factor' element and direct filter (`VIX_ENTRY_THRESHOLD`); `VIX_MA_PERIOD`.                      |
| **Component Ranking**         | Raw indicator values are ranked over `RANKING_LOOKBACK_WINDOW`.                                            | Normalizes inputs to the composite score, making them comparable.                                               |

---

## Strategy Parameters

The strategy's behavior is controlled by parameters in `fourelement.py`, categorized for clarity. These are primary targets for optimization.

| Parameter Category                 | Example Parameters from `fourelement.py`                                                                 | Description                                                                                                                                  |
|------------------------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| **Signal Processing & Thresholds** | `DEFAULT_SIGNAL_PROCESSING_PARAMS` (weights for 5 elements, `buy_score`, `exit_score`, `immediate_exit_score` thresholds), `ADX_THRESHOLD_DEFAULT`, `VIX_ENTRY_THRESHOLD` | Defines how raw signals are combined and the levels that trigger trading actions.                                                              |
| **Risk & Trade Management**        | `DEFAULT_LONG_RISK`, `MAX_OPEN_POSITIONS`, `MAX_POSITION_DURATION`, `INITIAL_CAPITAL`, `COMMISION`         | Controls risk per trade, portfolio concentration, max trade holding time, starting capital, and commission effects.                          |
| **Indicator Lookbacks & Settings** | `FAST`, `SLOW`, `WEEKLY_MA_PERIOD`, `RSI_LENGTH`, `ADX_LENGTH`, `ATR_LENGTH`, `BB_LEN`, `ST_DEV`, `MOMENTUM_LOOKBACK`, `MOMENTUM_VOLATILITY_LOOKBACK`, `RANKING_LOOKBACK_WINDOW`, `VIX_MA_PERIOD` | Lookback periods and settings for calculating technical indicators.                                                                          |
| **Optimization Control**           | `OPTIMIZATION`, `TRIALS`, `TARGET_SCORE`, `SCORE_TOLERANCE`, `OPTIMIZATION_DIRECTIONS`, `OBJECTIVE_WEIGHTS` | Parameters governing the optimization process itself.                                                                                        |
| **Operational & Data Control**     | `TICKER`, `IS_HISTORY_DAYS`, `FINAL_OOS_YEARS`, `FINAL_IS_YEARS`, `RISK_FREE_RATE_ANNUAL`                  | Defines the asset, data periods for various analyses, and risk-free rate for calculations.                                                   |
| **Monte Carlo & WFA Control**      | `BLOCK_SIZE`, `OPTIMIZATION_FREQUENCY`, `OOS_WINDOW`, `WFA_HISTORY_DAYS`                                   | Parameters specific to Monte Carlo simulations and Walk-Forward Analysis.                                                                    |

---

## Operational Modes

The `fourelement.py` script can be run in several modes, controlled by the `TYPE` parameter:
1.  **`TYPE = 1` (Full Run)**: Executes the strategy over the entire dataset with fixed parameters (typically for final review after optimization).
2.  **`TYPE = 2` (Walk-Forward Analysis)**: Performs WFA as described.
3.  **`TYPE = 3` (Monte Carlo Simulation)**: Conducts Monte Carlo testing.
4.  **`TYPE = 4` (Optimization)**: Runs the parameter optimization process.
5.  **`TYPE = 5` (Test)**: A specific mode for isolated testing or debugging, often with a smaller dataset or specific parameter set (as used in the `test` function).

---

## Technical Implementation Details
- **Libraries**: Core libraries include `pandas` for data manipulation, `numpy` for numerical operations, `scipy` and `statsmodels` for statistical calculations.
- **Performance**:
    - `numba` is used with `@jit(nopython=True)` for Just-In-Time compilation of computationally intensive functions to speed up execution.
    - `joblib` (`Parallel`, `delayed`) and `multiprocessing` (`Pool`) are imported, suggesting capabilities for parallel processing, likely used in optimization, Monte Carlo simulations, or data preparation tasks.
- **Output**: The `tabulate` library is used for formatting results into tables for display.

---
<br>

## Future Enhancements & Considerations
- **Dynamic Parameter Adjustment**: Explore methods to dynamically adjust strategy parameters based on changing market volatility or regime.
- **Cost Analysis**: More detailed modeling of transaction costs, including slippage.
- **Portfolio Application**: Extend the single-asset framework to a portfolio context, considering asset correlation and allocation.
- **Machine Learning Integration**: Further explore ML techniques for signal generation or regime identification, as mentioned in the roadmap.
- **Enhanced VIX Integration**: While VIX is used, further research could refine its application in modulating risk or signal strength more dynamically.