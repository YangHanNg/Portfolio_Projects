# Portfolio Projects

Hey there! Welcome to the landing page to where I plan to store my projects. 

Hopefully they are organised to your liking and that all README files would fill you in on what the projects and files entail.

>## How to Use:
>#### Clone the repo
>
>git clone https://github.com/yourusername/project-name.git
>
>cd project-name
>
>
>#### Activate virtual environment (optional)
>
>python -m venv venv
>
>source venv/bin/activate  # or venv\Scripts\activate on Windows
>
>
>#### Install dependencies
>
>pip install -r requirements.txt
>
>
>#### Run notebook or script
>
>jupyter notebook notebooks/analysis.ipynb


# Projects

The following will be a rolling descriptions of projects I am doing, their brief summaries, highlights, and tools that were used. 

<br>

## #1 S&P500 Valuation Multiple: Statistical Analysis of Trading Multiple Pricing 

Last Updated: 27/Apr/2025 &nbsp; &nbsp; Status: Finished

### Summary
A data analysis project that is an extension to comparable companies analysis, a relative valuation method used in the finance industry. The project aims to analyse Enterprise Value (EV) trading mutliples of S&P 500 listed companies. Rather than using traditional methods of mean and median statistics on peer companies, a linear regression study is used to interpret the performances of companies relative to their industry peers. The regression model uses various internal financial metrics to determine the pricing of trading multiples, and would therefore predict the potential trading multiple of companies given the performance of the industry in which they operate.

### Key Features

- Implementd an end-to-end data pipeline sourcing S&P 500 listed companies from AlphaVantage API to locally managed PostgreSQL database.
- Data from financial statements are computed to derive Enterprise Value (EV) trading multiples for linear regression models.
- Financial metrics such as Cost of Capital (WACC), Reinvestment Rate (RIR), Revenue Growth (Rev. Growth), After-tax Operating Margin (ATOM), and others are used as indepedent variables for the regression.
- Visualisations, statistical data, and computed financial metrics are upload into the database and assigned according to the data relation.
- Finalised database uses SQL queries to retrieve and visualise insights in PowerBI or Excel according to industries.

### Tools & Libraries Used
- pandas, numpy
- matplotlib, seaborn
- statsmodels / scikit-learn
- JSON / psycopg2

## #2 Momentum Trading Strategy: Quantitative Analysis and Optimization

Last Updated: 23/May/2025 &nbsp; &nbsp; Status: Ongoing

### Summary
This project implements a quantitative momentum trading strategy for financial assets, exemplified using SPY (S&P 500 ETF). It focuses on identifying and capitalizing on prevailing market trends by employing a variety of technical indicators and statistical methods. The core of the project involves data acquisition, signal generation, trade execution logic, strategy optimization, and rigorous backtesting, including Monte Carlo simulations and Walk-Forward Analysis to assess robustness and significance.

### Key Features
- Data acquisition and preparation for financial instruments.
- Calculation of a suite of technical indicators (Moving Averages, RSI, Bollinger Bands, ATR, ADX).
- Core momentum strategy logic with signal generation based on a composite momentum score.
- Trade management including position sizing and risk management (stop-loss/take-profit).
- Strategy optimization framework using weighted objectives (Sharpe Ratio, Profit Factor, etc.).
- Backtesting engine to simulate trades and calculate comprehensive performance KPIs.
- Robustness testing via Monte Carlo simulations and Walk-Forward Analysis.

### Tools & Libraries Used
- pandas, numpy
- numba (for JIT compilation)
- multiprocessing, joblib (for parallel execution)
- tabulate (for results presentation)
- yfinance (implicitly, for data acquisition)

## #3 Factor Zoo Explorer

Last Updated: 23/May/2025 &nbsp; &nbsp; Status: Currently Cooking

### Summary
Currently Cooking

### Key Features
Currently Cooking

### Tools & Libraries Used
Currently Cooking

## #4 Market-Making Simulator

Last Updated: 23/May/2025 &nbsp; &nbsp; Status: Currently Cooking

### Summary
Currently Cooking

### Key Features
Currently Cooking

### Tools & Libraries Used
Currently Cooking

## #5 Options Volatility Arbitrage

Last Updated: 23/May/2025 &nbsp; &nbsp; Status: Currently Cooking

### Summary
Currently Cooking

### Key Features
Currently Cooking

### Tools & Libraries Used
Currently Cooking
