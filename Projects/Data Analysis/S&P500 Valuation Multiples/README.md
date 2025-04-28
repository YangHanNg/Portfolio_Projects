# S&P 500 Valuation Multiples: Statistical Analysis of Trading Multiple Pricing 
### Project Outline

This is a data analysis project inspired by NYU Stern Business School Prof. Aswath Damodaran from his Valuations course. It is a relative valuation method built upon traditional comparable companies analysis. It is based upon Discounted Cash Flow (DCF) calculations to derive Enterprise Value (EV) using metrics like Weighted Average Cost of Capital (WACC), Reinvestment Rate (RIR), Expected Growth Rate Etc. These financial metrics represent independent variables for dependent variable EV, as such, it allows the opportunity for a linear regression study to interpret the performances of companies relative to their industry peers. The regressed line would serve as a projection tool built on the performance of the industry, and thus, it presents the concept of under/over-pricing for companies as their current financial metrics are projected forward. A possible strategy to this analysis insight would be long companies where theoretical trading mulitple are greater than its current value and short companies whose theoretical trading mulitple are projected to decrease. 

The project utilises AlphaVantage API to gather financial data from companies listed in the S&P 500 index. The data includes annual and quarterly Income Statement and Balance Sheet information which are stored in a PostgreSQL database for efficient data management and analysis. The project implements a caching system to track processing progress and avoid redundant API calls, making it resilient to interruptions and mindful of API rate limits. The SP500 CSV data is sourced from Kaggle in late 2024 and consolidate with a few data points published by Prof. Aswath.

---
<br>

## Execution

The project is broken into three distinct stages to tackle specific goals of the project. The first stage involve the creation and configuration of the database, as it is the base to which the requested data are stored. A function execute requests through API keys to append the data points accordingly. The next stage involve the bulk of the project operation where financial data are used to compute metrics, placed through statistical tests and a regression model, plotted 3-dimensionally for visualisation, and stored additionaly into the database. It is a long automated pipeline that processes industries within the S&P 500 index to produce the crucial data points we are aiming for. The last stage focuses on visualising the derived data points and validifying their models from the statistical tests. It aims to be present a company and its peer industry like a comparable companies analysis output page with an added focus on the theoretical EV trading multiples.

---

### Program 1: Data Retrival from Alpha Vantage API and Storage to Local PostgreSQL Database

To maximise the efficiency of data gathering and reducing the redundancy of API calls, Program 1 implements a robust data retrieval and storage system. The program first initializes a PostgreSQL database with a comprehensive schema designed to store company information, financial reports, and various metrics. Key features include:

1. **Database Schema**:
   - Companies table for storing basic company information including sector and industry classification
   - Industries table containing sector-specific metrics like cost of capital and growth rates
   - ReportingPeriods table to track fiscal periods
   - FinancialReports and FinancialData tables for storing the actual financial data
   - Additional tables for calculated metrics and regression analysis results

2. **Data Processing**:
   - Processes companies by industry groups (small: <6 companies, medium: 6-12 companies, large: >12 companies)
   - Implements a caching system to track progress and resume interrupted processing
   - Handles both annual and quarterly reports with configurable limits
   - Includes robust error handling and logging

3. **API Integration**:
   - Manages API rate limits with configurable delays between calls
   - Processes both Balance Sheet and Income Statement data
   - Validates and transforms data before storage

The data structure retrieved from AlphaVantage follows a standardized JSON format, with separate endpoints for Balance Sheet and Income Statement data shown below. The program processes this data and stores it in a normalized database structure for efficient querying and analysis.

<p align="center">
  <img src="figures/example_balance_sheet_data.png" width="400"/>
  <img src="figures/example_income_statement_data.png" width="400"/>
</p>

<p align="center">
  <b>Figure 1:</b> Example Balance Sheet data | <b>Figure 2:</b> Example Income Statement data
</p>

---

### Program 2: Database Requests, Metric Calculations, and Regression Analysis

Program 2 focuses on retrieving the stored financial data and performing complex calculations to derive various financial metrics and conduct regression analysis. The program features:

1. **Financial Metrics Calculation**:
   - Computes key metrics including:
     - Working Capital and Net Working Capital changes
     - Revenue Growth and Operating Margins
     - EBITDA and related margins
     - Reinvestment Rate and Return on Invested Capital
     - Expected Growth rates and Beta calculations

2. **Industry Analysis**:
   - Processes companies within their industry groups
   - Calculates industry-specific metrics and benchmarks
   - Incorporates industry-level cost of capital and reinvestment rates

3. **Regression Analysis**:
   - Performs statistical analysis on various EV multiples
   - Conducts diagnostic tests including:
     - Heteroscedasticity testing (Breusch-Pagan)
     - Multicollinearity analysis (VIF)
     - Normality checks on residuals
   - Generates confidence intervals and statistical significance measures

4. **Data Storage**:
   - Stores calculated metrics in dedicated tables
   - Maintains regression results including coefficients, diagnostics, and visualizations
   - Preserves analysis history for trending and comparison

---

### Stage 3: SQL Queries, Data Visualisation, and Statistical Validation


---
<br>

## EV Multiples and Financial Metric Methodology

This section describes how EV multiples are calculated using company financial metrics. An intrinsic DCF framework is applied to approximate the relationship between financial ratios and EV multiples. Project limitations are discussed, focusing on assumptions and generalizations necessary for large-scale S&P 500 analysis.

### Fundamental DCF-Based Approach

- EV is estimated using Free Cash Flow to Firm (FCFF), WACC, and Growth Rate:

$$ EV = \frac{FCFF}{WACC - Growth\ Rate} $$

- FCFF is derived as:

$$ FCFF = EBIT \times (1 - Tax\ Rate) \times (1 - RIR) $$

---

### Key Financial Metrics Used

| Metric | Notation | Formula |
|--------|----------|---------|
| Revenue Growth Rate | **Rev. Growth** | $\left( 1 - \frac{Current\ Revenue}{Previous\ Revenue} \right) $|
| After-Tax Operating Margin | **ATOM** | $\( EBIT \times (1 - Tax\ Rate) \)$ |
| Reinvestment Rate | **RIR** | $\left( \frac{Free\ Cash\ Flow}{ATOM} \right)$ |
| Sales to Capital | **SC** | $\left( \frac{Sales}{Capital\ Invested} \right)$ |
| Return on Invested Capital | **ROIC** | $\( ATOM \times Sales\ to\ Capital \)$ |
| Percentage D&A | **DA%** | $\left( \frac{DA}{EBITDA} \right)$ |
| Expected Growth Rate | **Exp. Growth** | $\( RIR \times ROIC \)$ |
| Industry WACC | **IWACC** | Industry Average WACC |
| Industry Reinvestment Rate | **IRIR** | Industry Average RIR |

---

### EV/EBIT Derivation Example

- Rearranging DCF terms leads to:

$$ \frac{EV}{EBIT} = \frac{(1 - Tax\ Rate) \times (1 - RIR)}{WACC - Growth\ Rate} $$

- EV/EBIT multiple is a function of Tax Rate, WACC, and Growth Rate.
- Similar logic applies to derive relationships for other EV multiples.

---

### Two-Stage DCF Extension

- To account for growth and terminal value, a two-stage DCF logic is applied:

$$
\frac{EV}{Metric} = Scaling\ Factor \times \left[\frac{(1 - RIR)(1 + Rev.\ Growth)\left(1 - \frac{(1 + Rev.\ Growth)^t}{(1 + WACC)^t}\right)}{WACC - Rev.\ Growth} + \frac{(1 - IRIR)(1 + Rev.\ Growth)^t(1 + Exp.\ Growth)}{(IWACC - Exp.\ Growth)(1 + WACC)^t}\right]
$$

- \( t \) represents the number of projection years.

| EV Metric | Scaling Factor | Proof |
|-----------|----------------|-------|
| EV/EBIT | \( 1/(1 - tax) \) |  |
| EV/EBIT(1-Tax) | 1 | EBIT(1-tax)/EBIT(1-tax) |
| EV/EBITDA | \( (1 - DA\%) (1 - tax) \) | \( EBIT(1-tax) = (EBITDA - DA)(1-tax) \) |
| EV/Sales | ATOM | \( EBIT(1-tax)/Sales = ATOM \) |
| EV/Capital Invested | ROIC | \( EBIT(1-tax) = ROIC \times IC \) |

---

## Limitations

Several limitations affect this large-scale EV multiple estimation:

---

### 1. Assumptions of the Model

- A uniform 3-year projection is assumed across companies, balancing growth and maturity.
- Growth, reinvestment rates, and expected growth are approximated using 3-year historical averages.
- Using backward-looking averages introduces bias, particularly for extreme events (e.g., COVID-19 pandemic).
- Terminal values use industry average WACC and RIR but retain company-specific 3-year expected growth.

---

### 2. WACC Calculation Simplifications

- WACC is based on average market risk premiums and credit default risks rather than firm-specific data.
- Ideal WACC calculation requires:
  - Specific credit ratings (e.g., S&P, Moody's, Fitch)
  - Accurate ERP per firm's operational market
- Due to data access limitations, WACC generalizations assume companies operate primarily in U.S. markets.

WACC formula:

$$
WACC = Cost\ of\ Debt \times (1 - Tax\ Rate) \times \left(\frac{Debt}{Debt+Equity}\right) + Cost\ of\ Equity \times \left(1 - \frac{Debt}{Debt+Equity}\right)
$$

where:

$$
Cost\ of\ Debt = Riskfree\ Rate + Credit\ Default\ Risk
$$

and

$$
Cost\ of\ Equity = Riskfree\ Rate + Levered\ Beta \times Equity\ Risk\ Premium
$$

---

### 3. Industry Variables and External Sources

- Industry averages for WACC, unlevered beta, and RIR are sourced from Prof. Aswath Damodaran.
- Individual company differences are not captured, which may lead to generalization bias.
- Collecting firm-specific betas, capital structures, and reinvestment behaviors was out of project scope.

---
<br>

## Statistics Technicals

Will be populated soon...

---
<br>

## References
To be added....
