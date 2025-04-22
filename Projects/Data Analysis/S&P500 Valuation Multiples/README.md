# S&P 500 Valuation Multiples: Linear Regression Data Analysis
### Project Outline

This is a data analysis project inspired by NYU Stern Business School Prof. Aswath Damodaran from his Valuations course. It is a relative valuation method built upon traditional comparable companies analysis. It is based upon DCF calculations to derive Enterprise Value (EV) using metrics that is Weighted Average Cost of Capital (WACC), Reinvestment Rate (RIR), Expected Growth Rate Etc. These financial metrics represent independent variables for dependent variable EV, as such, it allows the opportunity for a linear regression study to interpret the performances of companies relative to their industry peers. The regressed line would serve as a projection tool built on the performance of the industry, and thus, it presents the idea of under/over-pricing of companies as their current financial metrics are projected forward. 

#### For short:

- Company A with XYZ financial metrics would have X EV/EBITDA.
- The linear regression studies the financials of each company within the industry.
- The regressed model represent the performance of the industry from the weighted performances of each company.
- Reinserting the XYZ financials of Company A to the regressed line would threfore predict their potential EV/EBITDA in the industry.
- The potential EV/EBITDA of Company A could be higher or lower than the current multiple, therefore, higher potential EV/EBITDA indicates an underpricing of Company A in the industry and vice versa.

## Execution

The project is broken into three distinct stages represented by the respective python program files from Program 1-3. Each program aims to cover different aspects the project beginning from the data retrieval and database storage in Program 1, data request and metric calculations in Program 2, and finally the presentation of data through visualisation in Program 3. 

### Program 1: Data Retrival from Alpha Vantaga API and Storage to Local PostgreSQL Database

To maximise the efficiency of data gathering and reducing the redundancy of 


### Program 2: Database Requests, Metric Calculations, and Appending Data to Database

Data are requested from the database in 


### Program 3: Statistical Significance and Visualisation

The statistical calculations


## Financial Technicals

The data that Both Income Statement and Balance Sheet financials contain typical line items te

#### Financial Metrics: 

- Revenue Growth:
- Opearting Margin:
- Reinvestment Rate:
- Sales to Capital:
- Return on Invested Capital:
- Expected Growth:
- WACC: 

### Limitations

The method outlined to derive EV is not without fault as limitations were present when attempting to perform large scale analysis on the listed companies. One notable compromise in the project are the calculations for WACC. It is a metric used to reflect the risk of a company often derived to represent expected return of the next-best alternative of similar risk. It is calculated with the following formula,

$$
WACC = Cost\ of\ Debt * (1 - Tax\ Rate) * (\frac{Debt}{Debt+Equity}) + Cost\ of\ Equity * (1 - \frac{Debt}{Debt+Equity}) 
$$

where,

$$
Cost\ of\ Debt = Riskfree\ Rate + Credit\ Default\ Risk
$$

and

$$
Cost\ of\ Equity = Riskfree\ Rate + Levered\ Beta * Equity\ Risk\ Premium
$$

The calculation for each component can vary depending on preferences but would otherwise produce similar estimation to its respective value. The limitations of WACC calculations stems from the individual component needed for each calculation. Credit default risk is a simple term to reflect the potential risk of the company defaulting on its interest payments. In an ideal case, financial sources from Bloomberg Terminal or Capital IQ have brief records of credit rating from the likes of S&P Global Ratings, Moody's, and Fitch ratings. They a

## Statistics Technicals

asd
