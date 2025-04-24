# S&P 500 Valuation Multiples: Linear Regression Data Analysis
### Project Outline

This is a data analysis project inspired by NYU Stern Business School Prof. Aswath Damodaran from his Valuations course. It is a relative valuation method built upon traditional comparable companies analysis. It is based upon Discounted Cash Flow (DCF) calculations to derive Enterprise Value (EV) using metrics that is Weighted Average Cost of Capital (WACC), Reinvestment Rate (RIR), Expected Growth Rate Etc. These financial metrics represent independent variables for dependent variable EV, as such, it allows the opportunity for a linear regression study to interpret the performances of companies relative to their industry peers. The regressed line would serve as a projection tool built on the performance of the industry, and thus, it presents the idea of under/over-pricing of companies as their current financial metrics are projected forward. 

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

The data structure of Income Statement and Balance Sheet financials from AlphaVantag are shown in the following .JSON format.


### Program 2: Database Requests, Metric Calculations, and Appending Data to Database

Data are requested from the database in 


### Program 3: Statistical Significance and Visualisation

The statistical calculations


## Financial Technicals

This section aims provide the method to calculate EV multiples based on the financial metrics of companies. It uses an intrinsic DCF approach to aquire the relationship of certain metrics for each EV multiple. It then presents the limitations of this project from the assumptions and generalisation applied to the model to accomodate the large scale analysis of S&P 500 companies.  

The calculation to EV takes the form of a DCF approach using key financial metrics. The method takes fundamental valuation relationships of Free Cash Flow to Firm (FCFF), WACC, and Growth Rate to approximate EV, shown as the formula below:

$$ EV = \frac{FCFF}{WACC - Growth\ Rate} $$

where,

$$ FCFF = EBIT * (1 - Tax\ Rate) * (1 - RIR) $$

#### Financial Metrics

The following are financial metrics that are used to expand on the calculation for each EV multiple:

| Metrics      | Notation      | Formula |
| ------------- | ------------- | ----- |
|Revenue Growth Rate| **Rev. Growth**|1 - Current Revenue/Previous Revenue |
|After-Tax Operating Margin |**ATOM**| EBIT*(1 - Tax Rate)|
|Reinvestment Rate| **RIR**| Free Cash Flow/ ATOM|
|Sales to Capital| **SC**| Sales/ Capital Invested|
|Return on Invested Capital| **ROIC**| ATOM * Sales to Capital|
|Percentage D&A| **DA%**| DA/EBITDA|
|Expected Growth Rate| **Exp. Growth**| RIR * ROIC|
|Industry WACC |**IWACC**| Industry Average WACC|
|Industry Reinvestment Rate| **IRIR**| Industry Average RIR|

Below shows the example to derive EV/EBIT multiple using the DCF appraoch:

$$ \frac{EV}{EBIT} = \frac{1}{EBIT} * \frac{EBIT * (1 - Tax\ Rate) * (1 - RIR)}{WACC - Growth Rate} = \frac{(1 - Tax\ Rate) * (1 - RIR)}{WACC - Growth\ Rate} $$

It shows that EV/EBIT multiple is a function of Tax Rate, WACC, and Growth Rate. Similar conclusion can be drawn to find components to the function when applying the same theory to other EV multiples. 

Expanding on the calculation, a two-stage DCF logic are implemented to the formula to project potential growth and deriving a potential terminal value. The two-stage DCF discounting term are used on the assumption that S&P 500 companies would experience some level of growth before reaching terminal value. The term can be represent as the following,

$$ \frac{EV}{Metric} = Scaling\ Factor \times \left[\frac{(1 - RIR) * (1 + Rev. Growth) * \left(1 - \frac{(1 + Rev. Growth)^t}{(1 + WACC)^t}\right)}{WACC - Rev. Growth} + \frac{(1 - IRIR) * (1 + Rev. Growth)^t * (1 + Exp. Growth)}{(IWACC - Exp. Growth) * (1 + WACC)^t}\right] $$

where the Scaling Factor adjusts according to the EV multiple and t represents the number of years projected. The table below shows the assocaited scaling factor to the corresponding EV multiple.

| EV Metric      | Scaling Factor      | Proof |
| ------------- | ------------- | ----- |
| EV/EBIT | 1/1-tax |  |
| EV/EBIT(1-Tax) | 1 | EBIT(1-tax)/EBIT(1-tax) |
| EV/EBITDA | (1 - DA%)(1 - tax) | EBIT(1-tax) = (EBITDA - DA)(1-tax) |
| EV/Sales |  ATOM  | EBIT(1-tax)/Sales = ATOM |
| EV/Capital Invested |  ROIC  | EBIT(1-tax) = ROIC * IC |


### Limitations

There are a few limitations to this approach of analysing trading mutliples on S&P 500 listed companies. From an initial review over the approach, the limitations that can be drawn are:

- **Assumptions of the model**
- **WACC calculations**
- **Industry Variables**

#### Assumptions of the model

The two-stage DCF assumes growth in companies, however, the S&P 500 consists of various companies in different stages of their business life cycle. As such, the projection period for companies should ideally be projected based on their historical progression and likelihood for growth. Applying a unique time projection for each 500 listed companies would require lots of analysis into company history and not within the scope of this project. A short time projection of 3 years (t=3) is selected to balance between young and old companies within the index and to focus more on the near-term potential of companies.

To overcome biases in estimating the components used for EV calculations, the growth rate are approximated as 3-year moving average of historical revenue growth rate (Rev. Growth). It is not the best estimation for growth since it is backward looking, however, it gives a fair comparison because it not assigned from an external estimation (i.e. based solely on company performance). Of course extreme values would distort this moving average value when reflecting on the financial performances during the pandemic. Other components that take in the form of 3-year moving average also includes  reinvestment rate (RIR) and expected growth rate (Exp. Growth). It is on the assumption that reinvestments typically does not occur immediately and would require time to experience the growth achieved from reinvestments. The terminal components uses industry average RIR and WACC on the basis that companies would have equivalent cost of capital structure and RIR in a stable market, but an exeption are made to the terminal growth as the 3-year moving Exp. Growth are used to represent the advantagous competitive edge from their past reinvestments.

#### WACC Calculations
One notable compromise in the project are the calculations for WACC. It is a metric used to reflect the risk of a company often derived to represent expected return of the next-best alternative of similar risk. It is calculated with the following formula,

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

The calculation for each component can vary depending on preferences but would otherwise produce similar estimation to its respective value. The limitations of WACC calculations stems from the individual component needed for each calculation. Credit default risk is a simple term to reflect the potential risk of the company defaulting on its interest payments. In an ideal case, financial sources like Bloomberg Terminal or Capital IQ have brief records of credit rating from the likes of S&P Global Ratings, Moody's, and Fitch ratings. The average rating across three institutions would likely provide a good estimation as to the safety of principal (the initial sum borrowed to the company) and ability to pay scheduled interest payment. The same issues are present when estimating the Equity Risk Premiums (ERP) of companies because they have all established a monopolsitic barrier in their specific market each with unique operating advantages that allows them to be competitive. Ratings are typically unique to companies and are based the capital structure (their financial leverage), likewise, ERP aims to represent the risk of markets that firms operate in. 

To acquire each specific credit rating as well as the ERP for the 500 companies would prove to be time consuming as not all institutions provide rating to the firm and the risk of markets are different from one to another. As a result, the compromise to WACC calculations are the generalisation of the entire calculation through the use of average defualt risk and market risk premiums of companies in the US (assuming most companies operate in the US).

#### Industry Variables

Both industry averages for RIR and WACC are taken from Prof. Aswath Damodaran's website because the WACC calculations used for the large scale analysis are inaccurate as discussed previously. The RIR should in theory be used as the average of RIR accross industry, however to implement this feature would increase the operation performed as metrics need to be calculated, stored, and further calculations before the actual analysis. The unlevered beta used for WACC are also source from Prof. Aswath Damodaran on the same basis that levered beta are company specific that needs the unlevered and re-levered for WACC calculations.

## Statistics Technicals

asd
