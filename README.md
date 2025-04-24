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

Last Updated: 24/Apr/2025 &nbsp; &nbsp; Status: Ongoing

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

## #2
