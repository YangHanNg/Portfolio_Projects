import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.stats import t
from scipy.stats import zscore
import logging
import os
from dotenv import load_dotenv
import time
import psycopg2
from psycopg2.extras import RealDictCursor

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("financial_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "dbname": "postgres",
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

#------------------------------------------------------------------------------------------------------------------------------------------------------

def load_sp500_data(file_path='SP500.csv'):
    """
    Load and validate SP500 CSV data.
    
    Args:
        file_path (str): Path to the SP500 CSV file
    
    Returns:
        pd.DataFrame: Loaded DataFrame or None if error
    """
    try:
        sp500_df = pd.read_csv(file_path)
        return sp500_df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading SP500 data: {str(e)}")
        return None

#------------------------------------------------------------------------------------------------------------------------------------------------------

def display_available_industries(sp500_df):
    """
    Display available industries and their company counts.
    
    Args:
        sp500_df (pd.DataFrame): DataFrame containing SP500 data
    """
    industry_counts = sp500_df['Sector'].value_counts()
    print("Available Industries:")
    for industry, count in industry_counts.items():
        print(f"{industry}: {count} companies")

#------------------------------------------------------------------------------------------------------------------------------------------------------

def select_industry(sp500_df):
    """
    Interactively select an industry from available options.
    
    Args:
        sp500_df (pd.DataFrame): DataFrame containing SP500 data
    
    Returns:
        tuple: (selected_industry, industry_tickers) or (None, None)
    """
    while True:
        # Prompt user to select an industry
        selected_industry = input("\nSelect an industry (or type 'exit' to quit): ").strip()
        
        # Allow user to exit
        if selected_industry.lower() == 'exit':
            print("Exiting the program.")
            return None, None
        
        # Validate industry selection
        if selected_industry in sp500_df['Sector'].unique():
            # Retrieve tickers for the selected industry
            industry_tickers = sp500_df[sp500_df['Sector'] == selected_industry]['Symbol'].tolist()
            
            print(f"\nSelected Industry: {selected_industry}")
            print(f"Number of companies: {len(industry_tickers)}")
            print("Tickers:", ', '.join(industry_tickers))
            
            return selected_industry, industry_tickers
        else:
            print("Invalid industry. Please select from the list above.")
   
#------------------------------------------------------------------------------------------------------------------------------------------------------

def create_db_connection():
    """
    Create and return a psycopg2 database connection.
    
    Returns:
        tuple: Connection and cursor objects
    """
    try:
        # Establish connection
        conn = psycopg2.connect(
            dbname=DB_CONFIG['dbname'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            cursor_factory=RealDictCursor  # Returns results as dictionaries
        )
        
        # Create cursor
        cur = conn.cursor()
        
        return conn, cur
    
    except (psycopg2.Error, Exception) as e:
        logger.error(f"Error creating database connection: {str(e)}")
        raise
        
#------------------------------------------------------------------------------------------------------------------------------------------------------   

def process_financial_data(symbols, selected_industry):
    """
    Process financial data for given company symbols in a specific industry.
    
    Args:
        symbols (list): List of company symbols
        selected_industry (str): Industry name to process metrics for
    
    Returns:
        pandas.DataFrame: DataFrame with processed financial metrics
    """
    # Fetch industry-specific metrics for the selected industry
    industry_metrics = fetch_industry_metrics(selected_industry)
    
    # If no industry metrics found, return None
    if industry_metrics is None:
        logger.warning(f"No industry metrics found for {selected_industry}")
        return None
    
    all_metrics = []
    regression_data = {}
    
    for symbol in symbols:
        logger.info(f"Processing metrics for {symbol}")
        
        # Step 1: Fetch company financial metrics
        df = fetch_financial_metrics(symbol)
        
        # Step 2: Calculate additional metrics
        if df is not None and not df.empty:
            processed_df = calculate_financial_metrics(df, industry_metrics)
            all_metrics.append(processed_df)
            
            
            # Ensure we only take up to the first 3 rows
            for i in range(min(3, len(processed_df))):
                if i not in regression_data:
                    regression_data[i] = {}  # Initialize dictionary for each row index
                
                # Store row data under the symbol key
                regression_data[i][symbol] = processed_df.iloc[i, 
                    [24, 26, 35, 36, 38, 39, 44, 45, 46, 47, 48, 49, 50]].to_dict()

        time.sleep(0.5)  # Prevent rate limiting
    
    # Process regression_data into analyzed_data
    analyzed_data_list = []  # Store analyzed DataFrames
    for row_index, row_data in regression_data.items():
        # Convert row data into a DataFrame
        row_df = pd.DataFrame.from_dict(row_data, orient='index')
        row_df.reset_index(inplace=True)
        row_df.rename(columns={'index': 'Symbol'}, inplace=True)
        row_df['Row_Index'] = row_index  # Track original row index

        # Pass row_df through analyze_financial_data
        analyzed_df = analyze_financial_data(row_df, selected_industry, confidence=0.90)
        
        if analyzed_df is not None:
            analyzed_data_list.append(analyzed_df)

    # Combine all analyzed data into a single DataFrame
    analyzed_data = pd.concat(analyzed_data_list, ignore_index=True) if analyzed_data_list else None
    
    if all_metrics:
        combined_df = pd.concat(all_metrics, ignore_index=True)
        return combined_df
    
    return None

#------------------------------------------------------------------------------------------------------------------------------------------------------

def fetch_industry_metrics(selected_industry):
    """
    Fetch metrics for a specific industry using psycopg2
    
    Args:
        selected_industry (str): Specific industry to fetch metrics for
    
    Returns:
        pandas.DataFrame: DataFrame with industry metrics
    """
    conn, cur = None, None
    try:
        conn, cur = create_db_connection()
        
        # Query to fetch metrics for the specific industry
        query = """
        SELECT 
            i.sector_name AS industry_name, 
            i.reinvestment_rate AS industry_RIR, 
            i.cost_of_capital AS industry_WACC,
            (SELECT unlevered_data FROM companies 
             WHERE sector = i.sector_name 
             LIMIT 1) AS unlevered_data
        FROM 
            industries i
        WHERE 
            i.sector_name = %s
        """
        
        # Execute query with the selected industry
        cur.execute(query, (selected_industry,))
        
        # Fetch all results
        results = cur.fetchall()
        # Convert to dictionary
        if results:
            # Process the first RealDictRow
            first_result = results[0]
            
            # Convert Decimal to float for easier handling
            metric_dict = {
                'industry_name': first_result['industry_name'],
                'industry_rir': float(first_result['industry_rir']),
                'industry_wacc': float(first_result['industry_wacc']),
                'unlevered_data': float(first_result['unlevered_data'])
            }
            
            print(f"Successfully fetched industry metrics for {selected_industry}")
            print(metric_dict)
            return metric_dict
        else:
            print(f"No metrics found for industry: {selected_industry}")
            return None
    
    except (psycopg2.Error, Exception) as e:
        logger.error(f"Error fetching industry metrics for {selected_industry}: {str(e)}")
        return None
    
    finally:
        # Close cursor and connection
        if cur:
            cur.close()
        if conn:
            conn.close()

#------------------------------------------------------------------------------------------------------------------------------------------------------

def fetch_financial_metrics(symbol):
    """
    Fetch financial metrics for a specific company using psycopg2.
    
    Args:
        symbol (str): Company symbol
    
    Returns:
        pandas.DataFrame: DataFrame with financial metrics
    """
    conn, cur = None, None
    try:
        conn, cur = create_db_connection()
        
        # Parameterized query to prevent SQL injection
        query = """
        WITH FinancialMetricsPivot AS (
            SELECT 
                c.symbol,
                rp.fiscal_date_ending,
                MAX(CASE WHEN m.metric_name = 'totalRevenue' THEN fd.value END) AS "totalRevenue",
                MAX(CASE WHEN m.metric_name = 'ebit' THEN fd.value END) AS ebit,
                MAX(CASE WHEN m.metric_name = 'researchAndDevelopment' THEN fd.value END) AS "researchAndDevelopment",
                MAX(CASE WHEN m.metric_name = 'interestExpense' THEN fd.value END) AS "interestExpense",
                MAX(CASE WHEN m.metric_name = 'incomeBeforeTax' THEN fd.value END) AS "incomeBeforeTax",
                MAX(CASE WHEN m.metric_name = 'incomeTaxExpense' THEN fd.value END) AS "incomeTaxExpense",
                MAX(CASE WHEN m.metric_name = 'cashAndShortTermInvestments' THEN fd.value END) AS "cashAndShortTermInvestments",
                MAX(CASE WHEN m.metric_name = 'totalCurrentAssets' THEN fd.value END) AS "totalCurrentAssets",
                MAX(CASE WHEN m.metric_name = 'totalCurrentLiabilities' THEN fd.value END) AS "totalCurrentLiabilities",
                MAX(CASE WHEN m.metric_name = 'totalAssets' THEN fd.value END) AS "totalAssets",
                MAX(CASE WHEN m.metric_name = 'totalLiabilities' THEN fd.value END) AS "totalLiabilities",
                MAX(CASE WHEN m.metric_name = 'depreciationAndAmortization' THEN fd.value END) AS "depreciationAndAmortization",
                MAX(CASE WHEN m.metric_name = 'capitalLeaseObligations' THEN fd.value END) AS "capitalLeaseObligations",
                MAX(CASE WHEN m.metric_name = 'longTermDebt' THEN fd.value END) AS "longTermDebt",
                MAX(CASE WHEN m.metric_name = 'currentLongTermDebt' THEN fd.value END) AS "currentLongTermDebt",
                MAX(CASE WHEN m.metric_name = 'propertyPlantEquipment' THEN fd.value END) AS "propertyPlantEquipment",
                ROW_NUMBER() OVER (PARTITION BY c.symbol ORDER BY rp.fiscal_date_ending DESC) AS row_num
            FROM FinancialData fd
            JOIN FinancialMetrics m ON fd.metric_id = m.metric_id
            JOIN FinancialReports fr ON fd.report_id = fr.report_id
            JOIN ReportingPeriods rp ON fr.period_id = rp.period_id
            JOIN Companies c ON fr.company_id = c.company_id
            WHERE 
                c.symbol = %s AND 
                rp.period_type = 'Annual'
            GROUP BY c.symbol, rp.fiscal_date_ending
        )
        SELECT 
            symbol,
            fiscal_date_ending,
            "totalRevenue",
            ebit,
            "researchAndDevelopment",
            "interestExpense",
            "incomeBeforeTax",
            "incomeTaxExpense",
            "cashAndShortTermInvestments",
            "totalCurrentAssets",
            "totalCurrentLiabilities",
            "totalAssets",
            "totalLiabilities",
            "depreciationAndAmortization",
            "capitalLeaseObligations",
            "longTermDebt",
            "currentLongTermDebt",
            "propertyPlantEquipment"
        FROM FinancialMetricsPivot
        ORDER BY fiscal_date_ending DESC;
        """
        
        # Execute query with parameter
        cur.execute(query, (symbol,))
        
        # Fetch all results
        results = cur.fetchall()
        
        # Convert to DataFrame
        if results:
            df = pd.DataFrame(results)
            return df
        else:
            print(f"No financial metrics found for {symbol}")
            return None
    
    except (psycopg2.Error, Exception) as e:
        logger.error(f"Error fetching financial metrics for {symbol}: {str(e)}")
        return None
    
    finally:
        # Close cursor and connection
        if cur:
            cur.close()
        if conn:
            conn.close()

#------------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_financial_metrics(df, industry_metrics):
    """
    Calculate additional financial metrics with robust NaN handling.
    
    Args:
        df (pandas.DataFrame): DataFrame with financial metrics
        industry_metrics (pandas.DataFrame): DataFrame with industry metrics
    
    Returns:
        pandas.DataFrame: DataFrame with additional calculated metrics
    """
    if df is None or df.empty:
        return None
    
    try:    
        # Get industry metrics
        industry_rir = industry_metrics['industry_rir']
        industry_wacc = industry_metrics['industry_wacc']
        unlevered_data = industry_metrics['unlevered_data']
        
        # Safe calculation functions to handle NaNs
        def safe_division(a, b, fill_value=0):
            return np.where(b != 0, a / b, fill_value)
        
        def safe_subtract(a, b, fill_value=0):
            return np.nan_to_num(a - b, nan=fill_value)
    
        # Total Equity Calculation
        df['Total Equity'] = safe_subtract(
            df['totalAssets'], df['totalLiabilities'], fill_value=0
        )
        # Total Debt Calculation
        df['Total Debt'] = (
            df['capitalLeaseObligations'] + 
            df['longTermDebt'] + 
            df['currentLongTermDebt']
            )
        # Capital Invested Calculation
        df['Capital Invested'] = (
            df['totalAssets'] + 
            df['Total Debt'] + 
            df['cashAndShortTermInvestments']
            )
        # Debt-to-Equity Ratio Calculation
        df['Debt to Equity'] = safe_division(
            df['Total Debt'], 
            (df['Total Debt'] + df['Total Equity']), 
            fill_value=0
        )
        # Working Capital Calculation
        df['Working Capital'] = safe_subtract(
            df['totalCurrentAssets'], df['totalCurrentLiabilities'], fill_value=0
        )
        # Net Working Capital Calculation
        df['Net Working Capital'] = safe_subtract(
            df['Working Capital'], df['Working Capital'].shift(-1), fill_value=0
        )
        # Revenue growth rate 
        df['Revenue Growth'] = safe_division(
            df['totalRevenue'], df['totalRevenue'].shift(-1), fill_value=-1
        ) - 1
        # Operating margin
        df['Operating Margin'] = safe_division(
            df['ebit'], df['totalRevenue'], fill_value=0
        )
        df['EBITDA'] = (
            df['ebit'] + df['depreciationAndAmortization']
            )
        df['EBITDA Margin'] = safe_division(
            df['EBITDA'], df['totalRevenue'], fill_value=0
            )
        # Effective Tax rate
        df['Effective Tax'] = safe_division(
            df['incomeTaxExpense'], df['incomeBeforeTax'], fill_value=0
        )
        df['Effective Tax'] = pd.to_numeric(df['Effective Tax'], errors='coerce')
        # Tax rate 
        df['Tax Rate'] = (
            df['Effective Tax'].shift(-2).rolling(window=3, min_periods=1).mean().fillna(0)
            )
        
        # After tax EBIT 
        df['After Tax EBIT'] = (
             df['ebit'].astype(float)*(1-df['Tax Rate'])
        )
        
        # After tax Operating Margin
        df['After Tax Operating Margin'] = (
            pd.to_numeric(df['Operating Margin'])*(1-pd.to_numeric(df['Tax Rate']))
        )
        
        # Net CapEx 
        df['Net CapEx'] = safe_subtract(
            df['propertyPlantEquipment'], df['propertyPlantEquipment'].shift(-1), fill_value=0
        ) + df['depreciationAndAmortization']
        
        # FCF 
        df['Free Cash Flow'] = (
            df['Net CapEx'] + df['Net Working Capital']
        )
        
        # Reinvestment Rate 
        df['Reinvestment Rate'] = safe_division(
            df['Free Cash Flow'].astype(float), df['After Tax EBIT'], fill_value=0
        )
        
        # Sales to Capital 
        df['Sales to Capital'] = safe_division(
            df['totalRevenue'], df['Capital Invested'], fill_value=0
        )
        # Return On Invested Capital 
        df['Return On Invested Capital'] = (
            df['After Tax Operating Margin'] * 
            df['Sales to Capital'].astype(float)
            )
        # Expected growth rate
        df['Expected Growth'] = (
            df['Reinvestment Rate'] * 
            df['Return On Invested Capital']
            )
        # 3Y Exp Growth
        df['3Y Exp Growth'] = df['Expected Growth'].shift(-2).rolling(window=3).mean().fillna(0)
        # 3Y Rev Growth
        df['3Y Rev Growth'] = df['Revenue Growth'].shift(-2).rolling(window=3).mean().fillna(0)
        # 3Y Reinvestment Rate
        df['3Y RIR'] = df['Reinvestment Rate'].shift(-2).rolling(window=3).mean().fillna(0)
        
        
        # Levered Beta Calculation
        df['Levered Beta'] = (
            unlevered_data *( 1 + pd.to_numeric(df['Debt to Equity']) * (1 - pd.to_numeric(df['Tax Rate'])))
        )
        df['Cost of Debt'] = 0.045 + 0.044
        
        df['Cost of Equity'] = 0.045 + pd.to_numeric(df['Levered Beta'])*0.055
        
        df['WACC'] = (
            pd.to_numeric(df['Cost of Debt'])*(1-pd.to_numeric(df['Tax Rate']))*pd.to_numeric(df['Debt to Equity']) + 
            pd.to_numeric(df['Cost of Equity'])*(1-pd.to_numeric(df['Debt to Equity']))
            )
        
        df['EV'] = (
            (pd.to_numeric(df['ebit'])*(1-pd.to_numeric(df['Tax Rate']))*(1-pd.to_numeric(df['3Y RIR']))*
             (1-(((1+pd.to_numeric(df['3Y Rev Growth']))**3)/((1+pd.to_numeric(df['WACC']))**3))))/
            (pd.to_numeric(df['WACC']) - pd.to_numeric(df['3Y Rev Growth'])) + 
            (pd.to_numeric(df['ebit'])*(1-pd.to_numeric(df['Tax Rate']))*(1+pd.to_numeric(df['3Y Exp Growth']))*
             (1-industry_rir)*(1+pd.to_numeric(df['3Y Rev Growth']))**3)/
            ((industry_wacc-pd.to_numeric(df['3Y Exp Growth']))*((1+pd.to_numeric(df['WACC']))**3))
            )
            
        # Valuation multiples
        df['EV/EBIT'] = safe_division(pd.to_numeric(df['EV']), pd.to_numeric(df['ebit']), fill_value=0)
        df['EV/EBITDA'] = safe_division(pd.to_numeric(df['EV']), pd.to_numeric(df['ebit']) + pd.to_numeric(df['depreciationAndAmortization']), fill_value=0)
        df['EV/After Tax EBIT'] = safe_division(pd.to_numeric(df['EV']), pd.to_numeric(df['After Tax EBIT']), fill_value=0)
        df['EV/Sales'] = safe_division(pd.to_numeric(df['EV']), pd.to_numeric(df['totalRevenue']), fill_value=0)
        df['EV/Capital Invested'] = safe_division(pd.to_numeric(df['EV']), pd.to_numeric(df['Capital Invested']), fill_value=0)
        
        
        # Fill NaNs with 0 for final return
        df.fillna(0, inplace=True)
        return df
    
    except Exception as e:
        logger.error(f"Error calculating financial metrics: {str(e)}")
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()
        return None
    
#------------------------------------------------------------------------------------------------------------------------------------------------------

def analyze_financial_data(data, selected_industry, confidence=0.90):
    try:
        regress_data = {}

        regression_format = {
            'EV/EBIT': ['Operating Margin', 'WACC'],
            'EV/EBITDA': ['Sales to Capital', 'EBITDA Margin'],
            'EV/After Tax EBIT': ['3Y Exp Growth', 'WACC'],
            'EV/Sales': ['Operating Margin', '3Y Rev Growth'],
            'EV/Capital Invested': ['Return on Invested Capital', 'WACC']
        }

        for multiple, (x1_name, x2_name) in regression_format.items():
            y_multiple = data[multiple].values
            x1 = data[x1_name].values
            x2 = data[x2_name].values
            X = np.column_stack((x1, x2))

            # Calculate predictions and confidence intervals
            X1, X2, y_pred, confidence_interval, model = \
                calculate_predictions_and_confidence(X, y_multiple, confidence)

            # Store data in regression dictionary
            regress_data[multiple] = {
                'X': X,
                'y': y_multiple,
                x1_name: X1,  # Store under corresponding name
                x2_name: X2,  # Store under corresponding name
                'y_pred': y_pred,
                'confidence_interval': confidence_interval,
                'model_coefficients': model.coef_,
                'model_intercept': model.intercept_,
            }

        data['Regression'] = regress_data
        return data

    except Exception as e:
        logging.error(f"Error in analyze_financial_data: {str(e)}")
        raise
          
#------------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_predictions_and_confidence(X, y, confidence=0.90):
    # Fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict new values
    predictions = model.predict(X)

    # Create meshgrid for the regression plane
    X1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    X2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1_mesh, x2_mesh = np.meshgrid(X1, X2)

    # Flatten meshgrids to create prediction points
    X_pred = np.column_stack((x1_mesh.ravel(), x2_mesh.ravel()))
    y_pred = model.predict(X_pred)

    # Calculate prediction intervals
    n, p = X.shape
    mse = np.sum((y - model.predict(X))**2) / (n - p - 1)

    X_mean = X.mean(axis=0)
    X_centered = X - X_mean
    cov_matrix = np.linalg.inv(X_centered.T @ X_centered)
    X_pred_centered = X_pred - X_mean
    pred_var = np.diagonal(X_pred_centered @ cov_matrix @ X_pred_centered.T)

    t_value = t.ppf((1 + confidence) / 2, n - p - 1)
    confidence_interval = t_value * np.sqrt(mse * (1 + pred_var))

    return x1_mesh, x2_mesh, y_pred, confidence_interval, model

#------------------------------------------------------------------------------------------------------------------------------------------------------    

def main():
    """Main program execution with industry processing."""
    try:
        # Load SP500 data
        sp500_df = load_sp500_data()
        
        if sp500_df is None:
            return None
        
        # Display available industries
        display_available_industries(sp500_df)
        
        # Select industry
        selected_industry, industry_tickers = select_industry(sp500_df)
        
        # If user exits or selects invalid industry
        if selected_industry is None:
            return None
        
        # Process financial data
        result = process_financial_data(industry_tickers, selected_industry)
        return result
    
    except Exception as e:
        logging.error(f"An error occurred in main: {str(e)}", exc_info=True)
        return None

#------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
