import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import t
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import variance_inflation_factor
import statsmodels.api as sm
import logging
import os
from dotenv import load_dotenv
import time
import psycopg2
from psycopg2.extras import RealDictCursor

# Load environment variables
load_dotenv()

# Logging setup
# Create logs directory if it doesn't exist
log_dir = 'Projects/Data Analysis/S&P500 Valuation Multiples'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "financial_data.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

#------------------------------------------------------------------------------------------------------------------------------------------------------

def load_sp500_data(file_path='data/SP500.csv'):
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
    Display available industries with more than 8 companies.
    
    Args:
        sp500_df (pd.DataFrame): DataFrame containing SP500 data
    """
    industry_counts = sp500_df['Sector'].value_counts()
    print("Available Industries:")
    for industry, count in industry_counts.items():
        if count >= 8:
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
        if not selected_industry:
            logger.warning("Input cannot be empty. Please enter a valid industry.")
            continue
        
        # Allow user to exit
        if selected_industry.lower() == 'exit':
            logger.info("Exiting the program.")
        if 'Sector' in sp500_df.columns and not sp500_df['Sector'].empty and selected_industry in sp500_df['Sector'].unique():
        
            # Retrieve tickers for the selected industry
            filtered_df = sp500_df.loc[sp500_df['Sector'] == selected_industry]
            industry_tickers = filtered_df['Symbol'].tolist()
        
            print(f"\nSelected Industry: {selected_industry}")
            print(f"Number of companies: {len(industry_tickers)}")
            print("Tickers:", ', '.join(industry_tickers))
        
        return selected_industry, industry_tickers
 
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
    
    # Create column mapping in case of different column names
    column_mapping = {
        'Operating Margin': 'Operating Margin',
        'WACC': 'WACC',
        'Sales to Capital': 'Sales to Capital',
        'EBITDA Margin': 'EBITDA Margin',
        '3Y Exp Growth': '3Y Exp Growth',  # Possible different name
        '3Y Rev Growth': '3Y Rev Growth',   # Possible different name
        'Return On Invested Capital': 'Return On Invested Capital',  # Note the 'On' vs 'on'
        'EV/EBIT': 'EV/EBIT',
        'EV/EBITDA': 'EV/EBITDA',
        'EV/After Tax EBIT': 'EV/After Tax EBIT',
        'EV/Sales': 'EV/Sales',
        'EV/Capital Invested': 'EV/Capital Invested'
        }
    
    for symbol in symbols:
        logger.info(f"Processing metrics for {symbol}")
        
        # Step 1: Fetch company financial metrics
        df = fetch_financial_metrics(symbol)
        
        # Step 2: Calculate additional metrics
        if df is not None and not df.empty:
            processed_df = calculate_financial_metrics(df, industry_metrics)
            all_metrics.append(processed_df)
            
            # Initialize regression_data for each row if not exists
            for i in range(min(3, len(processed_df))):
                if i not in regression_data:
                    regression_data[i] = {}
                
                # Store data for current symbol
                regression_data[i][symbol] = {}
                for required_col, actual_col in column_mapping.items():
                    if actual_col not in processed_df.columns:
                        logger.warning(f"Missing column in processed_df: {actual_col} (mapped from {required_col})")
                        regression_data[i][symbol][required_col] = None
                    else:
                        # Store the value from processed_df
                        regression_data[i][symbol][required_col] = processed_df.iloc[i][actual_col]
            
            logger.info(f"Stored data for {symbol} in regression_data")
            
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
        industry_metrics (dict): Dictionary with industry metrics
    
    Returns:
        pandas.DataFrame: DataFrame with additional calculated metrics
    """
    if df is None or df.empty:
        return None

    try:
        # Convert all relevant columns to numeric upfront
        numeric_columns = [
            'totalAssets', 'totalLiabilities', 'capitalLeaseObligations', 'longTermDebt',
            'currentLongTermDebt', 'cashAndShortTermInvestments', 'totalCurrentAssets',
            'totalCurrentLiabilities', 'totalRevenue', 'ebit', 'depreciationAndAmortization',
            'incomeTaxExpense', 'incomeBeforeTax', 'propertyPlantEquipment'
        ]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Extract industry metrics
        industry_rir = industry_metrics['industry_rir']
        industry_wacc = industry_metrics['industry_wacc']
        unlevered_data = industry_metrics['unlevered_data']

        # Helper functions for safe calculations
        safe_division = lambda a, b: np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b != 0)
        safe_subtract = lambda a, b: a - b if (isinstance(a, (int, float)) and isinstance(b, (int, float))) else np.subtract(a, b)

        # Perform calculations
        df['Total Equity'] = safe_subtract(df['totalAssets'], df['totalLiabilities'])
        df['Total Debt'] = df['capitalLeaseObligations'] + df['longTermDebt'] + df['currentLongTermDebt']
        df['Capital Invested'] = df['totalAssets'] + df['Total Debt'] + df['cashAndShortTermInvestments']
        df['Debt to Equity'] = safe_division(df['Total Debt'], df['Total Debt'] + df['Total Equity'])
        df['Working Capital'] = safe_subtract(df['totalCurrentAssets'], df['totalCurrentLiabilities'])
        df['Net Working Capital'] = safe_subtract(df['Working Capital'], df['Working Capital'].shift(-1))
        df['Revenue Growth'] = safe_division(df['totalRevenue'], df['totalRevenue'].shift(-1)) - 1
        df['Operating Margin'] = safe_division(df['ebit'], df['totalRevenue'])
        df['EBITDA'] = df['ebit'] + df['depreciationAndAmortization']
        df['EBITDA Margin'] = safe_division(df['EBITDA'], df['totalRevenue'])
        df['DA'] = safe_division(df['depreciationAndAmortization'], df['EBITDA'])
        df['Effective Tax'] = safe_division(df['incomeTaxExpense'], df['incomeBeforeTax'])
        df['Tax Rate'] = df['Effective Tax'].rolling(window=3, min_periods=1).mean()
        df['After Tax EBIT'] = df['ebit'] * (1 - df['Tax Rate'])
        df['After Tax Operating Margin'] = df['Operating Margin'] * (1 - df['Tax Rate'])
        df['Net CapEx'] = safe_subtract(df['propertyPlantEquipment'], df['propertyPlantEquipment'].shift(-1)) + df['depreciationAndAmortization']
        df['Free Cash Flow'] = df['Net CapEx'] + df['Net Working Capital']
        df['Reinvestment Rate'] = safe_division(df['Free Cash Flow'], df['After Tax EBIT'])
        df['Sales to Capital'] = safe_division(df['totalRevenue'], df['Capital Invested'])
        df['Return On Invested Capital'] = df['After Tax Operating Margin'] * df['Sales to Capital']
        df['Expected Growth'] = df['Reinvestment Rate'] * df['Return On Invested Capital']
        df['3Y Exp Growth'] = df['Expected Growth'].rolling(window=3, min_periods=1).mean()
        df['3Y Rev Growth'] = df['Revenue Growth'].rolling(window=3, min_periods=1).mean()
        df['3Y RIR'] = df['Reinvestment Rate'].rolling(window=3, min_periods=1).mean()
        df['Levered Beta'] = unlevered_data * (1 + df['Debt to Equity'] * (1 - df['Tax Rate']))
        df['Cost of Debt'] = 0.045 + 0.044
        df['Cost of Equity'] = 0.045 + df['Levered Beta'] * 0.055
        df['WACC'] = (
            df['Cost of Debt'] * (1 - df['Tax Rate']) * df['Debt to Equity'] +
            df['Cost of Equity'] * (1 - df['Debt to Equity'])
        )
        
        df['DCF'] = (
            ((1 - df['3Y RIR']) * (1- df['3Y Rev Growth']) * (1 - ((1 + df['3Y Rev Growth'])**3 / (1 + df['WACC'])**3))) /
            (df['WACC'] - df['3Y Rev Growth']) +
            ((1 - industry_rir) * (1 + df['3Y Rev Growth'])**3 * (1 + df['3Y Exp Growth'])) /
            ((industry_wacc - df['3Y Exp Growth']) * (1 + df['WACC'])**3)
        )
        df['EV/EBIT'] = (1 - df['Tax Rate']) * df['DCF']
        df['EV/EBITDA'] = (1 - df['Tax Rate']) * (1 - df['DA']) * df['DCF']
        df['EV/After Tax EBIT'] = 1 * df['DCF']
        df['EV/Sales'] = df['After Tax Operating Margin'] * df['DCF']
        df['EV/Capital Invested'] = df['Return On Invested Capital'] * df['DCF'] 
        # Fill NaNs with 0 for final return
        df.fillna(0, inplace=True)
        return df

    except Exception as e:
        logger.error(f"Error calculating financial metrics: {str(e)}", exc_info=True)
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
            'EV/Capital Invested': ['Return On Invested Capital', 'WACC']
        }
        index = data['Row_Index'].iloc[0]

        for multiple, (x1_name, x2_name) in regression_format.items():
            # Convert to pandas series first to handle any potential missing values
            y_multiple = pd.to_numeric(data[multiple], errors='coerce').values
            x1 = pd.to_numeric(data[x1_name], errors='coerce').values
            x2 = pd.to_numeric(data[x2_name], errors='coerce').values
            
            # Create the design matrix X
            X = np.column_stack((x1, x2))

            # Calculate predictions and confidence intervals
            confidence_interval, model = \
                calculate_predictions_and_confidence(X, y_multiple,multiple, x1_name, x2_name, index, confidence, selected_industry=selected_industry)

            # Store data in regression dictionary
            regress_data[multiple] = {
                'confidence_interval': confidence_interval,
                'model_coefficients': model.params[1:],
                'model_intercept': model.params[0],
            }   

        data['Regression'] = regress_data
        
        return data

    except Exception as e:
        logging.error(f"Error in analyze_financial_data: {str(e)}")
        raise
          
#------------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_predictions_and_confidence(X, y, multiple, x1_name, x2_name, index, confidence=0.90, selected_industry=None):

    # Add a constant to the predictors for statsmodels
    X_with_const = sm.add_constant(X)

    # Fit the model using statsmodels
    model = sm.OLS(y, X_with_const).fit()

    # Convert numpy array to DataFrame for VIF calculation
    # Calculate VIF
    X_df = pd.DataFrame(X, columns=[x1_name, x2_name])
    vif_data = pd.DataFrame()
    vif_data["Metric"] = X_df.columns
    vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]
    
    # Calculate BP test
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    bp_labels = ['LM_statistic', 'p_value', 'f_value', 'f_p_value']
    bp_data = pd.DataFrame([bp_test], columns=bp_labels)
    
    # Create combined diagnostics DataFrame
    diagnostics = pd.concat([
        vif_data,
        pd.DataFrame({'feature': bp_labels, 'value': bp_test})
    ], ignore_index=True)
    
    print("\nRegression Diagnostics:")
    print(diagnostics)

    plt.rcParams.update({
    'font.family': 'Courier New',  # monospace font
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20
    }) 

    # Create meshgrid for the regression plane
    num_points = int(100)
    X1 = np.linspace(X[:, 0].min(), X[:, 0].max(), num_points)
    X2 = np.linspace(X[:, 1].min(), X[:, 1].max(), num_points)
    x1_mesh, x2_mesh = np.meshgrid(X1, X2)

    # Create prediction points from meshgrid
    X_pred = np.column_stack((x1_mesh.ravel(), x2_mesh.ravel()))
    X_pred_with_const = sm.add_constant(X_pred)

    # Calculate total points in the meshgrid
    total_points = num_points * num_points

    # Get predictions
    y_pred = model.predict(X_pred_with_const)

    # Calculate prediction intervals (make sure we get exactly the right number)
    pred = model.get_prediction(X_pred_with_const)
    pred_var = pred.var_pred_mean
    t_value = t.ppf((1 + confidence) / 2, df=model.df_resid)

    # Calculate 90% confidence interval directly from standard error
    confidence_interval = t_value * np.sqrt(pred_var)  # 1.645 is the z-score for 90% confidence

    # Ensure we have exactly the right number of points to reshape
    if len(confidence_interval) != total_points:
        confidence_interval = confidence_interval[:total_points]

    # Reshape to match the grid
    z_mesh = y_pred.reshape((num_points, num_points))
    ci_mesh = confidence_interval.reshape((num_points, num_points))

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot actual data points
    ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Actual Data')

    # Plot regression surface
    surf = ax.plot_surface(x1_mesh, x2_mesh, z_mesh, 
                           alpha=0.3, cmap='viridis')

    # Plot upper and lower confidence bounds
    ax.plot_surface(x1_mesh, x2_mesh, z_mesh + ci_mesh, 
                    alpha=0.1, color='red')
    ax.plot_surface(x1_mesh, x2_mesh, z_mesh - ci_mesh, 
                    alpha=0.1, color='red')
    

    # Labels and title
    ax.set_xlabel(f'{x1_name}')
    ax.set_ylabel(f'{x2_name}')
    ax.set_zlabel(f'{multiple}')
    ax.set_title(
        f"Linear Regression Model of {selected_industry} Industry with {int(confidence * 100)}% Confidence Interval\n"
        f"Multiple: {multiple}, Year: {index}"
    )

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.show()
    print(model.summary())

    return confidence_interval, model

#------------------------------------------------------------------------------------------------------------------------------------------------------    

def main():
    """Main program execution with industry processing."""
    try:
        # Set working directory
        os.chdir('Projects/Data Analysis/S&P500 Valuation Multiples')

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
