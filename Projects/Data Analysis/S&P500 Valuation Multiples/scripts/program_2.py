import pandas as pd
import numpy as np
import psycopg2
import logging
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sys

# Add parent directory to Python path to allow imports from scripts folder
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from program_1 import (
    ProcessingCache, 
    get_db_connection, 
    execute_query,
    FinancialDataAccess
)

# Load environment variables
load_dotenv()

# Set up logging
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

#------------------------------------------------------------------------------------------------------------------------------------------------------

def load_sp500_data(file_path='data/SP500.csv'):
    """Load S&P 500 data from CSV file."""
    try:
        logger.info(f"Reading SP500 data from {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to load S&P 500 data: {str(e)}")
        return None

#------------------------------------------------------------------------------------------------------------------------------------------------------

def fetch_industry_metrics(selected_industry):
    """
    Fetch metrics for a specific industry.
    
    Args:
        selected_industry (str): Specific industry to fetch metrics for
    
    Returns:
        dict: Dictionary with industry metrics
    """
    conn, cur = None, None
    try:
        conn, cur = get_db_connection()
        
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
        
        cur.execute(query, (selected_industry,))
        results = cur.fetchall()
        
        if results:
            # Convert to dictionary
            first_result = results[0]
            metric_dict = {
                'industry_name': first_result['industry_name'],
                'industry_rir': float(first_result['industry_rir']),
                'industry_wacc': float(first_result['industry_wacc']),
                'unlevered_data': float(first_result['unlevered_data'])
            }
            logger.info(f"Successfully fetched industry metrics for {selected_industry}")
            return metric_dict
            
        logger.warning(f"No metrics found for industry: {selected_industry}")
        return None
    
    except Exception as e:
        logger.error(f"Error fetching industry metrics: {str(e)}")
        return None
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

#------------------------------------------------------------------------------------------------------------------------------------------------------

def fetch_financial_metrics(symbol):
    """
    Fetch financial metrics for a company.
    
    Args:
        symbol (str): Company symbol
        
    Returns:
        DataFrame: Financial metrics for the company
    """
    try:
        conn, cur = get_db_connection()
        
        query = """
        WITH LatestReports AS (
            SELECT DISTINCT ON (fr.company_id, rp.fiscal_date_ending) 
                fr.report_id,
                c.symbol,
                rp.fiscal_date_ending,
                fr.report_type
            FROM 
                FinancialReports fr
                JOIN Companies c ON fr.company_id = c.company_id
                JOIN ReportingPeriods rp ON fr.period_id = rp.period_id
            WHERE 
                c.symbol = %s
                AND rp.period_type = 'Annual'
            ORDER BY 
                fr.company_id, 
                rp.fiscal_date_ending DESC,
                fr.report_type
        )
        SELECT 
            lr.symbol,
            lr.fiscal_date_ending,
            m.metric_name,
            fd.value
        FROM 
            LatestReports lr
            JOIN FinancialData fd ON lr.report_id = fd.report_id
            JOIN FinancialMetrics m ON fd.metric_id = m.metric_id
        ORDER BY 
            lr.fiscal_date_ending DESC;
        """
        
        cur.execute(query, (symbol,))
        results = cur.fetchall()
        
        if not results:
            logger.warning(f"No financial data found for {symbol}")
            return None
        
        # Convert to DataFrame and pivot
        df = pd.DataFrame(results)
        df_pivot = df.pivot(
            index=['symbol', 'fiscal_date_ending'],
            columns='metric_name',
            values='value'
        ).reset_index()
        
        return df_pivot
        
    except Exception as e:
        logger.error(f"Error fetching financial metrics for {symbol}: {str(e)}")
        return None
    finally:
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
        # Get industry metrics
        industry_rir = industry_metrics['industry_rir']
        industry_wacc = industry_metrics['industry_wacc']
        unlevered_data = industry_metrics['unlevered_data']

        # Helper functions for safe calculations
        safe_division = lambda a, b: np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b != 0)
        safe_subtract = lambda a, b: a - b if (isinstance(a, (int, float)) and isinstance(b, (int, float))) else np.subtract(a, b)

        # Convert all relevant columns to numeric
        numeric_columns = [
            'totalAssets', 'totalLiabilities', 'capitalLeaseObligations', 'longTermDebt',
            'currentLongTermDebt', 'cashAndShortTermInvestments', 'totalCurrentAssets',
            'totalCurrentLiabilities', 'totalRevenue', 'ebit', 'depreciationAndAmortization',
            'incomeTaxExpense', 'incomeBeforeTax', 'propertyPlantEquipment'
        ]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Calculate metrics
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
    """
    Perform regression analysis on financial data.
    
    Args:
        data (DataFrame): Financial data
        selected_industry (str): Industry name
        confidence (float): Confidence level for analysis
        
    Returns:
        dict: Regression analysis results
    """
    try:
        regress_data = {}

        regression_format = {
            'EV/EBIT': ['Return On Invested Capital', '3Y Rev Growth'],
            'EV/After Tax EBIT': ['Return On Invested Capital', 'After Tax Operating Margin'],
            'EV/EBITDA': ['Return On Invested Capital', 'DA'],
            'EV/Sales': ['After Tax Operating Margin', '3Y Rev Growth'],
            'EV/Capital Invested': ['Return On Invested Capital', '3Y Exp Growth']
        }
        index = data['Row_Index'].iloc[0]

        for multiple, (x1_name, x2_name) in regression_format.items():
            # Convert to pandas series first to handle any potential missing values
            numeric_data = data[[multiple, x1_name, x2_name]].apply(pd.to_numeric, errors='coerce')
            y_multiple = numeric_data[multiple].values
            x1 = numeric_data[x1_name].values
            x2 = numeric_data[x2_name].values
            
            # Create the design matrix X 
            X = np.column_stack((x1, x2))

            # Calculate predictions and confidence intervals
            confidence_interval, model, diagnostics = \
                calculate_predictions_and_confidence(X, y_multiple, multiple, x1_name, x2_name, index, confidence, selected_industry)

            # Store data in regression dictionary
            regress_data[multiple] = {
                'Summary':{
                'coefficients': model.params,
                'standard_errors': model.bse,
                'p_values': model.pvalues,
                't_values': model.tvalues,
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_pvalue': model.f_pvalue,
                'degrees_of_freedom_model': model.df_model,
                'degrees_of_freedom_resid': model.df_resid
                },
                'Diagnostics': {
                    metric: [val, passed] for metric, val, passed in zip(
                        diagnostics['Metric'], 
                        diagnostics['Value'],
                        diagnostics['Pass']
                    )
                }
            }

        data['Regression'] = regress_data
        return data

    except Exception as e:
        logging.error(f"Error in analyze_financial_data: {str(e)}")
        raise

#------------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_predictions_and_confidence(X, y, multiple, x1_name, x2_name, index, confidence=0.90, selected_industry=None):
    """
    Calculate regression predictions and confidence intervals.
    
    Args:
        X (ndarray): Design matrix
        y (ndarray): Target variable
        multiple (str): Multiple type
        x1_name (str): First predictor name
        x2_name (str): Second predictor name
        index (int): Row index
        confidence (float): Confidence level
        selected_industry (str): Industry name
        
    Returns:
        tuple: (confidence_interval, model, diagnostics)
    """
    try:
        # Fit the model
        model = OLS(y, X).fit()
        
        # Calculate predictions
        y_pred = model.predict(X)
        
        # Calculate residuals and standardized residuals
        residuals = y - y_pred
        standardized_residuals = residuals / np.std(residuals)
        
        # Perform Breusch-Pagan test for heteroscedasticity
        bp_test = het_breuschpagan(residuals, X)
        
        # Calculate VIF for multicollinearity
        X_with_const = np.column_stack([np.ones(len(X)), X])
        vif = [variance_inflation_factor(X_with_const, i) for i in range(X_with_const.shape[1])]
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        ci = model.conf_int(alpha)
        
        # Store diagnostic results
        diagnostics = {
            'Metric': ['Heteroscedasticity', 'Multicollinearity', 'Normality'],
            'Value': [bp_test[1], max(vif[1:]), np.mean(np.abs(standardized_residuals))],
            'Pass': [
                bp_test[1] > 0.05,  # Heteroscedasticity test
                all(v < 5 for v in vif[1:]),  # VIF test
                np.mean(np.abs(standardized_residuals)) < 2  # Normality test
            ]
        }
        
        return ci, model, diagnostics
        
    except Exception as e:
        logger.error(f"Error in calculate_predictions_and_confidence: {str(e)}")
        raise

#------------------------------------------------------------------------------------------------------------------------------------------------------    

def save_calculated_metrics(df, company_id, period_id):
    """Save calculated metrics to database."""
    try:
        conn, cur = get_db_connection()
        for col in df.columns:
            # Skip non-numeric columns
            if col in ['symbol', 'fiscal_date_ending']:
                continue

            # Insert metric definition first
            formula = get_metric_formula(col)
            if formula:
                cur.execute("""
                    INSERT INTO CalculatedMetricDefinitions (metric_name, formula, description)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (metric_name) DO NOTHING
                """, (col, formula, f"Formula for {col}"))

            # Insert calculated metric values
            cur.execute("""
                INSERT INTO CalculatedMetrics 
                (company_id, period_id, metric_name, metric_value, data_source)
                VALUES (%s, %s, %s, %s, 'calculated')
                ON CONFLICT (company_id, period_id, metric_name)
                DO UPDATE SET 
                    metric_value = EXCLUDED.metric_value,
                    updated_at = CURRENT_TIMESTAMP
            """, (company_id, period_id, col, float(df[col].iloc[0])))
        
        conn.commit()
        logger.info(f"Saved calculated metrics for company_id {company_id}, period_id {period_id}")
        
    except Exception as e:
        logger.error(f"Error saving calculated metrics: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def get_metric_formula(metric_name):
    """Return formula for calculated metrics."""
    formulas = {
        'Total Equity': 'totalAssets - totalLiabilities',
        'Total Debt': 'capitalLeaseObligations + longTermDebt + currentLongTermDebt',
        'Capital Invested': 'totalAssets + Total Debt + cashAndShortTermInvestments',
        'Debt to Equity': 'Total Debt / (Total Debt + Total Equity)',
        'Working Capital': 'totalCurrentAssets - totalCurrentLiabilities',
        'Net Working Capital': 'Working Capital[t] - Working Capital[t-1]',
        'Revenue Growth': '(totalRevenue[t] / totalRevenue[t-1]) - 1',
        'Operating Margin': 'ebit / totalRevenue',
        'EBITDA': 'ebit + depreciationAndAmortization',
        'EBITDA Margin': 'EBITDA / totalRevenue',
        'DA': 'depreciationAndAmortization / EBITDA',
        'Effective Tax': 'incomeTaxExpense / incomeBeforeTax',
        'Tax Rate': '3-year rolling average of Effective Tax',
        'After Tax EBIT': 'ebit * (1 - Tax Rate)',
        'After Tax Operating Margin': 'Operating Margin * (1 - Tax Rate)',
        'Net CapEx': '(propertyPlantEquipment[t] - propertyPlantEquipment[t-1]) + depreciationAndAmortization',
        'Free Cash Flow': 'Net CapEx + Net Working Capital',
        'Reinvestment Rate': 'Free Cash Flow / After Tax EBIT',
        'Sales to Capital': 'totalRevenue / Capital Invested',
        'Return On Invested Capital': 'After Tax Operating Margin * Sales to Capital',
        'Expected Growth': 'Reinvestment Rate * Return On Invested Capital'
    }
    return formulas.get(metric_name)

def save_regression_results(company_id, period_id, multiple, regress_data, fig):
    """Save regression analysis results to database."""
    try:
        conn, cur = get_db_connection()
        
        # Convert figure to binary for storage
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plot_data = buf.getvalue()
        buf.close()
        
        # Save regression results
        cur.execute("""
            INSERT INTO RegressionAnalysis (
                company_id, period_id, multiple_type,
                variable1_name, variable2_name,
                coefficients, standard_errors,
                p_values, t_values,
                r_squared, adj_r_squared,
                f_statistic, f_pvalue,
                diagnostics, plot_data
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (company_id, period_id, multiple_type)
            DO UPDATE SET
                coefficients = EXCLUDED.coefficients,
                standard_errors = EXCLUDED.standard_errors,
                p_values = EXCLUDED.p_values,
                t_values = EXCLUDED.t_values,
                r_squared = EXCLUDED.r_squared,
                adj_r_squared = EXCLUDED.adj_r_squared,
                f_statistic = EXCLUDED.f_statistic,
                f_pvalue = EXCLUDED.f_pvalue,
                diagnostics = EXCLUDED.diagnostics,
                plot_data = EXCLUDED.plot_data,
                updated_at = CURRENT_TIMESTAMP
        """, (
            company_id, period_id, multiple,
            regress_data['variable1_name'], regress_data['variable2_name'],
            regress_data['Summary']['coefficients'].tolist(),
            regress_data['Summary']['standard_errors'].tolist(),
            regress_data['Summary']['p_values'].tolist(),
            regress_data['Summary']['t_values'].tolist(),
            float(regress_data['Summary']['r_squared']),
            float(regress_data['Summary']['adj_r_squared']),
            float(regress_data['Summary']['f_statistic']),
            float(regress_data['Summary']['f_pvalue']),
            regress_data['Diagnostics'],
            plot_data
        ))
        
        conn.commit()
        
    except Exception as e:
        logger.error(f"Error saving regression results: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

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

        # Get industry metrics and perform calculations
        industry_metrics = {}
        for industry in sp500_df['Sector'].unique():
            metrics = fetch_industry_metrics(industry)
            if metrics:
                industry_metrics[industry] = metrics
        
        # Process companies by industry
        results = {}
        for industry, metrics in industry_metrics.items():
            logger.info(f"\nProcessing {industry}")
            
            # Get companies in this industry
            industry_companies = sp500_df[sp500_df['Sector'] == industry]['Symbol'].tolist()
            
            industry_data = []
            for symbol in industry_companies:
                # Fetch and calculate metrics
                financial_data = fetch_financial_metrics(symbol)
                if financial_data is not None:
                    calculated_data = calculate_financial_metrics(financial_data, metrics)
                    if calculated_data is not None:
                        industry_data.append(calculated_data)
            
            if industry_data:
                results[industry] = pd.concat(industry_data, ignore_index=True)
                
                # Perform regression analysis if enough companies
                if len(industry_companies) >= 8:
                    results[industry] = analyze_financial_data(results[industry], industry)
        
        return results
    
    except Exception as e:
        logging.error(f"An error occurred in main: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    main()
