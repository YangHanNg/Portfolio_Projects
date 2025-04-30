# Import necessary libraries
from __future__ import annotations  # For Python 3.7+ type hints
import pandas as pd
import numpy as np
import psycopg2
import logging
import os
import time
import json
import io
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.stats import t
import sys

# Add parent directory to Python path to allow imports from scripts folder
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import modules from program_1.py script
from program_1 import (
    ProcessingCache, 
    get_db_connection, 
    execute_query,
    FinancialDataAccess
)

# Load environment variables from .env file
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

# Define plot settings
plt.rcParams.update({
    'font.family': 'Courier New',
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Define color palette for plots
PLOT_COLORS = {
    'scatter': '#9671bd',        # soft purple
    'scatter_edge': '#6a408d',   # deeper purple
    'surface_cmap': 'plasma',    # colorful surface
    'confidence': 'red',         # confidence interval color
    'grid': 'gray'              # grid color
}

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to load S&P 500 data from CSV file
def load_sp500_data(file_path='data/SP500.csv'):

    try:
        logger.info(f"Reading SP500 data from {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to load S&P 500 data: {str(e)}")
        return None

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to retrieve industry data for a specific industry in the database
def fetch_industry_metrics(selected_industry):
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Query to fetch metrics for the specific industry
                query = """
                SELECT 
                    i.sector_name AS industry_name, 
                    i.reinvestment_rate AS industry_rir, 
                    i.cost_of_capital AS industry_wacc,
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
                        'industry_name': first_result[0],
                        'industry_rir': float(first_result[1]) if first_result[1] is not None else 0.0,
                        'industry_wacc': float(first_result[2]) if first_result[2] is not None else 0.0,
                        'unlevered_data': float(first_result[3]) if first_result[3] is not None else 0.0
                    }
                    logger.info(f"Successfully fetched industry metrics for {selected_industry}")
                    return metric_dict
                    
                logger.warning(f"No metrics found for industry: {selected_industry}")
                return None
    
    except Exception as e:
        logger.error(f"Error fetching industry metrics: {str(e)}")
        return None

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to retrieve financial data for a company from the database
def fetch_financial_data(symbol):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Query to fetch all available financial metrics with fiscal dates
                query = """
                WITH RankedMetrics AS (
                    SELECT 
                        c.company_id,
                        fr.period_id,
                        c.symbol,
                        c.sector,
                        rp.fiscal_date_ending,
                        MAX(CASE WHEN m.metric_name = 'totalRevenue' THEN fd.value END) AS "totalRevenue",
                        MAX(CASE WHEN m.metric_name = 'ebit' THEN fd.value END) AS "ebit",
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
                        ROW_NUMBER() OVER (PARTITION BY c.symbol ORDER BY rp.fiscal_date_ending DESC) as row_num
                    FROM FinancialData fd
                    JOIN FinancialMetrics m ON fd.metric_id = m.metric_id
                    JOIN FinancialReports fr ON fd.report_id = fr.report_id
                    JOIN ReportingPeriods rp ON fr.period_id = rp.period_id
                    JOIN Companies c ON fr.company_id = c.company_id
                    WHERE 
                        c.symbol = %s AND 
                        rp.period_type = 'Annual'
                    GROUP BY c.company_id, fr.period_id, c.symbol, c.sector, rp.fiscal_date_ending
                )
                SELECT *
                FROM RankedMetrics
                ORDER BY fiscal_date_ending DESC;
                """
                
                cur.execute(query, (symbol,))
                results = cur.fetchall()
                
                if not results:
                    logger.warning(f"No financial data found for {symbol}")
                    return None
                
                # Create DataFrame with explicit column names
                columns = [
                    'company_id', 'period_id', 'symbol', 'sector', 'fiscal_date_ending', 
                    'totalRevenue', 'ebit', 'researchAndDevelopment', 'interestExpense',
                    'incomeBeforeTax', 'incomeTaxExpense', 'cashAndShortTermInvestments',
                    'totalCurrentAssets', 'totalCurrentLiabilities', 'totalAssets',
                    'totalLiabilities', 'depreciationAndAmortization', 'capitalLeaseObligations',
                    'longTermDebt', 'currentLongTermDebt', 'propertyPlantEquipment', 'row_num'
                ]
                df = pd.DataFrame(results, columns=columns)
                logger.info(f"Successfully fetched {len(df)} years of financial data for {symbol}")
                return df
                
    except Exception as e:
        logger.error(f"Error fetching financial data for {symbol}: {str(e)}")
        return None

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Main function hanlding all financial calculations and regression data preparation
def calculate_financial_metrics(df, industry_metrics):
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

        # Calculate basic metrics first
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

        # Calculate metrics needed for regression analysis
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

        # Calculate beta and cost metrics
        df['Levered Beta'] = unlevered_data * (1 + df['Debt to Equity'] * (1 - df['Tax Rate']))
        df['Cost of Debt'] = 0.045 + 0.044
        df['Cost of Equity'] = 0.045 + df['Levered Beta'] * 0.055
        df['WACC'] = (
            df['Cost of Debt'] * (1 - df['Tax Rate']) * df['Debt to Equity'] +
            df['Cost of Equity'] * (1 - df['Debt to Equity'])
        )

        # Calculate DCF and multiples
        df['DCF'] = (
            ((1 - df['3Y RIR']) * (1- df['3Y Rev Growth']) * (1 - ((1 + df['3Y Rev Growth'])**3 / (1 + df['WACC'])**3))) / 
            (df['WACC'] - df['3Y Rev Growth']) +
            ((1 - industry_rir) * (1 + df['3Y Rev Growth'])**3 * (1 + df['3Y Exp Growth'])) /
            ((industry_wacc - df['3Y Exp Growth']) * (1 + df['WACC'])**3)
        )

        # Calculate enterprise value multiples
        df['EV/EBIT'] = (1 - df['Tax Rate']) * df['DCF']
        df['EV/EBITDA'] = (1 - df['Tax Rate']) * (1 - df['DA']) * df['DCF']
        df['EV/After Tax EBIT'] = 1 * df['DCF']
        df['EV/Sales'] = df['After Tax Operating Margin'] * df['DCF']
        df['EV/Capital Invested'] = df['Return On Invested Capital'] * df['DCF']

        # Create a separate DataFrame for regression metrics if more than 8 companies
        regression_metrics = {
            'EV/EBIT': ['Return On Invested Capital', '3Y Rev Growth'],
            'EV/After Tax EBIT': ['Return On Invested Capital', 'After Tax Operating Margin'],
            'EV/EBITDA': ['Return On Invested Capital', 'DA'],
            'EV/Sales': ['After Tax Operating Margin', '3Y Rev Growth'],
            'EV/Capital Invested': ['Return On Invested Capital', '3Y Exp Growth']
        }

        # Store regression-relevant metrics
        regression_data = {}
        for multiple, (x1, x2) in regression_metrics.items():
            regression_data[multiple] = df[[multiple, x1, x2, 'company_id', 'period_id', 'fiscal_date_ending']].copy()

        # Fill NaNs with 0 for final return
        df.fillna(0, inplace=True)
        return df, regression_data

    except Exception as e:
        logger.error(f"Error calculating financial metrics: {str(e)}", exc_info=True)
        return None, None

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Main function to save all calculated metrics to the database
def save_calculated_metrics(df, company_id, period_id):
    conn = None
    cur = None
    try:
        # Convert NumPy integers to Python integers
        if isinstance(company_id, (np.int64, np.int32)):
            company_id = int(company_id)
        if isinstance(period_id, (np.int64, np.int32)):
            period_id = int(period_id)
            
        with get_db_connection() as conn:
            with conn.cursor() as cur:
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

                    # Get value and ensure it's a Python native type
                    value = df[col].iloc[0]
                    if isinstance(value, (np.int64, np.int32)):
                        value = int(value)
                    elif isinstance(value, (np.float64, np.float32)):
                        value = float(value)

                    # Insert calculated metric values
                    cur.execute("""
                        INSERT INTO CalculatedMetrics 
                        (company_id, period_id, metric_name, metric_value, data_source)
                        VALUES (%s, %s, %s, %s, 'calculated')
                        ON CONFLICT (company_id, period_id, metric_name)
                        DO UPDATE SET 
                            metric_value = EXCLUDED.metric_value,
                            updated_at = CURRENT_TIMESTAMP
                    """, (company_id, period_id, col, value))
                
                conn.commit()
                logger.info(f"Saved calculated metrics for company_id {company_id}, period_id {period_id}")
        
    except Exception as e:
        logger.error(f"Error saving calculated metrics: {str(e)}")
        if conn and not conn.closed:
            conn.rollback()

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Main function to run regression analysis and save results
def analyze_financial_data(data, selected_industry, confidence=0.90):
    try:
        # Define regression metrics and their predictors
        regression_format = {
            'EV/EBIT': ['Return On Invested Capital', '3Y Rev Growth'],
            'EV/After Tax EBIT': ['Return On Invested Capital', 'After Tax Operating Margin'],
            'EV/EBITDA': ['Return On Invested Capital', 'DA'],
            'EV/Sales': ['After Tax Operating Margin', '3Y Rev Growth'],
            'EV/Capital Invested': ['Return On Invested Capital', '3Y Exp Growth']
        }

        # Get required identifiers
        company_id = data['company_id'].iloc[0]
        period_id = data['period_id'].iloc[0]
        fiscal_date_ending = data['fiscal_date_ending'].iloc[0]

        logger.info(f"Processing regression analysis for company {company_id}, period {period_id}")

        results = {}
        # Process each multiple regression separately
        for multiple, (x1_name, x2_name) in regression_format.items():
            # Extract only necessary columns for this regression
            regression_cols = [multiple, x1_name, x2_name]
            numeric_data = data[regression_cols].apply(pd.to_numeric, errors='coerce')
            
            if numeric_data.isna().any().any():
                logger.warning(f"Missing data for {multiple} regression, skipping...")
                continue

            X = np.column_stack((numeric_data[x1_name].values, numeric_data[x2_name].values))
            y = numeric_data[multiple].values

            try:
                # Fit model with constant term
                X_with_const = sm.add_constant(X)
                model = sm.OLS(y, X_with_const).fit()
                
                # Calculate theoretical values
                coefficients = {
                    'y_intercept': float(model.params[0]),
                    'c1_coefficient': float(model.params[1]),
                    'c2_coefficient': float(model.params[2])
                }
                theoretical_values = calculate_theoretical_values(X, coefficients)
                
                # Create DataFrame with theoretical values
                theoretical_df = pd.DataFrame({
                    f"{multiple}_Theoretical": theoretical_values,
                    'company_id': company_id,
                    'period_id': period_id,
                    'fiscal_date_ending': fiscal_date_ending
                })
                
                # Save theoretical values
                save_calculated_metrics(theoretical_df, company_id, period_id)
                
                # Run regression diagnostics and save results
                _, _, diagnostics, plot_binary = calculate_predictions_and_confidence(
                    X, y, multiple, x1_name, x2_name, 0, confidence, selected_industry
                )
                save_regression_results(company_id, period_id, multiple, model, diagnostics, plot_binary)
                
                results[multiple] = {
                    'theoretical_values': theoretical_values,
                    'diagnostics': diagnostics
                }
                
                logger.info(f"Completed regression analysis for {multiple}")

            except Exception as e:
                logger.error(f"Failed regression analysis for {multiple}: {str(e)}")
                continue

        return results if results else None

    except Exception as e:
        logger.error(f"Error in analyze_financial_data: {str(e)}")
        raise

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to calculate theoretical values based on regression coefficients
def calculate_theoretical_values(X, coefficients):
    
    return (coefficients['y_intercept'] + 
            coefficients['c1_coefficient'] * X[:, 0] + 
            coefficients['c2_coefficient'] * X[:, 1])

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to create a beautiful 3D plot of the regression model
def create_beautiful_3d_plot(X, y, model, x1_name, x2_name, multiple, selected_industry=None, confidence=0.90):
    # Create meshgrid with buffer
    num_points = 100
    buffer_ratio = 0.10

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    x_buffer = buffer_ratio * (x_max - x_min)
    y_buffer = buffer_ratio * (y_max - y_min)

    X1 = np.linspace(x_min - x_buffer, x_max + x_buffer, num_points)
    X2 = np.linspace(y_min - y_buffer, y_max + y_buffer, num_points)
    x1_mesh, x2_mesh = np.meshgrid(X1, X2)

    # Predict on mesh
    X_pred = np.column_stack((x1_mesh.ravel(), x2_mesh.ravel()))
    X_pred_with_const = sm.add_constant(X_pred)
    y_pred = model.predict(X_pred_with_const)

    # Calculate prediction standard errors for the mesh
    X_with_const = sm.add_constant(X)
    mse = np.sum(model.resid**2) / (len(y) - X_with_const.shape[1])
    
    # Calculate prediction variance for each point in the mesh
    prediction_var = np.zeros(len(X_pred))
    for i in range(len(X_pred)):
        x_i = X_pred_with_const[i:i+1]
        prediction_var[i] = mse * (1 + x_i.dot(np.linalg.inv(X_with_const.T.dot(X_with_const))).dot(x_i.T))
    
    # Calculate confidence intervals
    t_value = t.ppf((1 + confidence) / 2, df=model.df_resid)
    margin = t_value * np.sqrt(prediction_var)

    # Reshape predictions and confidence bounds
    z_mesh = y_pred.reshape((num_points, num_points))
    z_mesh_upper = (y_pred + margin).reshape((num_points, num_points))
    z_mesh_lower = (y_pred - margin).reshape((num_points, num_points))

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set background color
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Plot actual data points
    ax.scatter(
        X[:, 0], X[:, 1], y,
        s=100,
        color=PLOT_COLORS['scatter'],
        edgecolors=PLOT_COLORS['scatter_edge'],
        linewidths=2.0,
        alpha=1.0,
        label='Actual Data',
        zorder=10
    )

    # Plot regression surface
    surf = ax.plot_surface(
        x1_mesh, x2_mesh, z_mesh,
        cmap=PLOT_COLORS['surface_cmap'],
        alpha=0.6,
        edgecolor='gray',
        linewidth=0.5,
        rstride=10, cstride=10
    )

    # Plot confidence interval surfaces with linear bounds
    ax.plot_surface(
        x1_mesh, x2_mesh, z_mesh_upper,
        color=PLOT_COLORS['confidence'], 
        alpha=0.2, 
        edgecolor='none',
        label=f'{int(confidence * 100)}% CI Upper Bound'
    )
    ax.plot_surface(
        x1_mesh, x2_mesh, z_mesh_lower,
        color=PLOT_COLORS['confidence'], 
        alpha=0.2, 
        edgecolor='none',
        label=f'{int(confidence * 100)}% CI Lower Bound'
    )

    # Add colorbar
    mappable = plt.cm.ScalarMappable(cmap=PLOT_COLORS['surface_cmap'])
    mappable.set_array(z_mesh)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, pad=0.1)

    # Set labels and title
    ax.set_xlabel(f'{x1_name}', labelpad=20)
    ax.set_ylabel(f'{x2_name}', labelpad=20)
    ax.set_zlabel(f'{multiple}', labelpad=20)
    
    title = f"3D Regression Model of {selected_industry} Industry\n"
    title += f"Multiple: {multiple}, ({int(confidence * 100)}% CI)"
    fig.suptitle(title, fontsize=20, y=0.95)

    # Set axis limits with buffer
    ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
    ax.set_ylim(y_min - y_buffer, y_max + y_buffer)

    z_min = min(np.min(z_mesh_lower), y.min())
    z_max = max(np.max(z_mesh_upper), y.max())
    z_buffer = 0.2 * (z_max - z_min)
    ax.set_zlim(z_min - z_buffer, z_max + z_buffer)

    # Style grid
    ax.xaxis._axinfo['grid'].update(color=PLOT_COLORS['grid'], linestyle='--', alpha=0.3)
    ax.yaxis._axinfo['grid'].update(color=PLOT_COLORS['grid'], linestyle='--', alpha=0.3)
    ax.zaxis._axinfo['grid'].update(color=PLOT_COLORS['grid'], linestyle='--', alpha=0.3)

    # Set view angle
    ax.view_init(elev=35, azim=140)

    # Add custom legend
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', label='Actual Data',
               markerfacecolor=PLOT_COLORS['scatter'], 
               markeredgecolor=PLOT_COLORS['scatter_edge'], 
               markersize=10),
        Line2D([0], [0], color='black', lw=4, label='Regression Surface'),
        Line2D([0], [0], color=PLOT_COLORS['confidence'], lw=4, 
               linestyle='--', label=f'{int(confidence * 100)}% Confidence Interval')
    ]

    fig.legend(
        handles=custom_lines,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.88),
        ncol=3,
        frameon=False
    )

    return fig, ax

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Function for regression analysis, statistical tests, and plotting
def calculate_predictions_and_confidence(X, y, multiple, x1_name, x2_name, index, confidence=0.90, selected_industry=None):
    try:
        # Fit the model with constant term
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()
        
        # Calculate residuals for diagnostics
        residuals = y - model.predict(X_with_const)
        standardized_residuals = residuals / np.std(residuals)
        
        # Statistical tests - use X_with_const for Breusch-Pagan test
        bp_test = het_breuschpagan(residuals, X_with_const)
        vif = [variance_inflation_factor(X_with_const, i) for i in range(X_with_const.shape[1])]
        
        # Calculate MSE and create prediction variance array
        mse = np.sum(model.resid**2) / (len(y) - X_with_const.shape[1])
        X_inv = np.linalg.inv(X_with_const.T.dot(X_with_const))
        prediction_var = np.zeros(len(X))
        
        # Fix deprecation warning by extracting scalar values properly
        for i in range(len(X)):
            x_i = X_with_const[i:i+1]
            var_i = mse * (1 + x_i.dot(X_inv).dot(x_i.T))
            prediction_var[i] = var_i.item()  # Properly extract scalar value
            
        # Test for linearity using properly extracted scalar values
        reset_pvalue = 1 - stats.f.cdf(
            ((model.rsquared - sm.OLS(y, np.column_stack([X_with_const, model.predict(X_with_const)**2])).fit().rsquared) / 1) / 
            ((1 - model.rsquared) / model.df_resid),
            1, model.df_resid
        )
        
        # Create beautiful 3D plot
        fig, ax = create_beautiful_3d_plot(
            X, y, model, x1_name, x2_name, 
            multiple, selected_industry, confidence
        )
        
        # Save plot to binary PDF with high quality
        buf = io.BytesIO()
        plt.savefig(buf, format='pdf', dpi=300, bbox_inches='tight')
        plot_binary = buf.getvalue()
        buf.close()
        plt.close()
        
        # Package diagnostic results
        diagnostics = {
            'Metric': ['Heteroscedasticity', 'Multicollinearity', 'Normality', 'Linearity'],
            'Value': [
                bp_test[1], 
                max(vif[1:]), 
                np.mean(np.abs(standardized_residuals)),
                reset_pvalue
            ],
            'Pass': [
                bp_test[1] > 0.05,
                all(v < 5 for v in vif[1:]),
                np.mean(np.abs(standardized_residuals)) < 2,
                reset_pvalue > 0.05
            ]
        }
        
        return model.conf_int(1 - confidence), model, diagnostics, plot_binary
        
    except Exception as e:
        logger.error(f"Error in calculate_predictions_and_confidence: {str(e)}")
        raise

#------------------------------------------------------------------------------------------------------------------------------------------------------    

# Function that defines the formulas for calculated metrics
def get_metric_formula(metric_name):
    
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

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to save regression and statistical results to the database
def save_regression_results(company_id, period_id, multiple, model, diagnostics, plot_binary):
    conn = None
    cur = None
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Prepare statistical data
                stats_data = {
                    'r_squared': float(model.rsquared),
                    'adj_r_squared': float(model.rsquared_adj),
                    'f_statistic': float(model.fvalue),
                    'f_pvalue': float(model.f_pvalue),
                    'aic': float(model.aic),
                    'bic': float(model.bic)
                }
                
                # Prepare model parameters
                model_params = {
                    'coefficients': model.params.tolist(),
                    'standard_errors': model.bse.tolist(),
                    't_values': model.tvalues.tolist(),
                    'p_values': model.pvalues.tolist(),
                    'confidence_intervals': model.conf_int().tolist()
                }
                
                # Convert diagnostics to proper JSON format
                diagnostic_data = {
                    'heteroscedasticity_test': {
                        'test_value': float(diagnostics['Value'][0]),
                        'passes': diagnostics['Pass'][0]
                    },
                    'multicollinearity_test': {
                        'test_value': float(diagnostics['Value'][1]),
                        'passes': diagnostics['Pass'][1]
                    },
                    'normality_test': {
                        'test_value': float(diagnostics['Value'][2]),
                        'passes': diagnostics['Pass'][2]
                    },
                    'linearity_test': {
                        'test_value': float(diagnostics['Value'][3]),
                        'passes': diagnostics['Pass'][3]
                    }
                }
                
                # Insert main regression analysis results
                cur.execute("""
                    INSERT INTO regression_analysis (
                        company_id, period_id, multiple_type,
                        r_squared, adj_r_squared, f_statistic, f_pvalue, aic, bic,
                        coefficients, standard_errors, t_values, p_values, confidence_intervals,
                        heteroscedasticity_test, multicollinearity_test, normality_test, linearity_test,
                        regression_plot
                    )
                    VALUES (
                        %s, %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s
                    )
                    ON CONFLICT (company_id, period_id, multiple_type)
                    DO UPDATE SET
                        r_squared = EXCLUDED.r_squared,
                        adj_r_squared = EXCLUDED.adj_r_squared,
                        f_statistic = EXCLUDED.f_statistic,
                        f_pvalue = EXCLUDED.f_pvalue,
                        aic = EXCLUDED.aic,
                        bic = EXCLUDED.bic,
                        coefficients = EXCLUDED.coefficients,
                        standard_errors = EXCLUDED.standard_errors,
                        t_values = EXCLUDED.t_values,
                        p_values = EXCLUDED.p_values,
                        confidence_intervals = EXCLUDED.confidence_intervals,
                        heteroscedasticity_test = EXCLUDED.heteroscedasticity_test,
                        multicollinearity_test = EXCLUDED.multicollinearity_test,
                        normality_test = EXCLUDED.normality_test,
                        linearity_test = EXCLUDED.linearity_test,
                        regression_plot = EXCLUDED.regression_plot,
                        analysis_date = CURRENT_TIMESTAMP
                    RETURNING analysis_id
                """, (
                    company_id, period_id, multiple,
                    stats_data['r_squared'], stats_data['adj_r_squared'],
                    stats_data['f_statistic'], stats_data['f_pvalue'],
                    stats_data['aic'], stats_data['bic'],
                    json.dumps(model_params['coefficients']),
                    json.dumps(model_params['standard_errors']),
                    json.dumps(model_params['t_values']),
                    json.dumps(model_params['p_values']),
                    json.dumps(model_params['confidence_intervals']),
                    json.dumps(diagnostic_data['heteroscedasticity_test']),
                    json.dumps(diagnostic_data['multicollinearity_test']),
                    json.dumps(diagnostic_data['normality_test']),
                    json.dumps(diagnostic_data['linearity_test']),
                    plot_binary
                ))
                
                # Get the analysis_id for storing coefficients
                analysis_id = cur.fetchone()[0]
                
                # Store coefficients for theoretical calculations
                regression_format = {
                    'EV/EBIT': ['Return On Invested Capital', '3Y Rev Growth'],
                    'EV/After Tax EBIT': ['Return On Invested Capital', 'After Tax Operating Margin'],
                    'EV/EBITDA': ['Return On Invested Capital', 'DA'],
                    'EV/Sales': ['After Tax Operating Margin', '3Y Rev Growth'],
                    'EV/Capital Invested': ['Return On Invested Capital', '3Y Exp Growth']
                }
                
                x1_name, x2_name = regression_format[multiple]
                
                # Store coefficients in RegressionCoefficients table
                cur.execute("""
                    INSERT INTO RegressionCoefficients (
                        analysis_id, multiple_type, x1_name, x2_name,
                        c1_coefficient, c2_coefficient, y_intercept
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (analysis_id, multiple_type)
                    DO UPDATE SET
                        x1_name = EXCLUDED.x1_name,
                        x2_name = EXCLUDED.x2_name,
                        c1_coefficient = EXCLUDED.c1_coefficient,
                        c2_coefficient = EXCLUDED.c2_coefficient,
                        y_intercept = EXCLUDED.y_intercept
                """, (
                    analysis_id, multiple, x1_name, x2_name,
                    float(model.params[1]), float(model.params[2]), float(model.params[0])
                ))
                
                # Insert detailed diagnostic results
                for test_type, (metric, value, passes) in zip(
                    ['Heteroscedasticity', 'Multicollinearity', 'Normality', 'Linearity'],
                    zip(diagnostics['Metric'], diagnostics['Value'], diagnostics['Pass'])
                ):
                    cur.execute("""
                        INSERT INTO regression_diagnostics (
                            analysis_id, diagnostic_type, test_value,
                            passes_threshold, threshold_value
                        )
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        analysis_id, test_type, float(value),
                        passes, 0.05 if test_type in ['Heteroscedasticity', 'Linearity'] else (5.0 if test_type == 'Multicollinearity' else 2.0)
                    ))
                
                conn.commit()
                logger.info(f"Successfully saved regression results for {multiple}")
                
    except Exception as e:
        logger.error(f"Error saving regression results: {str(e)}")
        if conn and not conn.closed:
            conn.rollback()
        raise

#------------------------------------------------------------------------------------------------------------------------------------------------------    

# Main function to execute the program
def main():
    try:
        # Set working directory
        os.chdir('Projects/Data Analysis/S&P500 Valuation Multiples')

        # Load SP500 data
        sp500_df = load_sp500_data()
        if sp500_df is None:
            return None

        # Get industry company counts and sort from smallest to largest
        industry_counts = sp500_df['Sector'].value_counts().sort_values()
        logger.info("Industry counts (sorted from smallest to largest):")
        for industry, count in industry_counts.items():
            logger.info(f"{industry}: {count} companies")

        # Process each industry
        for industry, count in industry_counts.items():
            logger.info(f"\nProcessing {industry} with {count} companies")
            
            # Get companies in this industry
            industry_companies = sp500_df[sp500_df['Sector'] == industry]['Symbol'].tolist()
            
            # Get industry metrics
            industry_metrics = fetch_industry_metrics(industry)
            if not industry_metrics:
                logger.warning(f"No industry metrics available for {industry}")
                continue

            # Store for regression analysis if industry has enough companies
            regression_ready = count >= 8
            if regression_ready:
                industry_regression_data = {
                    'metrics': [],  # Store regression metrics for all companies
                    'company_data': {}  # Map of company_id to their data
                }

            # Process each company in the industry
            for symbol in industry_companies:
                # Fetch financial data for the company
                financial_data = fetch_financial_data(symbol)
                if financial_data is None:
                    continue

                # Calculate metrics for the company
                calculated_metrics, regression_metrics = calculate_financial_metrics(financial_data, industry_metrics)
                if calculated_metrics is None:
                    continue

                # Save calculated metrics for all companies
                for _, row in calculated_metrics.iterrows():
                    save_calculated_metrics(pd.DataFrame([row]), row['company_id'], row['period_id'])

                # If industry qualifies for regression, store the regression metrics
                if regression_ready:
                    company_id = calculated_metrics['company_id'].iloc[0]
                    industry_regression_data['company_data'][company_id] = regression_metrics
                    
            # Perform regression analysis if industry qualifies
            if regression_ready:
                logger.info(f"Performing regression analysis for {industry}")
                
                # Get the maximum number of years available across all companies
                max_years = min(5, max(
                    len(metrics[next(iter(metrics))]) 
                    for metrics in industry_regression_data['company_data'].values()
                ))

                # Process each year, starting from the most recent
                for year_index in range(max_years):
                    year_data = []
                    
                    # Collect data for this year from each company
                    for company_id, company_metrics in industry_regression_data['company_data'].items():
                        for multiple, metrics_df in company_metrics.items():
                            if len(metrics_df) > year_index:
                                year_data.append(metrics_df.iloc[year_index])
                    
                    if len(year_data) >= 8:  # Ensure enough companies have data for this year
                        year_df = pd.DataFrame(year_data)
                        analyzed_data = analyze_financial_data(year_df, industry)
                        if analyzed_data is not None:
                            logger.info(f"Completed regression analysis for {industry} - Year {year_index + 1}")
                    else:
                        logger.warning(f"Insufficient data for regression analysis in {industry} - Year {year_index + 1}")

            logger.info(f"Completed processing {industry}")

        logger.info("Completed processing all industries")
        return True

    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}", exc_info=True)
        return None

#------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

