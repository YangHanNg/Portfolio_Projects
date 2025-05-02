# Import necessary libraries
import pandas as pd
import requests
import json
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
import os
import time
from datetime import datetime
import pickle
from pathlib import Path
from dotenv import load_dotenv
from contextlib import contextmanager
from typing import Dict, List, Any

# Load environment variables from .env file
load_dotenv()

# Cache configuration for temporary storage of processing state
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"
CACHE_FILE = DATA_DIR / "processing_state.pkl"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True, parents=True)

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Class module for processing cache
class ProcessingCache:
    def __init__(self):
        self.processed_industries: List[str] = []
        self.current_industry: str = None
        self.last_company: str = None
        self.start_time: float = None
        self.industry_times: Dict[str, float] = {}

    @classmethod
    def load(cls) -> 'ProcessingCache':
        try:
            if CACHE_FILE.exists():
                with open(CACHE_FILE, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
        return cls()

    def save(self):
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def mark_industry_complete(self, industry: str, processing_time: float):
        self.processed_industries.append(industry)
        self.industry_times[industry] = processing_time
        self.current_industry = None
        self.last_company = None
        self.save()

    def start_industry(self, industry: str):
        self.current_industry = industry
        self.start_time = time.time()
        self.save()

    def update_progress(self, company: str):
        self.last_company = company
        self.save()

# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "financial_data.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database configuration from environment variables
DB_CONFIG = {
    "dbname": "postgres",  # Input the database name here
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# API directory configuration
API_URL = "https://www.alphavantage.co/query"

# Constants for report types
REPORT_TYPES = {
    "BALANCE_SHEET": "Balance Sheet",
    "INCOME_STATEMENT": "Income Statement",
    "CASH_FLOW": "Cash Flow"
}

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Context manager for database connections
@contextmanager
def get_db_connection(dbname=None):
    
    conn_params = DB_CONFIG.copy()
    
    # Always use postgres database
    conn_params["dbname"] = "postgres"
    
    conn = psycopg2.connect(**conn_params)
    try:
        yield conn
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    else:
        conn.commit()
    finally:
        conn.close()

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to initialize the database and create tables function
def initialize_database():
    """Initialize the database and create necessary tables."""
    try:
        create_database()
        create_tables()
        logger.info("Database initialization complete")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to execute SQL queries with error handling
def execute_query(query, params=None, fetch=False, many=False, dbname=None):
    """
    Args:
        query (str): SQL query to execute
        params (tuple or list, optional): Parameters for the query
        fetch (bool, optional): Whether to fetch results
        many (bool, optional): Whether to execute many statements
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            try:
                if many and params:
                    cursor.executemany(query, params)
                else:
                    cursor.execute(query, params)
                
                if fetch:
                    return cursor.fetchall()
                if cursor.description and not fetch:
                    return cursor.fetchone()
                
            except Exception as e:
                logger.error(f"Query execution error: {str(e)}")
                logger.error(f"Query: {query}")
                logger.error(f"Params: {params}")
                raise

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to create the database if it doesn't exist
def create_database():
    """Initialize the postgres database schema if needed."""
    try:
        # Connect directly to postgres database
        with get_db_connection() as conn:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            logger.info("Connected to postgres database successfully")
            return True
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to create all necessary tables for the financial database
def create_tables():
    """Create all necessary tables for the financial database."""
    try:
        # SQL statements for table creation
        tables = {
            "companies": """
                CREATE TABLE IF NOT EXISTS companies (
                    company_id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) UNIQUE NOT NULL,
                    name VARCHAR(255),
                    industry_id INTEGER REFERENCES Industries(industry_id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "industries": """
                CREATE TABLE IF NOT EXISTS industries (
                    industry_id SERIAL PRIMARY KEY,
                    sector_name VARCHAR(100) NOT NULL,
                    industry_name VARCHAR(100) NOT NULL,
                    cost_of_capital DECIMAL(7,4),
                    growth_rate DECIMAL(7,4),
                    reinvestment_rate DECIMAL(7,4),
                    unlevered_data DECIMAL(7,4),
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT unique_industry_sector UNIQUE (sector_name, industry_name)
                )
            """,
            "reporting_periods": """
                CREATE TABLE IF NOT EXISTS reporting_periods (
                    period_id SERIAL PRIMARY KEY,
                    period_type VARCHAR(20) NOT NULL,
                    fiscal_date_ending DATE NOT NULL,
                    year INTEGER NOT NULL,
                    quarter INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (period_type, fiscal_date_ending)
                )
            """,
            "financial_reports": """
                CREATE TABLE IF NOT EXISTS financial_reports (
                    report_id SERIAL PRIMARY KEY,
                    company_id INTEGER REFERENCES Companies(company_id),
                    period_id INTEGER REFERENCES ReportingPeriods(period_id),
                    report_type VARCHAR(50) NOT NULL,
                    reported_currency VARCHAR(10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (company_id, period_id, report_type)
                )
            """,
            "financial_metrics": """
                CREATE TABLE IF NOT EXISTS financial_metrics (
                    metric_id SERIAL PRIMARY KEY,
                    metric_name VARCHAR(100) UNIQUE NOT NULL,
                    display_name VARCHAR(100),
                    category VARCHAR(50),
                    data_type VARCHAR(20) DEFAULT 'currency',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "financial_data": """
                CREATE TABLE IF NOT EXISTS financial_data (
                    data_id SERIAL PRIMARY KEY,
                    report_id INTEGER REFERENCES FinancialReports(report_id),
                    metric_id INTEGER REFERENCES FinancialMetrics(metric_id),
                    value NUMERIC,
                    is_null BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (report_id, metric_id)
                )
            """,
            "calculated_metrics": """
                CREATE TABLE IF NOT EXISTS calculated_metrics (
                    calc_metric_id SERIAL PRIMARY KEY,
                    company_id INTEGER REFERENCES Companies(company_id),
                    period_id INTEGER REFERENCES ReportingPeriods(period_id),
                    metric_id INTEGER REFERENCES FinancialMetrics(metric_id),
                    metric_value DECIMAL(19,4) NOT NULL,
                    calculation_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    data_source VARCHAR(50),
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT unique_calc_metric UNIQUE (company_id, period_id, metric_id)
                )
            """,
            "calculated_metric_definitions": """
                CREATE TABLE IF NOT EXISTS calculated_metric_definitions (
                    definition_id SERIAL PRIMARY KEY,
                    metric_name VARCHAR(100) UNIQUE NOT NULL,
                    formula TEXT NOT NULL,
                    description TEXT,
                    units VARCHAR(50),
                    is_percentage BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "regression_analysis": """
                CREATE TABLE IF NOT EXISTS regression_analysis (
                    analysis_id SERIAL PRIMARY KEY,
                    company_id INTEGER REFERENCES Companies(company_id),
                    period_id INTEGER REFERENCES ReportingPeriods(period_id),
                    multiple_type VARCHAR(50),
                    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    r_squared FLOAT,
                    adj_r_squared FLOAT,
                    f_statistic FLOAT,
                    f_pvalue FLOAT,
                    aic FLOAT,
                    bic FLOAT,
                    coefficients JSONB,
                    standard_errors JSONB,
                    t_values JSONB,
                    p_values JSONB,
                    confidence_intervals JSONB,
                    heteroscedasticity_test JSONB,
                    multicollinearity_test JSONB,
                    normality_test JSONB,
                    linearity_test JSONB,
                    regression_plot BYTEA,
                    UNIQUE(company_id, period_id, multiple_type)
                )
            """,
            "regression_diagnostics": """
                CREATE TABLE IF NOT EXISTS regression_diagnostics (
                    diagnostic_id SERIAL PRIMARY KEY,
                    analysis_id INTEGER REFERENCES regression_analysis(analysis_id),
                    diagnostic_type VARCHAR(50),
                    test_name VARCHAR(100),
                    test_value FLOAT,
                    p_value FLOAT,
                    passes_threshold BOOLEAN,
                    threshold_value FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "regression_coefficients": """
                CREATE TABLE IF NOT EXISTS regression_coefficients (
                    coefficient_id SERIAL PRIMARY KEY,
                    analysis_id INTEGER REFERENCES regression_analysis(analysis_id),
                    multiple_type VARCHAR(50),
                    x1_name VARCHAR(100),
                    x2_name VARCHAR(100),
                    c1_coefficient FLOAT,
                    c2_coefficient FLOAT,
                    y_intercept FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(analysis_id, multiple_type)
                )
            """
        }
        
        # Execute all table creation statements
        for name, query in tables.items():
            execute_query(query)
            logger.debug(f"Created table: {name}")
        
        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_financial_data_report_id ON FinancialData(report_id)",
            "CREATE INDEX IF NOT EXISTS idx_financial_reports_company_id ON FinancialReports(company_id)",
            "CREATE INDEX IF NOT EXISTS idx_financial_reports_period_id ON FinancialReports(period_id)",
            "CREATE INDEX IF NOT EXISTS idx_companies_sector ON Companies(sector)",
            "CREATE INDEX IF NOT EXISTS idx_companies_industry ON Companies(industry_id)",
            "CREATE INDEX IF NOT EXISTS idx_industries_sector ON Industries(sector_name)",
            "CREATE INDEX IF NOT EXISTS idx_calculated_metrics_company ON CalculatedMetrics(company_id)",
            "CREATE INDEX IF NOT EXISTS idx_calculated_metrics_period ON CalculatedMetrics(period_id)",
            "CREATE INDEX IF NOT EXISTS idx_calculated_metrics_metric ON CalculatedMetrics(metric_id)",
            "CREATE INDEX IF NOT EXISTS idx_regression_company ON regression_analysis(company_id)",
            "CREATE INDEX IF NOT EXISTS idx_regression_period ON regression_analysis(period_id)",
            "CREATE INDEX IF NOT EXISTS idx_regression_multiple ON regression_analysis(multiple_type)",
            "CREATE INDEX IF NOT EXISTS idx_diagnostics_analysis ON regression_diagnostics(analysis_id)",
            "CREATE INDEX IF NOT EXISTS idx_coefficients_analysis ON RegressionCoefficients(analysis_id)",
            "CREATE INDEX IF NOT EXISTS idx_coefficients_multiple ON RegressionCoefficients(multiple_type)"
        ]
        
        # Create all indexes
        for index_query in indexes:
            execute_query(index_query)
            
        logger.info("All tables and indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        raise

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Class module to handle all database queries for operations
class FinancialDataAccess:
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    # Function to retrieve industry-level financial data for a given sector_name
    def get_industry(self, sector_name):
        
        # SQL query to fetch industry data
        query = """
            SELECT cost_of_capital, growth_rate, reinvestment_rate
            FROM industries
            WHERE sector_name = %s
            LIMIT 1;
        """
        try:
            result = execute_query(query, params=(sector_name,), fetch=True)
            if result and len(result) > 0:
                row = result[0]  # Because fetchall returns a list of rows
                return {
                    'cost_of_capital': row[0],
                    'growth_rate': row[1],
                    'reinvestment_rate': row[2]
                }
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to fetch industry data for {sector_name}: {str(e)}")
            return None
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    # Function to insert new company data or update existing one
    @staticmethod
    def insert_company(symbol, name=None, sector_name=None, industry_name=None):
        # First get the industry_id if sector is provided
        industry_id = None
        if sector_name:
            query = """
            SELECT industry_id FROM Industries 
            WHERE sector_name = %s AND industry_name = COALESCE(%s, 'N/A')
            """
            result = execute_query(query, (sector_name, industry_name), fetch=True)
            if result:
                industry_id = result[0][0]
        
        # Then insert the company
        query = """
        INSERT INTO Companies (symbol, name, industry_id)
        VALUES (%s, %s, %s)
        ON CONFLICT (symbol) 
        DO UPDATE SET 
            name = COALESCE(%s, Companies.name),
            industry_id = COALESCE(%s, Companies.industry_id),
            updated_at = CURRENT_TIMESTAMP
        RETURNING company_id
        """
        params = (symbol, name, industry_id, name, industry_id)
        
        result = execute_query(query, params)
        return result[0] if result else None
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    # Function to insert new reporting period or update existing one
    @staticmethod
    def insert_reporting_period(fiscal_date_ending, period_type):
        
        # Extract year and quarter if applicable
        year = int(fiscal_date_ending.split('-')[0])
        quarter = None
        if period_type == 'Quarterly':
            month = int(fiscal_date_ending.split('-')[1])
            quarter = (month - 1) // 3 + 1
        # SQL query to insert or update reporting period data
        query = """
        INSERT INTO ReportingPeriods (period_type, fiscal_date_ending, year, quarter)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (period_type, fiscal_date_ending) 
        DO UPDATE SET 
            year = %s,
            quarter = %s
        RETURNING period_id
        """
        params = (period_type, fiscal_date_ending, year, quarter, year, quarter)
        
        result = execute_query(query, params)
        return result[0] if result else None
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    # Function to insert new financial report or update existing one
    @staticmethod
    def insert_financial_report(company_id, period_id, report_type, reported_currency=None):
        
        # SQL query to insert or update financial report data
        query = """
        INSERT INTO FinancialReports (company_id, period_id, report_type, reported_currency)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (company_id, period_id, report_type) 
        DO UPDATE SET 
            reported_currency = COALESCE(%s, FinancialReports.reported_currency),
            updated_at = CURRENT_TIMESTAMP
        RETURNING report_id
        """
        params = (company_id, period_id, report_type, reported_currency, reported_currency)
        
        result = execute_query(query, params)
        return result[0] if result else None
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    # Function to retrieve or create a financial metric
    @staticmethod
    def get_or_create_metric(metric_name, display_name=None, category=None, data_type='currency'):
       
        # If display_name is not provided, create one from the metric_name
        if not display_name:
            display_name = ' '.join(word.capitalize() for word in metric_name.split('_'))
        
        #SQL query to insert or update financial metric data
        query = """
        INSERT INTO FinancialMetrics (metric_name, display_name, category, data_type)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (metric_name) 
        DO UPDATE SET 
            display_name = COALESCE(%s, FinancialMetrics.display_name),
            category = COALESCE(%s, FinancialMetrics.category),
            data_type = COALESCE(%s, FinancialMetrics.data_type)
        RETURNING metric_id
        """
        params = (metric_name, display_name, category, data_type, display_name, category, data_type)
        
        result = execute_query(query, params)
        return result[0] if result else None
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    # Function to insert financial data or update if it exists
    @staticmethod
    def insert_financial_data(report_id, metric_id, value):
        
        is_null = (value == "None" or value is None)
        numeric_value = None if is_null else value
        
        # SQL query to insert or update financial data
        query = """
        INSERT INTO FinancialData (report_id, metric_id, value, is_null)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (report_id, metric_id) 
        DO UPDATE SET 
            value = %s,
            is_null = %s,
            created_at = CURRENT_TIMESTAMP
        """
        params = (report_id, metric_id, numeric_value, is_null, numeric_value, is_null)
        
        execute_query(query, params)
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    # Function to insert new industry or update existing one
    @staticmethod
    def insert_industry(sector_name, industry_name, cost_of_capital=None, growth_rate=None, reinvestment_rate=None, unlevered_data=None):
        
        # SQL query to insert or update industry data
        query = """
        INSERT INTO Industries (sector_name, industry_name, cost_of_capital, growth_rate, reinvestment_rate, unlevered_data)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (sector_name, industry_name) 
        DO UPDATE SET 
            cost_of_capital = COALESCE(%s, Industries.cost_of_capital),
            growth_rate = COALESCE(%s, Industries.growth_rate),
            reinvestment_rate = COALESCE(%s, Industries.reinvestment_rate),
            unlevered_data = COALESCE(%s, Industries.unlevered_data),
            last_updated = CURRENT_TIMESTAMP
        RETURNING industry_id
        """
        params = (sector_name, industry_name, cost_of_capital, growth_rate, reinvestment_rate, unlevered_data,
                  cost_of_capital, growth_rate, reinvestment_rate, unlevered_data)
        
        result = execute_query(query, params)
        return result[0] if result else None

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Class module to handle S&P 500 data import and processing
class SP500DataImporter:
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    # Function to initialize the importer with CSV path
    def __init__(self, csv_path="data/SP500.csv"):  # Updated default path
        
        self.csv_path = csv_path
        self.db = None
        self.sp500_df = None
    
    #----------------------------------------------------------------------------------------------------------------------------------    

    # Function to connect to the database   
    def connect_db(self):

        if self.db is None:
            self.db = FinancialDataAccess()
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    # Function to load S&P 500 company data from CSV file
    def load_data(self):

        try:
            logger.info(f"Reading SP500 data from {self.csv_path}")
            self.sp500_df = pd.read_csv(self.csv_path)
            return self.sp500_df
        except Exception as e:
            logger.error(f"Failed to load S&P 500 data: {str(e)}")
            raise
            
    #----------------------------------------------------------------------------------------------------------------------------------
    
    # Function to import all industry data from the CSV file
    def import_industry_data(self, sector_name):
        self.connect_db()
        
        if self.sp500_df is None:
            self.load_data()
            
        # Filter companies by industry
        industry_df = self.sp500_df[self.sp500_df['Sector'] == sector_name]
        
        if industry_df.empty:
            logger.warning(f"No companies found for industry: {sector_name}")
            return 0
            
        # Convert DataFrame to list of dictionaries
        industry_data = industry_df.to_dict('records')
        
        # Get industry details from first company
        first_company = industry_data[0]
        
        try:
            # More robust float conversion for unlevered_data
            unlevered_data = float(str(first_company.get('Unlevered Data', 0)).replace(',', '.'))
        except (ValueError, TypeError):
            logger.warning(f"Invalid unlevered data for industry {sector_name}: {first_company.get('Unlevered Data')}")
            unlevered_data = 0.0
        
        # Insert/update the industry into the database
        self.db.insert_industry(
            sector_name=first_company['Sector'],
            industry_name=first_company.get('Industry', 'N/A'),
            cost_of_capital=first_company.get('Industry Cost of Capital'),
            growth_rate=first_company.get('Industry Growth'),
            reinvestment_rate=first_company.get('Industry Reinvestment rate'),
            unlevered_data=unlevered_data
        )
        
        # Insert all companies in this industry
        for company in industry_data:
            try:
                # More robust float conversion
                unlevered_data = float(str(company.get('Unlevered Data', 0)).replace(',', '.'))
            except (ValueError, TypeError):
                logger.warning(f"Invalid unlevered data for {company['Symbol']}: {company.get('Unlevered Data')}")
                unlevered_data = 0.0
        
            self.db.insert_company(
                symbol=company['Symbol'],
                name=company['Name'],
                sector=company['Sector'],
                industry=company.get('Industry', 'N/A'),
                unlevered_data=unlevered_data
                )
    
        logger.info(f"Imported {len(industry_data)} companies for industry: {sector_name}")
        return len(industry_data)
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    # Function to check the relevance of industry data in the database
    def is_industry_data_current(self, sector_name):
        
        self.connect_db()
    
        if self.sp500_df is None:
            self.load_data()
    
        # Filter companies by industry
        industry_df = self.sp500_df[self.sp500_df['Sector'] == sector_name]
    
        if industry_df.empty:
            logger.warning(f"No companies found for industry: {sector_name}")
            return False
    
        # Get a random company from this industry
        random_company = industry_df.sample(n=1).iloc[0]
    
        # Get the industry data from the database
        db_industry = self.db.get_industry(sector_name=sector_name)
    
        # If industry doesn't exist in database, data is not current
        if not db_industry:
            return False
    
        # Compare the key industry metrics
        csv_cost_of_capital = random_company['Industry Cost of Capital']
        csv_growth_rate = random_company['Industry Growth']
        csv_reinvestment_rate = random_company['Industry Reinvestment rate']
    
        # Check if values are the same (accounting for potential None values)
        if (csv_cost_of_capital == db_industry['cost_of_capital'] and
            csv_growth_rate == db_industry['growth_rate'] and
            csv_reinvestment_rate == db_industry['reinvestment_rate']):
            logger.info(f"Industry data for {sector_name} is current")
            return True
    
        logger.info(f"Industry data for {sector_name} needs updating")
        return False
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    # Function to process the idnustry if data is not current
    def process_industry(self, industry_name):
        
        if self.is_industry_data_current(industry_name):
            logger.info(f"Skipping import for {industry_name} - data is current")
            return False
    
        # Import industry data if not current
        company_count = self.import_industry_data(industry_name)
        logger.info(f"Imported {company_count} companies for {industry_name}")
        return True
    
    #----------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Class module to handle financial data processing operations
class FinancialDataProcessor:
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    # Function to initialize the processor with API key and connection to the database
    def __init__(self, api_key):
        
        self.api_key = api_key
        self.db = FinancialDataAccess()
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    # Function  to fetch financial data from the API source
    def fetch_financial_data(self, symbol, function):
        
        try:
            response = requests.get(API_URL, params={
                "function": function,
                "symbol": symbol,
                "apikey": self.api_key
            }, timeout=30)
            
            response.raise_for_status()
            data = response.json()
            
            # Check for error messages in the response
            if "Error Message" in data:
                logger.error(f"API Error for {symbol}: {data['Error Message']}")
                return None
                
            logger.info(f"Retrieved {function} for {symbol}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error retrieving {function} for {symbol}: {str(e)}")
            return None
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response for {symbol} {function}")
            return None
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    # Main function that process financial data from API request to tables in the database
    def process_financial_data(self, symbol, data, report_type, max_quarterly_reports=12):
        
        if not data:
            logger.warning(f"No data to process for {symbol} {report_type}")
            return
        
        # Insert company data
        company_id = self.db.insert_company(symbol)
        if not company_id:
            logger.error(f"Failed to insert company {symbol}")
            return
    
        # Process annual reports
        if "annualReports" in data:
            self._process_reports(company_id, data["annualReports"], report_type, "Annual")
    
        # Process quarterly reports with limit
        if "quarterlyReports" in data:
            # Sort quarterly reports by date (most recent first)
            if len(data["quarterlyReports"]) > 0 and "fiscalDateEnding" in data["quarterlyReports"][0]:
                sorted_reports = sorted(
                    data["quarterlyReports"], 
                    key=lambda x: x["fiscalDateEnding"], 
                    reverse=True
                    )
                # Take only the most recent reports up to the limit to reduce processing time
                limited_reports = sorted_reports[:max_quarterly_reports]
                logger.info(f"Processing {len(limited_reports)} of {len(sorted_reports)} available quarterly reports for {symbol}")
                self._process_reports(company_id, limited_reports, report_type, "Quarterly")
            else:
                logger.warning(f"Quarterly reports for {symbol} are missing date information")
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    # Function to process the reporting periods
    def _process_reports(self, company_id, reports, report_type, period_type):
        
        for report in reports:
            fiscal_date_ending = report.get("fiscalDateEnding")
            reported_currency = report.get("reportedCurrency")
            
            if not fiscal_date_ending:
                logger.warning(f"Missing fiscal_date_ending in report for company {company_id}")
                continue
                
            # Insert reporting period
            period_id = self.db.insert_reporting_period(fiscal_date_ending, period_type)
            
            # Insert financial report
            report_id = self.db.insert_financial_report(company_id, period_id, report_type, reported_currency)
            
            # Insert financial data
            for key, value in report.items():
                if key not in ["fiscalDateEnding", "reportedCurrency"]:
                    metric_id = self.db.get_or_create_metric(key)
                    self.db.insert_financial_data(report_id, metric_id, value)

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to categorize industries based on company count
def get_categorized_industries(sp500_df, medium_min=6, medium_max=12, automated=False):
    
    if sp500_df is None:
        logger.error("S&P 500 data not loaded")
        return None
    
    industry_counts = sp500_df['Sector'].value_counts()

    # Categorize industries
    small_industries = industry_counts[industry_counts < medium_min]
    medium_industries = industry_counts[(industry_counts >= medium_min) & 
                                  (industry_counts <= medium_max)]
    large_industries = industry_counts[industry_counts > medium_max]

    # Log the medium industries only when not in automated mode
    if not automated:
        logger.info(f"Industries with {medium_min}-{medium_max} companies:")
        for industry, count in medium_industries.items():
            logger.info(f"{industry}: {count} companies")

    # Return categorized industries with counts
    return {
        'small': {
            'industries': small_industries.sort_values().index.tolist(),
            'counts': small_industries.to_dict(),
            'description': f"Industries with <{medium_min} companies"
        },
        'medium': {
            'industries': medium_industries.sort_values().index.tolist(),
            'counts': medium_industries.to_dict(),
            'description': f"Industries with {medium_min}-{medium_max} companies"
        },
        'large': {
            'industries': large_industries.sort_values().index.tolist(),
            'counts': large_industries.to_dict(),
            'description': f"Industries with >{medium_max} companies"
        }
    }

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Main function to process all companies in a given industry
def process_industry(sp500_importer, industry, processor, sp500_df, max_quarterly_reports=12, delay=12):
    
    # Get tickers for the selected industry
    industry_tickers = sp500_df[sp500_df['Sector'] == industry]['Symbol'].tolist()
    logger.info(f"Selected {industry} with {len(industry_tickers)} companies")
    
    # Check if industry data needs updating
    if sp500_importer.is_industry_data_current(industry):
        logger.info(f"Industry data for {industry} is current. Skipping data import.")
        return False

    # Fetch and process data for each ticker
    reports = ['BALANCE_SHEET', 'INCOME_STATEMENT']
    
    for ticker in industry_tickers:
        for report in reports:
            data = processor.fetch_financial_data(ticker, report)
            if data:
                processor.process_financial_data(ticker, data, report, max_quarterly_reports)
            time.sleep(delay)  # Delay between API calls

    logger.info(f"Completed processing {industry}")
    return True

#------------------------------------------------------------------------------------------------------------------------------------------------------

# This function processes all industries in the S&P 500, either automatically or through user selection.
def process_industries(sp500_importer, industry_groups, processor, sp500_df, automated=False):
    
    cache = ProcessingCache.load()
    total_start_time = time.time()

    #----------------------------------------------------------------------------------------------------------------------------------
    if automated:
        # Process industries in order: small -> medium -> large
        categories = [
            ('small', industry_groups['small']['industries']),
            ('medium', industry_groups['medium']['industries']),
            ('large', industry_groups['large']['industries'])
        ]

        for category, industries in categories:
            logger.info(f"\nProcessing {category} industries")
            industry_counts = {ind: len(sp500_df[sp500_df['Sector'] == ind]) for ind in industries}
            sorted_industries = sorted(industries, key=lambda x: industry_counts[x])

            for industry in sorted_industries:
                if industry in cache.processed_industries:
                    logger.info(f"Skipping {industry} - already processed")
                    continue

                cache.start_industry(industry)
                industry_start_time = time.time()

                try:
                    logger.info(f"\nProcessing {industry} ({industry_counts[industry]} companies)")
                    process_industry(sp500_importer, industry, processor, sp500_df)
                    
                    industry_time = time.time() - industry_start_time
                    cache.mark_industry_complete(industry, industry_time)
                    logger.info(f"Completed {industry} in {industry_time:.2f} seconds")

                except Exception as e:
                    logger.error(f"Error processing {industry}: {str(e)}")
                    cache.save()
                    raise

                if industry != sorted_industries[-1]:
                    logger.info("Waiting 60 seconds before next industry...")
                    time.sleep(60)

    #----------------------------------------------------------------------------------------------------------------------------------
    else:
        while True:
            print("\nSelect an industry group to process:")
            print("1. Medium Industries (6-12 companies)")
            print("2. Small Industries (<6 companies)")
            print("3. Large Industries (>12 companies)")
            print("4. Process a single specific industry")
            print("5. Exit")
            
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice in ['1', '2', '3']:
                category = {
                    '1': 'medium',
                    '2': 'small',
                    '3': 'large'
                }[choice]
                
                industries = industry_groups[category]['industries']
                for industry in industries:
                    if industry in cache.processed_industries:
                        logger.info(f"Skipping {industry} - already processed")
                        continue

                    cache.start_industry(industry)
                    industry_start_time = time.time()
                    
                    try:
                        process_industry(sp500_importer, industry, processor, sp500_df)
                        
                        industry_time = time.time() - industry_start_time
                        cache.mark_industry_complete(industry, industry_time)
                        logger.info(f"Completed {industry} in {industry_time:.2f} seconds")

                    except Exception as e:
                        logger.error(f"Error processing {industry}: {str(e)}")
                        cache.save()
                        continue

                    if industry != industries[-1]:
                        logger.info("Waiting 60 seconds before next industry...")
                        time.sleep(60)
                        
            elif choice == '4':
                all_industries = []
                for cat in ['small', 'medium', 'large']:
                    all_industries.extend(industry_groups[cat]['industries'])
                
                print("\nAvailable industries:")
                for i, ind in enumerate(all_industries, 1):
                    print(f"{i}. {ind}")
                
                try:
                    industry_index = int(input("\nSelect industry number: ")) - 1
                    if 0 <= industry_index < len(all_industries):
                        industry = all_industries[industry_index]
                        cache.start_industry(industry)
                        industry_start_time = time.time()
                        
                        process_industry(sp500_importer, industry, processor, sp500_df)
                        
                        industry_time = time.time() - industry_start_time
                        cache.mark_industry_complete(industry, industry_time)
                        logger.info(f"Completed {industry} in {industry_time:.2f} seconds")
                    else:
                        logger.error("Invalid industry selection.")
                except ValueError:
                    logger.error("Please enter a valid number.")
                    
            elif choice == '5':
                logger.info("Exiting industry processing")
                break
            else:
                logger.error("Invalid choice. Please try again.")

    total_time = time.time() - total_start_time
    logger.info(f"\nAll industries processed in {total_time:.2f} seconds")
    logger.info("\nIndustry processing times:")
    for ind, proc_time in cache.industry_times.items():
        logger.info(f"{ind}: {proc_time:.2f} seconds")

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Function for all theoretical multiple formula
def get_metric_formula(metric_name):
    
    formulas = {
        'EV/EBIT_Theoretical': 'C1*Return On Invested Capital + C2*3Y Rev Growth + y-intercept',
        'EV/After Tax EBIT_Theoretical': 'C1*Return On Invested Capital + C2*After Tax Operating Margin + y-intercept',
        'EV/EBITDA_Theoretical': 'C1*Return On Invested Capital + C2*DA + y-intercept',
        'EV/Sales_Theoretical': 'C1*After Tax Operating Margin + C2*3Y Rev Growth + y-intercept',
        'EV/Capital Invested_Theoretical': 'C1*Return On Invested Capital + C2*3Y Exp Growth + y-intercept'
    }
    return formulas.get(metric_name)

#------------------------------------------------------------------------------------------------------------------------------------------------------

# Main function to run the program
def main(run_automated=False):
    try:
        # Set working directory using absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.dirname(script_dir))

        # Initialize the database
        if not initialize_database():
            logger.error("Failed to initialize database. Exiting.")
            return
        
        # Import S&P 500 data
        sp500_importer = SP500DataImporter()
        sp500_df = sp500_importer.load_data()
        
        # Display some basic information
        logger.info(f"Loaded {len(sp500_df)} companies from S&P 500 data")
        
        # Get categorised groups of industries
        industry_groups = get_categorized_industries(sp500_df, automated=run_automated)
        
        # Load API key
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            logger.error("No API key found. Please set ALPHA_VANTAGE_API_KEY environment variable.")
            return
        
        # Initialize the financial data processor
        processor = FinancialDataProcessor(api_key)
        
        # Process industries
        process_industries(sp500_importer, industry_groups, processor, sp500_df, automated=run_automated)
            
        logger.info("Financial data processing complete")
        
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}", exc_info=True)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process S&P 500 financial data')
    parser.add_argument('--automated', action='store_true', help='Run in automated mode')
    args = parser.parse_args()
    main(run_automated=args.automated)