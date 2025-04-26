import pandas as pd
import requests
import json
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
import os
from dotenv import load_dotenv
from contextlib import contextmanager
import time

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("financial_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database configuration from environment variables
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# API configuration
API_URL = "https://www.alphavantage.co/query"

# Constants
REPORT_TYPES = {
    "BALANCE_SHEET": "Balance Sheet",
    "INCOME_STATEMENT": "Income Statement",
    "CASH_FLOW": "Cash Flow"
}

#------------------------------------------------------------------------------------------------------------------------------------------------------

@contextmanager
def get_db_connection(dbname=None):
    """
    Context manager for database connections.
    
    Args:
        dbname (str, optional): Database name. If None, connects to default PostgreSQL server.
    
    Yields:
        connection: PostgreSQL connection object
    """
    conn_params = DB_CONFIG.copy()
    if dbname is None:
        conn_params.pop("dbname", None)
    elif dbname != DB_CONFIG["dbname"]:
        conn_params["dbname"] = dbname
    
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

def execute_query(query, params=None, fetch=False, many=False, dbname=None):
    """
    Execute a database query with proper error handling.
    
    Args:
        query (str): SQL query to execute
        params (tuple or list, optional): Parameters for the query
        fetch (bool, optional): Whether to fetch results
        many (bool, optional): Whether to execute many statements
        dbname (str, optional): Database name
        
    Returns:
        list or None: Query results if fetch is True, otherwise None
    """
    with get_db_connection(dbname) as conn:
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

def create_database():
    """Create the financial database if it doesn't exist."""
    try:
        # Connect to PostgreSQL server (without specifying a database)
        with get_db_connection(dbname="postgres") as conn:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cursor:
                # Check if database exists
                cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_CONFIG["dbname"],))
                exists = cursor.fetchone()
                
                if not exists:
                    cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_CONFIG["dbname"])))
                    logger.info(f"Database {DB_CONFIG['dbname']} created successfully")
                else:
                    logger.info(f"Database {DB_CONFIG['dbname']} already exists")
    except Exception as e:
        logger.error(f"Failed to create database: {str(e)}")
        raise

#------------------------------------------------------------------------------------------------------------------------------------------------------

def create_tables():
    """Create all necessary tables for the financial database."""
    try:
        # SQL statements for table creation
        tables = {
            "Companies": """
                CREATE TABLE IF NOT EXISTS Companies (
                    company_id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) UNIQUE NOT NULL,
                    name VARCHAR(255),
                    sector VARCHAR(100),
                    industry VARCHAR(100),
                    unlevered_data FLOAT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "Industries": """
                CREATE TABLE IF NOT EXISTS Industries (
                    industry_id SERIAL PRIMARY KEY,
                    sector_name VARCHAR(100) NOT NULL,
                    industry_name VARCHAR(100) NOT NULL,
                    cost_of_capital DECIMAL(7,4),
                    growth_rate DECIMAL(7,4),
                    reinvestment_rate DECIMAL(7,4),
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT unique_industry_sector UNIQUE (sector_name, industry_name)
                )
            """,
            "ReportingPeriods": """
                CREATE TABLE IF NOT EXISTS ReportingPeriods (
                    period_id SERIAL PRIMARY KEY,
                    period_type VARCHAR(20) NOT NULL,
                    fiscal_date_ending DATE NOT NULL,
                    year INTEGER NOT NULL,
                    quarter INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (period_type, fiscal_date_ending)
                )
            """,
            "FinancialReports": """
                CREATE TABLE IF NOT EXISTS FinancialReports (
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
            "FinancialMetrics": """
                CREATE TABLE IF NOT EXISTS FinancialMetrics (
                    metric_id SERIAL PRIMARY KEY,
                    metric_name VARCHAR(100) UNIQUE NOT NULL,
                    display_name VARCHAR(100),
                    category VARCHAR(50),
                    data_type VARCHAR(20) DEFAULT 'currency',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            "FinancialData": """
                CREATE TABLE IF NOT EXISTS FinancialData (
                    data_id SERIAL PRIMARY KEY,
                    report_id INTEGER REFERENCES FinancialReports(report_id),
                    metric_id INTEGER REFERENCES FinancialMetrics(metric_id),
                    value NUMERIC,
                    is_null BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (report_id, metric_id)
                )
            """,
            "CalculatedMetrics": """
                CREATE TABLE IF NOT EXISTS CalculatedMetrics (
                    calc_metric_id SERIAL PRIMARY KEY,
                    company_id INTEGER REFERENCES Companies(company_id),
                    period_id INTEGER REFERENCES ReportingPeriods(period_id),
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value DECIMAL(19,4) NOT NULL,
                    calculation_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    data_source VARCHAR(50),
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT unique_calc_metric UNIQUE (company_id, period_id, metric_name)
                )
            """,
            "CalculatedMetricDefinitions": """
                CREATE TABLE IF NOT EXISTS CalculatedMetricDefinitions (
                    definition_id SERIAL PRIMARY KEY,
                    metric_name VARCHAR(100) UNIQUE NOT NULL,
                    formula TEXT NOT NULL,
                    description TEXT,
                    units VARCHAR(50),
                    is_percentage BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        }
        
        # Indexes to create
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_financial_data_report_id ON FinancialData(report_id)",
            "CREATE INDEX IF NOT EXISTS idx_financial_reports_company_id ON FinancialReports(company_id)",
            "CREATE INDEX IF NOT EXISTS idx_financial_reports_period_id ON FinancialReports(period_id)",
            "CREATE INDEX IF NOT EXISTS idx_companies_sector ON Companies(sector)",
            "CREATE INDEX IF NOT EXISTS idx_companies_industry ON Companies(industry)",
            "CREATE INDEX IF NOT EXISTS idx_industries_sector ON Industries(sector_name)",
            "CREATE INDEX IF NOT EXISTS idx_calculated_metrics_company ON CalculatedMetrics(company_id)",
            "CREATE INDEX IF NOT EXISTS idx_calculated_metrics_period ON CalculatedMetrics(period_id)",
            "CREATE INDEX IF NOT EXISTS idx_calculated_metrics_name ON CalculatedMetrics(metric_name)"
        ]
        
        # Execute all table creation statements
        for name, query in tables.items():
            execute_query(query)
            logger.debug(f"Created table: {name}")
        
        # Create all indexes
        for index_query in indexes:
            execute_query(index_query)
            
        logger.info("All tables and indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        raise

#------------------------------------------------------------------------------------------------------------------------------------------------------

class FinancialDataAccess:
    """Class to handle all database operations related to financial data."""
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    def get_industry(self, sector_name):
        """
        Retrieve industry-level financial data for a given sector_name from the database.
        
        Args:
            sector_name (str): Name of the sector (as used in CSV)
        
        Returns:
            dict or None: A dictionary with keys cost_of_capital, growth_rate, reinvestment_rate,
                          or None if not found.
        """
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
    
    @staticmethod
    def insert_company(symbol, name=None, sector=None, industry=None, description=None, unlevered_data=0.0):
        """
        Insert a new company or update if it exists.
        Returns the company_id.
        """
        query = """
        INSERT INTO Companies (symbol, name, sector, industry, description, unlevered_data)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol) 
        DO UPDATE SET 
            name = COALESCE(%s, Companies.name),
            sector = COALESCE(%s, Companies.sector),
            industry = COALESCE(%s, Companies.industry),
            description = COALESCE(%s, Companies.description),
            unlevered_data = COALESCE(%s, Companies.unlevered_data),
            updated_at = CURRENT_TIMESTAMP
        RETURNING company_id
        """
        params = (symbol, name, sector, industry, description, unlevered_data, 
                  name, sector, industry, description, unlevered_data)
        
        result = execute_query(query, params)
        return result[0] if result else None
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def insert_reporting_period(fiscal_date_ending, period_type):
        """
        Insert a new reporting period or return existing one.
        Returns the period_id.
        """
        # Extract year and quarter if applicable
        year = int(fiscal_date_ending.split('-')[0])
        quarter = None
        if period_type == 'Quarterly':
            month = int(fiscal_date_ending.split('-')[1])
            quarter = (month - 1) // 3 + 1
        
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
    
    @staticmethod
    def insert_financial_report(company_id, period_id, report_type, reported_currency=None):
        """
        Insert a new financial report or update if it exists.
        Returns the report_id.
        """
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
    
    @staticmethod
    def get_or_create_metric(metric_name, display_name=None, category=None, data_type='currency'):
        """
        Get an existing metric or create a new one.
        Returns the metric_id.
        """
        # If display_name is not provided, create one from the metric_name
        if not display_name:
            display_name = ' '.join(word.capitalize() for word in metric_name.split('_'))
        
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
    
    @staticmethod
    def insert_financial_data(report_id, metric_id, value):
        """
        Insert financial data or update if it exists.
        """
        is_null = (value == "None" or value is None)
        numeric_value = None if is_null else value
        
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
    
    @staticmethod
    def insert_industry(sector_name, industry_name, cost_of_capital=None, growth_rate=None, reinvestment_rate=None):
        """
        Insert a new industry or update if it exists.
        Returns the industry_id.
        """
        query = """
        INSERT INTO Industries (sector_name, industry_name, cost_of_capital, growth_rate, reinvestment_rate)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (sector_name, industry_name) 
        DO UPDATE SET 
            cost_of_capital = COALESCE(%s, Industries.cost_of_capital),
            growth_rate = COALESCE(%s, Industries.growth_rate),
            reinvestment_rate = COALESCE(%s, Industries.reinvestment_rate),
            last_updated = CURRENT_TIMESTAMP
        RETURNING industry_id
        """
        params = (sector_name, industry_name, cost_of_capital, growth_rate, reinvestment_rate,
                  cost_of_capital, growth_rate, reinvestment_rate)
        
        result = execute_query(query, params)
        return result[0] if result else None

#------------------------------------------------------------------------------------------------------------------------------------------------------

class FinancialDataProcessor:
    """Class to handle financial data processing operations."""
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    def __init__(self, api_key):
        """
        Initialize the processor with API key.
        
        Args:
            api_key (str): API key for financial data service
        """
        self.api_key = api_key
        self.db = FinancialDataAccess()
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    def fetch_financial_data(self, symbol, function):
        """
        Fetch financial data from API.
        
        Args:
            symbol (str): Company symbol
            function (str): API function name
            
        Returns:
            dict: API response data or None if error
        """
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
    
    def process_financial_data(self, symbol, data, report_type, max_quarterly_reports=12):
        """
        Process financial data from API response and store in database.
    
        Args:
            symbol (str): The company symbol (e.g., 'IBM')
            data (dict): The API response data
            report_type (str): The type of report (e.g., 'BALANCE_SHEET', 'INCOME_STATEMENT')
            max_quarterly_reports (int): Maximum number of quarterly reports to process
            """
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
                # Take only the most recent reports up to the limit
                limited_reports = sorted_reports[:max_quarterly_reports]
                logger.info(f"Processing {len(limited_reports)} of {len(sorted_reports)} available quarterly reports for {symbol}")
                self._process_reports(company_id, limited_reports, report_type, "Quarterly")
            else:
                logger.warning(f"Quarterly reports for {symbol} are missing date information")
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    def _process_reports(self, company_id, reports, report_type, period_type):
        """
        Process a list of financial reports.
        
        Args:
            company_id (int): Company ID
            reports (list): List of report dictionaries
            report_type (str): Report type
            period_type (str): Period type ('Annual' or 'Quarterly')
        """
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

class SP500DataImporter:
    """Class to handle S&P 500 data import operations."""
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    def __init__(self, csv_path="SP500.csv"):
        """
        Initialize the importer with CSV path, but don't connect to DB yet.
        
        Args:
            csv_path (str): Path to S&P 500 CSV file
        """
        self.csv_path = csv_path
        self.db = None
        self.sp500_df = None
    
    #----------------------------------------------------------------------------------------------------------------------------------    
        
    def connect_db(self):
        """
        Connect to the database only when needed.
        """
        if self.db is None:
            self.db = FinancialDataAccess()
    
    #----------------------------------------------------------------------------------------------------------------------------------
            
    def load_data(self):
        """
        Load S&P 500 data from CSV file.
        
        Returns:
            DataFrame: Pandas DataFrame with S&P 500 data
        """
        try:
            logger.info(f"Reading SP500 data from {self.csv_path}")
            self.sp500_df = pd.read_csv(self.csv_path)
            return self.sp500_df
        except Exception as e:
            logger.error(f"Failed to load S&P 500 data: {str(e)}")
            raise
            
    #----------------------------------------------------------------------------------------------------------------------------------
    
    def import_industry_data(self, sector_name):
        """
        Import data for a specific industry.
        
        Args:
            industry_name (str): Name of the industry to import
            
        Returns:
            int: Number of companies imported
        """
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
        
        # Insert/update the industry once
        self.db.insert_industry(
            sector_name=first_company['Sector'],
            industry_name=first_company.get('Industry', 'N/A'),
            cost_of_capital=first_company.get('Industry Cost of Capital'),
            growth_rate=first_company.get('Industry Growth'),
            reinvestment_rate=first_company.get('Industry Reinvestment rate')
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
    
    def is_industry_data_current(self, sector_name):
        """
        Check if the industry data in the database is current by comparing
        with a random company from that industry in the CSV.
    
        Args:
           industry_name (str): Name of the industry to check
        
        Returns:
            bool: True if data is current, False if update needed
            """
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
    
    def process_industry(self, industry_name):
        """
        Process a single industry - check if data is current and import if needed.
    
        Args:
            industry_name (str): Name of the industry to process
    
        Returns:
            bool: True if processed, False if skipped
            """
        if self.is_industry_data_current(industry_name):
            logger.info(f"Skipping import for {industry_name} - data is current")
            return False
    
        # Import industry data if not current
        company_count = self.import_industry_data(industry_name)
        logger.info(f"Imported {company_count} companies for {industry_name}")
        return True
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    def get_categorized_industries(self, medium_min=6, medium_max=12):
        """
        Categorize industries into groups based on company count and return filtered results.
    
        Args:
            medium_min (int): Minimum number of companies for medium category
            medium_max (int): Maximum number of companies for medium category
    
        Returns:
            dict: Dictionary with categories of industries and their counts
            """
        if self.sp500_df is None:
            self.load_data()
        
        industry_counts = self.sp500_df['Sector'].value_counts()
    
        # Categorize industries
        small_industries = industry_counts[industry_counts < medium_min]
        medium_industries = industry_counts[(industry_counts >= medium_min) & 
                                      (industry_counts <= medium_max)]
        large_industries = industry_counts[industry_counts > medium_max]
    
        # Log the medium industries (previously done in filter_industries)
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

def process_industries_menu(sp500_importer, industry_groups, processor, sp500_df):
    """
    Display menu and process industries based on user choice.
    
    Args:
        sp500_importer: SP500DataImporter instance
        industry_groups: Dictionary containing categorized industries
        processor: FinancialDataProcessor instance
        sp500_df: DataFrame with S&P 500 data
    """
    while True:
        # Display menu for user selection
        print("\nSelect an industry group to process:")
        print("1. Medium Industries (6-12 companies)")
        print("2. Small Industries (<6 companies)")
        print("3. Large Industries (>12 companies)")
        print("4. Process a single specific industry")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        # Process based on user choice
        if choice == '1':
            process_industry_group(sp500_importer, industry_groups['medium']['industries'], processor, sp500_df)
        elif choice == '2':
            process_industry_group(sp500_importer, industry_groups['small']['industries'], processor, sp500_df)
        elif choice == '3':
            process_industry_group(sp500_importer, industry_groups['large']['industries'], processor, sp500_df)
        elif choice == '4':
            # Display all industries for user selection
            all_industries = []
            for category in ['small', 'medium', 'large']:
                all_industries.extend(industry_groups[category]['industries'])
            
            print("\nAvailable industries:")
            for i, industry in enumerate(all_industries, 1):
                print(f"{i}. {industry}")
            
            try:
                industry_index = int(input("\nSelect industry number: ")) - 1
                if 0 <= industry_index < len(all_industries):
                    selected_industry = all_industries[industry_index]
                    process_single_industry(sp500_importer, selected_industry, processor, sp500_df)
                else:
                    logger.error("Invalid industry selection.")
            except ValueError:
                logger.error("Please enter a valid number.")
        elif choice == '5':
            logger.info("Exiting industry processing")
            break
        else:
            logger.error("Invalid choice. Please try again.")

#------------------------------------------------------------------------------------------------------------------------------------------------------

def process_industry_group(sp500_importer, industry_list, processor, sp500_df, delay_between_industries=60):
    """
    Process a group of industries with delay between each.
    
    Args:
        sp500_importer: SP500DataImporter instance
        industry_list: List of industries to process
        processor: FinancialDataProcessor instance
        sp500_df: DataFrame with S&P 500 data
        delay_between_industries: Delay in seconds between processing industries
    """
    total_industries = len(industry_list)
    
    for i, industry in enumerate(industry_list, 1):
        logger.info(f"Processing industry {i}/{total_industries}: {industry}")
        
        # Check if industry data is current
        if sp500_importer.is_industry_data_current(industry):
            logger.info(f"Industry data for {industry} is current. Skipping data import.")
        else:
            process_single_industry(sp500_importer, industry, processor, sp500_df)
        
        # Delay between industries (except after the last one)
        if i < total_industries:
            logger.info(f"Waiting {delay_between_industries} seconds before processing next industry...")
            time.sleep(delay_between_industries)
    
    logger.info(f"Finished processing {total_industries} industries")

#------------------------------------------------------------------------------------------------------------------------------------------------------

def process_single_industry(sp500_importer, industry, processor, sp500_df, max_quarterly_reports=12):
    """
    Process a single industry.
    
    Args:
        sp500_importer: SP500DataImporter instance
        industry: Industry name
        processor: FinancialDataProcessor instance
        sp500_df: DataFrame with S&P 500 data
        max_quarterly_reports: Maximum number of quarterly reports to process
    """
    # Get tickers for the selected industry
    industry_tickers = sp500_df[sp500_df['Sector'] == industry]['Symbol'].tolist()
    logger.info(f"Selected {industry} with {len(industry_tickers)} companies")
    
    # Fetch and process data for each ticker
    reports = ['BALANCE_SHEET', 'INCOME_STATEMENT']
    
    for ticker in industry_tickers:
        for report in reports:
            # Fetch data from API
            data = processor.fetch_financial_data(ticker, report)
            
            # Process the data with limited quarterly reports
            if data:
                processor.process_financial_data(ticker, data, report, max_quarterly_reports)
                
            # Add a delay to avoid API rate limits
            time.sleep(12)  # 12 seconds between API calls
    
    logger.info(f"Completed processing {industry}")

#------------------------------------------------------------------------------------------------------------------------------------------------------

def main():
    """Main program execution with automated industry processing."""
    try:
        # Set working directory
        os.chdir('Projects/Data Analysis/S&P500 Valuation Multiples')

        # Initialize the database
        if not initialize_database():
            logger.error("Failed to initialize database. Exiting.")
            return
        
        # Import S&P 500 data
        sp500_importer = SP500DataImporter()
        sp500_df = sp500_importer.load_data()
        
        # Display some basic information
        logger.info(f"Loaded {len(sp500_df)} companies from S&P 500 data")
        
        # Get categorized industries
        industry_groups = sp500_importer.get_categorized_industries()
        
        # Medium industries that have already been imported
        already_imported = ["Information Technology", "Healthcare", "Financials", "Consumer Discretionary"]
        
        # Remove already imported industries from the medium list
        for industry in already_imported:
            if industry in industry_groups['medium']['industries']:
                industry_groups['medium']['industries'].remove(industry)
                logger.info(f"Removed {industry} from medium industries as it has already been imported")
        
        # Load API key
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            logger.error("No API key found. Please set ALPHA_VANTAGE_API_KEY environment variable or create a Key.py file.")
            return
        
        # Initialize the financial data processor
        processor = FinancialDataProcessor(api_key)
        
        # Process industries based on user choice
        process_industries_menu(sp500_importer, industry_groups, processor, sp500_df)
            
        logger.info("Financial data processing complete")
        
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}", exc_info=True)

#------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()