"""
📊 Script: 01_data_ingestion_to_rds.py

This script performs the following:
1. Connects to an AWS RDS MySQL instance
2. Creates a database if it does not already exist
3. Reads an Excel file containing historical motor insurance claim data
4. Defines explicit SQL column types for data integrity
5. Uploads the data into a specified MySQL table

Intended for one-time or batch ingestion of clean insurance data into cloud database for ML analysis.
"""

# ---------------------------- #
# 📦 Required Libraries        #
# ---------------------------- #
import pandas as pd                    # Handle tabular data and DataFrames
from sqlalchemy import create_engine   # Create connection engine to SQL databases
from sqlalchemy import text            # Use raw SQL queries with SQLAlchemy safely
from sqlalchemy.exc import SQLAlchemyError  # Handle SQL-related exceptions robustly
from sqlalchemy.types import (         # Define data types explicitly for SQL columns
    VARCHAR, INTEGER, FLOAT, TEXT, Boolean)

# ----------------------------------------
# 🔐 Configuration & Credentials
# ----------------------------------------

# ✅ MySQL login credentials
db_user: str = "admin"  # MySQL admin username
db_pass: str = "mExmuk-kitqim-jodza9"  # MySQL password (secure this in production)
db_host: str = "suncorp.ct2ykcc82vni.ap-southeast-2.rds.amazonaws.com"  # RDS endpoint
db_port: str = "3306"  # Default MySQL port
db_name: str = "Suncorp"  # Target database name to create/use

# 📄 Excel file path and target table name
excel_path: str = "data/motor_claims_fasttrack.xlsx"  # Local Excel file with motor claims
table_name: str = "motor_claims_fasttrack"  # Destination MySQL table

# ----------------------------------------
# 🚀 Main Script Block
# ----------------------------------------

try:
    # Step 1️⃣: Connect to MySQL server without selecting a specific database yet
    print("🔗 Connecting to MySQL server...")
    engine = create_engine(f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}:{db_port}/")

    # Step 2️⃣: Create the database if it doesn't already exist
    """
    Ensures the target MySQL database is present before proceeding.
    Required only for the first run or on new RDS setup.
    """
    with engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {db_name};"))
        print(f"✅ Database '{db_name}' is ready.")

    # Step 3️⃣: Reconnect to use the newly created or existing database
    engine = create_engine(f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}")

    # Step 4️⃣: Load the Excel file into a pandas DataFrame
    """
    Reads the claims data from a structured Excel spreadsheet into memory.
    """
    print(f"📄 Loading Excel file: {excel_path}")
    df = pd.read_excel(excel_path)
    print(f"✅ Excel loaded successfully. Rows: {len(df)}, Columns: {len(df.columns)}")

    # Step 5️⃣: Define explicit MySQL-compatible SQL data types for each column
    """
    This mapping ensures accurate table creation with intended column formats.
    """
    sql_dtypes = {
        'claim_id': VARCHAR(50),                           # Unique claim ID
        'vehicle_make': VARCHAR(50),                       # Car make (e.g. Toyota)
        'vehicle_model': VARCHAR(50),                      # Car model (e.g. Corolla)
        'vehicle_year': INTEGER,                           # Year of manufacture
        'vehicle_mileage': INTEGER,                        # Mileage in kilometers
        'accident_location_type': VARCHAR(20),             # Location type: Urban/Rural/Highway
        'damage_level_reported': VARCHAR(20),              # Severity: Minor/Moderate/Severe
        'customer_tenure': FLOAT,                          # Duration as a customer (in years)
        'historical_claims_count': INTEGER,                # Previous claim count
        'garage_estimate_provided': Boolean,               # Whether a garage estimate was submitted
        'days_between_accident_and_claim': INTEGER,        # Delay between incident and claim filing
        'is_fast_tracked': Boolean,                        # Target label: 1 = yes, 0 = no
        'damage_description': TEXT                         # Free text field with description of damage
    }

    # Step 6️⃣: Upload the DataFrame into MySQL with defined schema
    """
    Writes the data to the specified MySQL table using `to_sql`.
    Replaces table if it already exists. Ensures type-safety with `dtype=`.
    """
    df.to_sql(table_name, con=engine, index=False, if_exists='replace', dtype=sql_dtypes)
    print(f"✅ Table '{table_name}' uploaded to database '{db_name}' with schema!")

# ----------------------------------------
# ❌ Error Handling
# ----------------------------------------

except SQLAlchemyError as e:
    print("❌ SQLAlchemy error:", str(e))  # Common for login, engine, or syntax errors
except FileNotFoundError:
    print(f"❌ Excel file not found at path: {excel_path}")  # Bad Excel path
except Exception as e:
    print("❌ Unexpected error:", str(e))  # Catch-all for any unhandled issues