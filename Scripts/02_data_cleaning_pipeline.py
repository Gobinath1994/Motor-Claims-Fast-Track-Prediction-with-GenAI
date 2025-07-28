"""
🧹 Script: 02_data_cleaning_pipeline.py

This script performs end-to-end data cleaning for motor insurance claims.
It reads raw data from AWS RDS (MySQL), processes and cleans it, flags outliers,
and saves the cleaned dataset back to the RDS for downstream ML/EDA tasks.

Includes handling of:
- Missing values
- Outliers (Isolation Forest)
- Short/incomplete text
- Duplicate rows
- Schema normalization

"""

# ---------------------------- #
# 📦 Required Libraries        #
# ---------------------------- #
import pandas as pd                                # Handle tabular data and DataFrame operations
import numpy as np                                 # Support numerical computations and NaN operations
from sqlalchemy import create_engine               # Create a database engine for AWS RDS connection
from sklearn.ensemble import IsolationForest       # Detect outliers using Isolation Forest algorithm

# -------------------------------------------
# STEP 1: Connect to AWS RDS
# -------------------------------------------

# RDS credentials and database/table config
db_user = "admin"  # RDS MySQL username
db_pass = "mExmuk-kitqim-jodza9"  # RDS password
db_host = "suncorp.ct2ykcc82vni.ap-southeast-2.rds.amazonaws.com"  # RDS endpoint
db_port = "3306"  # MySQL port
db_name = "Suncorp"  # Database name
raw_table = "motor_claims_fasttrack"  # Table with raw data
cleaned_table = "motor_claims_cleaned"  # Output table name for cleaned data

# Create SQLAlchemy engine for MySQL connection
engine = create_engine(
    f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
)
print("✅ Connected to AWS RDS")

# -------------------------------------------
# STEP 2: Load Raw Data from MySQL
# -------------------------------------------

# Read raw data from the RDS table into a DataFrame
df_raw = pd.read_sql(f"SELECT * FROM {raw_table}", con=engine)
print(f"\n📥 Loaded raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
print("\n🧾 Raw data info:")
print(df_raw.info())
print(df_raw.isnull().sum())

# -------------------------------------------
# STEP 3: Initialize Clean Copy
# -------------------------------------------

# Create working copy of the data
df = df_raw.copy()

# Normalize column names: lowercase, remove spaces
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# -------------------------------------------
# STEP 4: Clean Text Column: damage_description
# -------------------------------------------

# Convert to string, strip whitespace, and lowercase for NLP compatibility
df["damage_description"] = df["damage_description"].astype(str).str.strip().str.lower()

# Drop rows with empty or very short damage descriptions
before_text = df.shape[0]
df = df[df["damage_description"].str.len() > 10]
print(f"🧹 Dropped {before_text - df.shape[0]} rows with empty or short descriptions")

# -------------------------------------------
# STEP 5: Handle Missing Values
# -------------------------------------------

# Fill missing mileage and tenure with zero (neutral value)
df["vehicle_mileage"] = df["vehicle_mileage"].fillna(0)
df["customer_tenure"] = df["customer_tenure"].fillna(0)

# Replace common invalid strings with actual NaN values for object columns
invalid_vals = ["n/a", "na", "none", "null", "unknown", ""]
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].replace(invalid_vals, np.nan)

# -------------------------------------------
# STEP 6: Clip Extreme Numerical Values
# -------------------------------------------

# Cap mileage at 300,000 km and claim delay at 60 days
df["vehicle_mileage"] = df["vehicle_mileage"].clip(upper=300000)
df["days_between_accident_and_claim"] = df["days_between_accident_and_claim"].clip(upper=60)

# -------------------------------------------
# STEP 7: Fix Data Types
# -------------------------------------------

# Convert binary columns to appropriate types
df["garage_estimate_provided"] = df["garage_estimate_provided"].astype(bool)
df["is_fast_tracked"] = df["is_fast_tracked"].astype(int)

# -------------------------------------------
# STEP 8: Convert Categorical Columns
# -------------------------------------------

# Convert common category columns to categorical dtype to save memory and aid modeling
cat_cols = ["vehicle_make", "vehicle_model", "accident_location_type", "damage_level_reported"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")

# -------------------------------------------
# STEP 9: Remove Duplicate Records
# -------------------------------------------

# Drop exact duplicate rows
before_dups = df.shape[0]
df = df.drop_duplicates()
removed_dups = before_dups - df.shape[0]
print(f"🧹 Removed {removed_dups} duplicate rows")

# -------------------------------------------
# STEP 10: Outlier Detection using Isolation Forest
# -------------------------------------------

# Numerical columns to check for anomaly detection
numericals = [
    "vehicle_year", "vehicle_mileage", "customer_tenure",
    "historical_claims_count", "days_between_accident_and_claim"
]

# Apply Isolation Forest to detect outliers (1% contamination)
iso = IsolationForest(contamination=0.01, random_state=42)
df["outlier_flag"] = iso.fit_predict(df[numericals])

# Convert prediction result to binary: 1 = outlier, 0 = normal
df["outlier_flag"] = df["outlier_flag"].map({1: 0, -1: 1})
print(f"🚨 Outliers flagged: {df['outlier_flag'].sum()} rows")

# Optional: Filter outliers if desired
# df = df[df["outlier_flag"] == 0]

# -------------------------------------------
# STEP 11: Cleaning Summary Output
# -------------------------------------------

# Summary of cleaning impact
print("\n✅ Final Cleaning Summary:")
print(f"🔹 Rows before cleaning: {df_raw.shape[0]}")
print(f"🔹 Rows after cleaning: {df.shape[0]}")
print(f"🔹 Duplicate rows removed: {removed_dups}")
print(f"🔹 Short/empty text removed: {before_text - df.shape[0]}")
print(f"🔹 Remaining nulls:\n{df.isnull().sum()}")

# Sample comparison rows for validation
print("\n🔍 Sample raw rows:")
print(df_raw[["damage_description", "vehicle_mileage"]].sample(3))

print("\n🔍 Sample cleaned rows:")
print(df[["damage_description", "vehicle_mileage", "outlier_flag"]].sample(3))

# -------------------------------------------
# STEP 12: Save Cleaned Data to RDS
# -------------------------------------------

# Upload cleaned dataset to RDS as a new table
df.to_sql(cleaned_table, con=engine, index=False, if_exists="replace")
print(f"\n💾 Cleaned data written to RDS table: `{cleaned_table}`")