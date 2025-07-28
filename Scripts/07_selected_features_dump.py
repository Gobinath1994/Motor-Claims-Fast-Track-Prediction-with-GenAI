"""
📁 Script: 07_selected_features_dump.py

🔍 Purpose:
    This script connects to the AWS RDS MySQL database, loads the feature-engineered dataset,
    extracts the final selected input features (excluding the target variable), and saves them
    to disk as a Pickle file. This list is essential for ensuring that downstream model
    inference or real-time prediction pipelines apply the same column order and set.

✅ Key Functions:
    - Connect to AWS RDS using SQLAlchemy
    - Read selected features table from the database
    - Exclude the binary target column (`is_fast_tracked`)
    - Save the feature column names to `models/selected_features.pkl` using joblib

🔐 Security Note:
    Ensure the RDS credentials and connection string are securely stored in production
    (e.g., use environment variables or a secrets manager instead of hardcoding them).

"""

# -------------------------- #
# 📦 Required Libraries      #
# -------------------------- #
import os                            # Provides functions to interact with the operating system (e.g., file/folder operations)
import pandas as pd                  # Used to handle tabular data and read from SQL
import joblib                        # Used for saving Python objects (e.g., list) to .pkl
import sqlalchemy                    # SQLAlchemy engine to connect to AWS RDS MySQL

# Ensure the 'models' directory exists for saving plots and visual outputs.
# If it already exists, do nothing (thanks to exist_ok=True)
os.makedirs("models", exist_ok=True)

# -------------------------- #
# 🔌 Connect to AWS RDS      #
# -------------------------- #

# RDS connection string in SQLAlchemy format: dialect+driver://username:password@host/db
rds_uri = (
    "mysql+pymysql://admin:mExmuk-kitqim-jodza9"
    "@suncorp.ct2ykcc82vni.ap-southeast-2.rds.amazonaws.com/Suncorp"
)

# Create SQLAlchemy engine which will be used to execute SQL commands
engine = sqlalchemy.create_engine(rds_uri)

# -------------------------- #
# 📥 Load Feature Table      #
# -------------------------- #

# Read the full contents of the feature-selected dataset
# Table contains final engineered features used in modeling
df = pd.read_sql("SELECT * FROM motor_claims_selected_features", engine)

# -------------------------- #
# ✂️ Extract Input Features  #
# -------------------------- #

# The target column in this use case is 'is_fast_tracked' (1 = fast-tracked claim, 0 = not)
# We remove it to isolate only the independent features for model input
selected_features = [col for col in df.columns if col != 'is_fast_tracked']

# -------------------------- #
# 💾 Save Features to Disk   #
# -------------------------- #

# Save the list of selected features to a Pickle file
# This ensures future inference uses the same column order and set
joblib.dump(selected_features, "models/selected_features.pkl")

# Print a confirmation message with feature count
print(f"✅ Saved {len(selected_features)} selected features to models/selected_features.pkl")