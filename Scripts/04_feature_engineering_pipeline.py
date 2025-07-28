# -----------------------------------
# Motor Claims Feature Engineering Script
# -----------------------------------
# This script performs advanced feature engineering and NLP transformation
# on motor insurance claims data stored in an AWS RDS MySQL database.
# It also applies SMOTE oversampling to handle class imbalance and
# saves the resulting feature-enriched dataset back to RDS.

# ------------------------------ #
# 📦 Required Libraries          #
# ------------------------------ #
import os      # 🗂️ Provides functions to interact with the operating system (e.g., creating directories)
import pandas as pd                                # Handle tabular data and DataFrame operations
import numpy as np                                 # Numerical computations and array manipulations
import re                                          # Regular expressions for pattern matching in text
import sqlalchemy                                  # Connect and interact with AWS RDS using SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text to TF-IDF vectors
from sklearn.decomposition import PCA              # Dimensionality reduction using Principal Component Analysis
from sklearn.cluster import KMeans                 # Group text embeddings into clusters
from sklearn.impute import SimpleImputer           # Impute missing values using statistical strategies
from sentence_transformers import SentenceTransformer  # Generate dense sentence embeddings (MiniLM etc.)
from imblearn.over_sampling import SMOTE           # Synthetic Minority Over-sampling Technique for class balance
from collections import Counter                    # Count frequencies of categorical class labels
import joblib  # 📦 Used for saving and loading Python objects like models or transformers (e.g., TF-IDF, PCA)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# -----------------------------------
# 📂 Load Data from RDS
# -----------------------------------

def load_data():
    """
    Loads the cleaned motor claims dataset from an AWS RDS MySQL instance.

    Returns:
        df (pd.DataFrame): The cleaned motor insurance claims data.
        engine (sqlalchemy.engine.Engine): SQLAlchemy engine object to allow reuse of the DB connection.
    """

    # 🔗 Define the RDS URI (including username, password, host, and DB name)
    # Format: mysql+pymysql://<username>:<password>@<host>/<database>
    rds_uri = (
        "mysql+pymysql://admin:mExmuk-kitqim-jodza9@"
        "suncorp.ct2ykcc82vni.ap-southeast-2.rds.amazonaws.com/Suncorp"
    )

    # 🏗️ Create SQLAlchemy engine to establish and manage DB connection
    engine = sqlalchemy.create_engine(rds_uri)

    # 🧾 Query the table `motor_claims_cleaned` from the RDS database into a Pandas DataFrame
    df = pd.read_sql("SELECT * FROM motor_claims_cleaned", engine)

    # 🔁 Return both the loaded DataFrame and engine object for future use (e.g., saving processed data)
    return df, engine

# -----------------------------------
# ⚖️ Class Distribution Logger
# -----------------------------------

def log_class_distribution(df):
    """
    Logs the class distribution of the target variable 'is_fast_tracked'.
    
    Purpose:
        - Understands how imbalanced the target labels are.
        - Helps guide modeling decisions such as resampling (e.g., SMOTE) or choosing evaluation metrics.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the target column `is_fast_tracked`.

    Returns:
        None — prints value counts and proportions directly to console/log.
    """

    # 👀 Visual header for readability in logs or notebooks
    print("\n⚖️ Target Class Distribution:")

    # 📊 Print proportion of each class (i.e., class 0 and class 1) as percentages
    # normalize=True gives relative frequencies, rename() assigns column label
    print(df["is_fast_tracked"].value_counts(normalize=True).rename("Proportion"))

    # 🔢 Print absolute counts of each class (how many samples per label)
    print(df["is_fast_tracked"].value_counts().rename("Counts"))

# -----------------------------------
# 🎯 Feature Engineering: Vehicle Attributes
# -----------------------------------

def engineer_vehicle_features(df):
    """
    Creates domain-based features using vehicle-related fields.

    Features created:
        - vehicle_age: Age of the vehicle at claim time.
        - mileage_per_year: Average usage intensity based on age.
        - high_mileage_flag: Binary indicator of excessive mileage (>150,000 km).
        - log_mileage: Log-transformed mileage to normalize skewness.

    Parameters:
        df (pd.DataFrame): Input DataFrame with raw vehicle fields.

    Returns:
        df (pd.DataFrame): Updated DataFrame with engineered vehicle features.
    """

    # ⏳ Vehicle Age: Subtracts the manufacturing year from current year (2025 assumed)
    df["vehicle_age"] = 2025 - df["vehicle_year"]

    # 🛣️ Mileage Per Year: Divides total mileage by age to estimate yearly usage
    # Replaces age=0 with 1 to avoid division by zero
    df["mileage_per_year"] = df["vehicle_mileage"] / df["vehicle_age"].replace(0, 1)

    # 🚨 High Mileage Flag: Binary feature for vehicles with mileage over 150,000 km
    df["high_mileage_flag"] = (df["vehicle_mileage"] > 150000).astype(int)

    # 📉 Log Mileage: Applies log(1 + x) transform to compress outliers and normalize distribution
    df["log_mileage"] = np.log1p(df["vehicle_mileage"])

    return df  # 🔁 Returns the updated DataFrame with added features

# -----------------------------------
# 👥 Feature Engineering: Customer Attributes
# -----------------------------------

def engineer_customer_features(df):
    """
    Constructs derived features from customer-related variables that may signal
    risk level, loyalty, or behavioral tendencies in the claim process.

    Features Created:
        - prior_claim_rate: Frequency of prior claims per tenure year.
        - is_high_claim_history: Binary flag for customers with >1 prior claim.
        - tenure_group: Categorical banding of customer tenure.
        - is_loyal_customer: Flag for customers with tenure > 7 years.
        - log_tenure: Log-transformed tenure to reduce skewness.

    Parameters:
        df (pd.DataFrame): Input DataFrame with customer-related columns.

    Returns:
        df (pd.DataFrame): Updated DataFrame with engineered customer features.
    """

    # 📈 Prior Claim Rate: Normalizes historical claim count by number of years with the company
    # Uses replace(0,1) to avoid division by zero if tenure is 0
    df["prior_claim_rate"] = df["historical_claims_count"] / df["customer_tenure"].replace(0, 1)

    # 🚨 High Claim History Flag: Flags customers with more than 1 prior claim as higher risk
    df["is_high_claim_history"] = (df["historical_claims_count"] > 1).astype(int)

    # 🧮 Tenure Grouping: Buckets customer tenure into labeled bands for categorical modeling
    df["tenure_group"] = pd.cut(
        df["customer_tenure"], 
        bins=[0, 2, 5, 10, 20],
        labels=["<2y", "2-5y", "5-10y", "10+y"]
    )

    # 💎 Loyalty Flag: Customers with tenure > 7 years are considered loyal
    df["is_loyal_customer"] = (df["customer_tenure"] > 7).astype(int)

    # 📉 Log Tenure: Applies log(1 + x) transformation to normalize distribution for modeling
    df["log_tenure"] = np.log1p(df["customer_tenure"])

    return df  # 🔁 Returns the enriched DataFrame with new customer-based features

# -----------------------------------
# 📝 Text Feature Engineering from Descriptions
# -----------------------------------

def engineer_text_features(df):
    """
    Generates interpretable NLP-derived features from the 'damage_description' field.
    
    Features Created:
        - desc_length: Word count of each description.
        - low_effort_text: Binary flag for short/insufficient descriptions (<12 words).
        - has_safety_keywords: Presence of safety-critical or high-severity terms.
        - mentions_minor: Mentions of minor/non-severe damage indicators.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing a 'damage_description' column.

    Returns:
        df (pd.DataFrame): Updated DataFrame with engineered text features.
    """

    # 🧮 Description Length: Number of words in the damage description
    df["desc_length"] = df["damage_description"].str.split().apply(len)

    # ⚠️ Low-Effort Text: Flag descriptions with <12 words as potentially vague or low quality
    df["low_effort_text"] = (df["desc_length"] < 12).astype(int)

    # 🔍 Define keyword groups for binary flagging (can be expanded with domain knowledge)
    keywords = {
        "has_safety_keywords": ["fluid", "leaks", "alignment", "safety", "confirmed", "concern"],
        "mentions_minor": ["minor", "scuff", "surface", "bending"]
    }

    # 🏷️ Create binary columns based on keyword presence using regex word-boundary search
    for colname, terms in keywords.items():
        df[colname] = df["damage_description"].apply(
            lambda x: int(  # Convert boolean to int (1 if any term found, else 0)
                any(       # Check if any keyword in the list is present
                    re.search(rf"\b{term}\b", str(x).lower())  # Case-insensitive word match
                    for term in terms
                )
            )
        )

    return df  # 🔁 Return DataFrame enriched with text flags and description metrics

# -----------------------------------
# 🧠 Add NLP Features (TF-IDF, Embeddings, PCA, Clustering)
# -----------------------------------

# -----------------------------------
# 🧠 Add NLP Features (TF-IDF, Embeddings, PCA, Clustering)
# -----------------------------------

import joblib
import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

def add_nlp_features(df):
    """
    Performs NLP transformation of the 'damage_description' column:
    1. TF-IDF vectorization (Top 30 terms)
    2. SentenceTransformer embeddings (MiniLM)
    3. PCA reduction to top 5 dimensions
    4. KMeans clustering on embeddings to derive semantic cluster labels
    Also saves the fitted models to disk for future inference.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'damage_description'.

    Returns:
        df (pd.DataFrame): Updated with TF-IDF, embedding, PCA, and cluster features.
    """
    # 1️⃣ TF-IDF Vectorizer: Extract top 30 important words
    tfidf = TfidfVectorizer(stop_words="english", max_features=30)
    tfidf_matrix = tfidf.fit_transform(df["damage_description"])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{t}" for t in tfidf.get_feature_names_out()])
    df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    # 🔐 Save the TF-IDF vectorizer
    joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")

    # 2️⃣ Sentence Embeddings using MiniLM
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["damage_description"].tolist(), show_progress_bar=True)
    embeddings_df = pd.DataFrame(embeddings, columns=[f"text_emb_{i}" for i in range(embeddings.shape[1])])
    df = pd.concat([df.reset_index(drop=True), embeddings_df], axis=1)

    # 3️⃣ PCA on embeddings (reduce to 10 dimensions)
    pca = PCA(n_components=10)
    pca_comps = pca.fit_transform(embeddings)
    for i in range(10):
        df[f"text_pca_{i+1}"] = pca_comps[:, i]

    # 🔐 Save PCA transformer
    joblib.dump(pca, "models/pca_transformer.joblib")

    # 4️⃣ KMeans Clustering (semantic clustering)
    km = KMeans(n_clusters=5, random_state=42)
    df["text_cluster"] = km.fit_predict(embeddings)

    # 🔐 Save KMeans model
    joblib.dump(km, "models/kmeans_cluster.joblib")

    return df

# -----------------------------------
# 🔧 Business Logic Driven Features
# -----------------------------------

def engineer_domain_logic(df):
    """
    Add domain-specific interaction features and proxy scores:
    - Damage severity, estimate flags, cross features
    - Delay categories and manual binning
    - Risk proxy using rules from business context

    Parameters:
        df (pd.DataFrame): DataFrame with engineered base features
    Returns:
        df (pd.DataFrame): Updated DataFrame with domain logic features
    """
    df["is_severe_damage"] = (df["damage_level_reported"] == "Severe").astype(int)
    df["is_garage_provided"] = df["garage_estimate_provided"].astype(int)
    df["est_provided_x_severity"] = df["is_garage_provided"] * df["is_severe_damage"]

    df["vehicle_age_x_mileage"] = df["vehicle_age"] * df["mileage_per_year"]
    df["claim_delay_x_severity"] = df["days_between_accident_and_claim"] * df["is_severe_damage"]
    df["tenure_x_claim_rate"] = df["customer_tenure"] * df["prior_claim_rate"]

    # 📊 Categorize delay into custom bins
    df["delay_bin"] = pd.cut(df["days_between_accident_and_claim"],
                             bins=[0, 3, 7, 14, 30, 100],
                             labels=["<3d", "3-7d", "7-14d", "14-30d", "30d+"])

    df["delay_flag"] = (df["days_between_accident_and_claim"] > 14).astype(int)

    # ⚠️ Risk Score Proxy = weighted heuristic for downstream analysis
    df["risk_score_proxy"] = (
        df["is_severe_damage"] * 3 +
        df["is_high_claim_history"] * 2 +
        df["delay_flag"] +
        df["has_safety_keywords"]
    )
    return df

# -----------------------------------
# ⚖️ Apply SMOTE Oversampling
# -----------------------------------

def apply_smote(df):
    """
    Address class imbalance using SMOTE oversampling strategy:
    - Keeps only numeric features for compatibility
    - Fills missing values using median strategy
    - Outputs balanced X + y dataframe

    Parameters:
        df (pd.DataFrame): Original DataFrame (must include 'is_fast_tracked')
    Returns:
        df_bal (pd.DataFrame): Balanced DataFrame with oversampled minority class
    """
    y = df["is_fast_tracked"]  # 🎯 Target variable
    X = df.drop(columns=["is_fast_tracked"])

    # ✅ Filter for numeric columns only
    X = X.select_dtypes(include=[np.number])

    # 🧼 Impute missing values before SMOTE
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 🧪 Apply SMOTE to balance target classes
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_imputed, y)
    print("\n✅ After SMOTE:", Counter(y_res))

    # 🧾 Return a combined balanced dataset
    df_bal = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name="is_fast_tracked")], axis=1)
    return df_bal

# -----------------------------------
# 💾 Save Final Dataset to RDS
# -----------------------------------

def save_to_rds(df, engine):
    """
    Clean column names and save final DataFrame into RDS

    Parameters:
        df (pd.DataFrame): Final engineered dataset
        engine: SQLAlchemy engine for MySQL connection
    """
    # 🧹 Clean column names to make SQL-safe
    df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)

    # 💾 Write to MySQL table (replace if exists)
    df.to_sql("motor_claims_features_engineered", engine, if_exists="replace", index=False)
    print("✅ Feature-engineered data saved to RDS")

# -----------------------------------
# 🚀 EXECUTION FLOW
# -----------------------------------
if __name__ == "__main__":
    # 🔄 Load cleaned base data
    df, engine = load_data()

    # ⚖️ Show class imbalance before feature work
    log_class_distribution(df)

    # 🧱 Feature Engineering Pipeline
    df = engineer_vehicle_features(df)
    df = engineer_customer_features(df)
    df = engineer_text_features(df)
    df = add_nlp_features(df)
    df = engineer_domain_logic(df)

    print("🧪 Final Shape Before SMOTE:", df.shape)

    # ⚖️ Balance Classes using