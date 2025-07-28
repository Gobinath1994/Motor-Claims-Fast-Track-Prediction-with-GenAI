# -----------------------------------
# Motor Claims Feature Selection Script
# -----------------------------------
# This script performs automated feature selection using multiple statistical,
# tree-based, and wrapper methods on engineered motor claims data stored in AWS RDS.
# It selects the most predictive features using voting and saves final features to RDS.

# 📦 Imports
import pandas as pd  # Data manipulation
import numpy as np  # Numeric operations
from sklearn.feature_selection import mutual_info_classif  # Mutual Information
from sklearn.linear_model import LogisticRegressionCV  # Logistic Regression with Cross Validation
from sklearn.ensemble import RandomForestClassifier  # Tree-based model for Boruta
from sklearn.inspection import permutation_importance  # Permutation importance
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler  # Feature scaling
from sklearn.decomposition import PCA  # PCA for embeddings
from lightgbm import LGBMClassifier  # LightGBM importance
from imblearn.ensemble import BalancedRandomForestClassifier  # Balanced RF
from boruta import BorutaPy  # Wrapper feature selection
import sqlalchemy  # RDS connection
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------
# 📂 Load Feature-Engineered Data from RDS
# -----------------------------------
def load_data():
    """
    Loads the feature-engineered claims dataset from RDS.

    Returns:
        df (pd.DataFrame): DataFrame with engineered features
        engine: SQLAlchemy connection engine
    """
    # Define the RDS database URI (MySQL format) with credentials and connection details
    rds_uri = (
        "mysql+pymysql://admin:mExmuk-kitqim-jodza9@"
        "suncorp.ct2ykcc82vni.ap-southeast-2.rds.amazonaws.com/Suncorp"
    )

    # Create a SQLAlchemy engine using the provided RDS URI — used to interact with the RDS instance
    engine = sqlalchemy.create_engine(rds_uri)

    # Query the table 'motor_claims_features_engineered' from the RDS database and load into a pandas DataFrame
    df = pd.read_sql("SELECT * FROM motor_claims_features_engineered", engine)

    # Return both the DataFrame and the SQLAlchemy engine for further processing or saving
    return df, engine

# -----------------------------------
# 🔄 Apply PCA on Embedding Features
# -----------------------------------
def apply_pca_on_embeddings(X, n_components=10):
    """
    Applies PCA to compress high-dimensional text embedding features.

    Parameters:
        X (pd.DataFrame): Input feature DataFrame (includes columns like 'text_emb_0', 'text_emb_1', ...)
        n_components (int): Number of PCA components to retain (default is 10)

    Returns:
        X (pd.DataFrame): Modified DataFrame with original text embeddings replaced by PCA components
        pca_df (pd.DataFrame): DataFrame containing PCA components for optional use or saving
    """
    
    # Identify columns that start with "text_emb_" (i.e., embedding features to compress)
    embedding_cols = [col for col in X.columns if col.startswith("text_emb_")]
    
    # If no embedding columns are found, return X unchanged and None for PCA components
    if not embedding_cols:
        return X, None

    # Initialize PCA object with n_components and a fixed random seed for reproducibility
    pca = PCA(n_components=n_components, random_state=42)

    # Fit PCA on the embedding columns and transform them into lower-dimensional components
    pca_comps = pca.fit_transform(X[embedding_cols])

    # Create a DataFrame with PCA components and name columns as "text_pca_1", ..., "text_pca_n"
    pca_df = pd.DataFrame(pca_comps, columns=[f"text_pca_{i+1}" for i in range(n_components)])

    # Drop the original high-dimensional embedding columns
    X = X.drop(columns=embedding_cols).reset_index(drop=True)

    # Concatenate PCA components with the rest of the features
    X = pd.concat([X, pca_df], axis=1)

    # Return the updated DataFrame and the PCA-only DataFrame separately
    return X, pca_df

# -----------------------------------
# ❌ Drop Highly Correlated Features
# -----------------------------------
def drop_high_corr_features(data, threshold=0.95):
    """
    Removes highly correlated features from a DataFrame to reduce redundancy.

    Parameters:
        data (pd.DataFrame): Input feature DataFrame with numeric values.
        threshold (float): Correlation coefficient threshold above which one of the two correlated features is dropped.
                           Default is 0.95.

    Returns:
        pd.DataFrame: DataFrame with reduced feature set after dropping highly correlated columns.
    """
    # Compute the absolute correlation matrix between all pairs of features
    corr = data.corr().abs()

    # Create an upper triangle matrix of correlations (excluding self-correlation and duplicates)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Identify columns with any correlation value above the threshold
    drop_cols = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop the identified columns from the dataset
    return data.drop(columns=drop_cols)

# -----------------------------------
# 🧠 Run Feature Selection Methods
# -----------------------------------
def run_feature_selection(X_scaled, y):
    """
    Runs six feature selection methods to identify top informative features:
    1. Mutual Information
    2. LightGBM Feature Importance
    3. Permutation Importance with Logistic Regression (ROC AUC)
    4. L1-penalized Logistic Regression (Lasso)
    5. Boruta feature selection using RandomForest
    6. Balanced Random Forest importance

    Parameters:
        X_scaled (pd.DataFrame): Scaled input feature matrix.
        y (pd.Series or np.array): Target variable (binary classification).

    Returns:
        dict: A dictionary mapping method names to lists of selected/top features.
    """

    features = {}  # Dictionary to store top features from each method

    # ---------- 1. Mutual Information ----------
    # Compute mutual information between each feature and the target
    mi = mutual_info_classif(X_scaled, y, random_state=42)

    # Select top 30 features based on mutual information score
    features["mi_top"] = pd.Series(mi, index=X_scaled.columns) \
                            .sort_values(ascending=False) \
                            .head(30).index.tolist()

    # ---------- 2. LightGBM Feature Importance ----------
    # Initialize a LightGBM classifier with balanced class weights
    lgbm = LGBMClassifier(random_state=42, class_weight='balanced')

    # Train the model on the scaled features and target
    lgbm.fit(X_scaled, y)

    # Get feature importances, sort them, and take the top 30
    features["lgbm_top"] = pd.Series(lgbm.feature_importances_, index=X_scaled.columns) \
                              .sort_values(ascending=False) \
                              .head(30).index.tolist()

    # ---------- 3. Permutation Importance ----------
    # Split data into train/test sets for permutation importance evaluation
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=0.2, random_state=42
    )

    # Use LogisticRegressionCV with ROC AUC scoring
    lr_auc = LogisticRegressionCV(
        cv=5, scoring="roc_auc", class_weight="balanced", max_iter=1000
    )

    # Fit the logistic model to training data
    lr_auc.fit(x_train, y_train)

    # Compute permutation importance on test data
    perm = permutation_importance(
        lr_auc, x_test, y_test, scoring="roc_auc", n_repeats=5, random_state=42
    )

    # Select top 30 features based on permutation importance mean
    features["perm_top"] = pd.Series(perm.importances_mean, index=X_scaled.columns) \
                               .sort_values(ascending=False) \
                               .head(30).index.tolist()

    # ---------- 4. L1-Penalized Logistic Regression (Lasso) ----------
    # Logistic Regression with L1 penalty selects sparse features
    lr_l1 = LogisticRegressionCV(
        cv=5, penalty="l1", solver="saga", class_weight="balanced",
        scoring="roc_auc", max_iter=1000
    )

    # Fit the L1-regularized model
    lr_l1.fit(X_scaled, y)

    # Get features with non-zero coefficients (selected by L1 regularization)
    features["l1_selected"] = X_scaled.columns[(lr_l1.coef_ != 0).flatten()].tolist()

    # ---------- 5. Boruta Feature Selection ----------
    # Use RandomForest with balanced class weight for Boruta feature selection
    rf = RandomForestClassifier(
        n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42
    )

    # Initialize Boruta with automatic estimator count
    boruta_selector = BorutaPy(rf, n_estimators='auto', random_state=42)

    # Fit Boruta on raw values of X and y
    boruta_selector.fit(X_scaled.values, y.values)

    # Get features confirmed by Boruta as important
    features["boruta_selected"] = X_scaled.columns[boruta_selector.support_].tolist()

    # ---------- 6. Balanced Random Forest ----------
    # Train a Balanced Random Forest classifier to handle class imbalance
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model
    brf.fit(X_scaled, y)

    # Get top 30 features by feature importance from BRF
    features["brf_top"] = pd.Series(brf.feature_importances_, index=X_scaled.columns) \
                             .sort_values(ascending=False) \
                             .head(30).index.tolist()

    return features  # Return dictionary with feature list from each method

# -----------------------------------
# 🗳️ Voting Mechanism
# -----------------------------------
def vote_features(method_outputs, min_votes=3):
    """
    Aggregates features selected by multiple feature selection methods.
    A feature is retained if it appears in at least `min_votes` methods.

    Parameters:
        method_outputs (dict): Dictionary where keys are method names
                               and values are lists of selected features.
        min_votes (int): Minimum number of method "votes" required to retain a feature.

    Returns:
        final_feats (list): List of features that passed the vote threshold.
    """

    # Flatten all feature lists from each method into one long list
    all_feats = [f for method in method_outputs.values() for f in method]

    # Count how many times each feature appears across all methods
    vote_counts = pd.Series(all_feats).value_counts()

    # Retain features that appear in at least `min_votes` methods
    final_feats = vote_counts[vote_counts >= min_votes].index.tolist()

    # Return the final list of selected features
    return final_feats

# -----------------------------------
# 💾 Save Final Feature Set to RDS
# -----------------------------------
def save_selected_features(df_full, final_features, engine, pca_df=None):
    """
    Saves the selected features along with the target column into the RDS table
    named 'motor_claims_selected_features'.

    Parameters:
        df_full (pd.DataFrame): The original feature-engineered DataFrame.
        final_features (list): List of selected feature names to retain.
        engine (sqlalchemy.engine.Engine): SQLAlchemy engine for RDS connection.
        pca_df (pd.DataFrame, optional): Optional PCA features to append if available.
    """

    # If PCA components are provided, concatenate them to the full dataset
    if pca_df is not None:
        df_full = df_full.reset_index(drop=True)  # Reset index to ensure alignment
        df_full = pd.concat([df_full, pca_df], axis=1)  # Add PCA features to the DataFrame

    # Check for any selected features that are missing in the DataFrame
    missing = [f for f in final_features if f not in df_full.columns]
    if missing:
        raise ValueError(f"Final selected features missing in DataFrame: {missing}")

    # Subset the DataFrame to only include selected features + target column
    selected_df = df_full[final_features + ["is_fast_tracked"]]

    # Save the selected features into RDS table, replacing existing table if it exists
    selected_df.to_sql("motor_claims_selected_features", engine, if_exists="replace", index=False)

    # Print a success message with number of features saved
    print(f"Selected {len(final_features)} features saved to RDS.")

# -----------------------------------
# 🚀 Main Execution
# -----------------------------------
def main():
    # Step 1: Load the full feature-engineered dataset from AWS RDS
    df, engine = load_data()

    # Separate input features (X) and target variable (y)
    X = df.drop(columns=["is_fast_tracked"])  # Exclude target column from features
    y = df["is_fast_tracked"]  # Target: whether a claim was fast-tracked

    # Step 2: Apply PCA to compress sentence embeddings (if present)
    X, pca_df = apply_pca_on_embeddings(X)

    # Keep only numeric features and fill any missing values with 0
    X = X.select_dtypes(include=[np.number]).fillna(0)

    # Drop highly correlated features to avoid multicollinearity
    X = drop_high_corr_features(X)

    # Step 3: Scale features using StandardScaler (important for models like Logistic Regression)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Step 4: Run 6 different feature selection methods and gather their top features
    method_outputs = run_feature_selection(X_scaled, y)

    # Step 5: Vote across all methods to retain only features selected by 3 or more techniques
    final_features = vote_features(method_outputs, min_votes=3)

    # Step 6: Save the final selected features + target column to AWS RDS
    save_selected_features(df, final_features, engine, pca_df)

# Execute the main function when this script is run directly
if __name__ == "__main__":
    main()