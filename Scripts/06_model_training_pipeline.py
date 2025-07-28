"""
06_model_training_pipeline.py

🔍 Purpose:
This script is used to train and evaluate various machine learning models to classify
motor insurance claims as "Fast-Tracked" (Class 1) or "Not Fast-Tracked" (Class 0).

🏗️ Pipeline Steps:
1. Load preprocessed data from AWS RDS (table: motor_claims_selected_features)
2. Split into training and testing datasets with stratified sampling
3. Apply feature scaling using StandardScaler
4. Train models (XGBoost, LightGBM, CatBoost, RandomForest, etc.)
5. Optimize classification threshold using F1-score
6. Evaluate models using PR AUC, Balanced Accuracy, and Class 1 metrics
7. Log models and artifacts into MLflow for experiment tracking

📦 Dependencies: pandas, numpy, sklearn, xgboost, lightgbm, catboost, mlflow, matplotlib, sqlalchemy, joblib, optuna

🧑 Author: [Your Name]
📅 Date: [YYYY-MM-DD]
"""

# --------------------- #
# 📦 Import Libraries   #
# --------------------- #
import os                                    # Provides functions to interact with the operating system (e.g., file/folder operations)
import pandas as pd                          # For data manipulation
import numpy as np                           # For numerical operations
import matplotlib.pyplot as plt              # For plotting precision-recall curves
import optuna                                # For hyperparameter tuning (if needed later)
import joblib                                # For saving models
import mlflow                                # For model logging and experiment tracking
import shap                                  # For model explainability (optional)
import warnings                              # To ignore sklearn warnings
import sqlalchemy                            # To connect to AWS RDS using SQLAlchemy

# ⚙️ Scikit-learn imports for ML pipeline
from sklearn.model_selection import train_test_split                      # Data splitting
from sklearn.metrics import (classification_report, precision_recall_curve,
                             average_precision_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score)     # Evaluation metrics
from sklearn.ensemble import RandomForestClassifier                       # Random Forest model
from sklearn.linear_model import LogisticRegression                       # Logistic Regression
from sklearn.calibration import CalibratedClassifierCV                    # Calibrated classifier
from sklearn.preprocessing import StandardScaler                          # StandardScaler
from xgboost import XGBClassifier                                          # XGBoost model
from lightgbm import LGBMClassifier                                       # LightGBM model
from catboost import CatBoostClassifier                                   # CatBoost model
from tabulate import tabulate                                             # Pretty print tables
import dataframe_image as dfi                                             # Save model comparison image

warnings.filterwarnings("ignore")  # Ignore warnings for clean output

# Ensure the 'charts' and 'models' directory exists for saving plots and visual outputs.
# If it already exists, do nothing (thanks to exist_ok=True)
os.makedirs("charts", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Store all model evaluation summaries
model_metrics_summary = []

# ----------------------------- #
# 1️⃣ Load Data from AWS RDS    #
# ----------------------------- #
def load_data():
    """
    Load the cleaned and feature-selected claims dataset from AWS RDS.
    Returns:
        df (pd.DataFrame): Dataset containing selected features and target column 'is_fast_tracked'
    """
    rds_uri = "mysql+pymysql://admin:mExmuk-kitqim-jodza9@suncorp.ct2ykcc82vni.ap-southeast-2.rds.amazonaws.com/Suncorp"
    engine = sqlalchemy.create_engine(rds_uri)
    df = pd.read_sql("SELECT * FROM motor_claims_selected_features", engine)
    return df

# ----------------------------------- #
# 2️⃣ Prepare Train/Test Split        #
# ----------------------------------- #
def prepare_data(df):
    """
    Split data into features (X) and target (y), then apply stratified train-test split.
    Args:
        df (pd.DataFrame): Input data with all features and target
    Returns:
        X_train, X_test, y_train, y_test: Train/test split of features and target
    """
    X = df.drop(columns=["is_fast_tracked"])  # Features
    y = df["is_fast_tracked"]                 # Binary Target
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# -------------------------------------- #
# 3️⃣ Standardize Features (Z-Score)     #
# -------------------------------------- #
def scale_data(X_train, X_test):
    """
    Apply Z-score standardization to training and test features.
    Args:
        X_train (DataFrame): Unscaled training features
        X_test (DataFrame): Unscaled test features
    Returns:
        X_train_scaled (ndarray): Scaled training data
        X_test_scaled (ndarray): Scaled test data
        scaler (StandardScaler): Trained scaler object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
    X_test_scaled = scaler.transform(X_test)        # Transform test data with same scaler
    return X_train_scaled, X_test_scaled, scaler

# ---------------------------------------------- #
# 4️⃣ Find Optimal Classification Threshold      #
# ---------------------------------------------- #
def optimize_threshold(y_true, y_probs):
    """
    Iterate through classification thresholds from 0.1 to 0.9 to find the optimal 
    threshold that maximizes the F1 score, which balances precision and recall.

    Parameters:
    ----------
    y_true : array-like
        Ground truth binary class labels (0 or 1) for the test set.
    y_probs : array-like
        Predicted probabilities from the model (for class 1).

    Returns:
    -------
    best_thresh : float
        Optimal classification threshold that yields the best F1 score.
    best_f1 : float
        Best F1 score achieved at the optimal threshold.
    """

    # Generate 100 evenly spaced thresholds between 0.1 and 0.9
    thresholds = np.linspace(0.1, 0.9, 100)

    # Initialize best F1 and best threshold
    best_f1 = 0
    best_thresh = 0.5  # Default fallback threshold

    # Iterate through thresholds to compute F1 score
    for t in thresholds:
        preds = (y_probs > t).astype(int)  # Binarize predictions

        # Calculate precision = TP / (TP + FP)
        precision = np.sum((preds == 1) & (y_true == 1)) / (np.sum(preds == 1) + 1e-6)

        # Calculate recall = TP / (TP + FN)
        recall = np.sum((preds == 1) & (y_true == 1)) / (np.sum(y_true == 1) + 1e-6)

        # Calculate F1 Score: harmonic mean of precision and recall
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        # Update best threshold if F1 improves
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # Return best-performing threshold and its F1 score
    return best_thresh, best_f1

# ------------------------------------------------------- #
# 5️⃣ Evaluate Model Performance and Log to MLflow        #
# ------------------------------------------------------- #
def evaluate_model(name, model, X_test, y_test, y_probs, threshold):
    """
    Evaluate model predictions against test labels using the specified threshold,
    generate classification metrics, plot the precision-recall curve, and log
    all results to MLflow.

    Parameters:
    ----------
    name : str
        Name of the model (used for logging and plotting).
    model : sklearn-compatible classifier
        Trained machine learning model.
    X_test : array-like
        Scaled test features.
    y_test : array-like
        Ground truth binary labels for test set.
    y_probs : array-like
        Predicted class probabilities (for class 1) from model.
    threshold : float
        Custom threshold to convert probabilities into class predictions.

    Returns:
    -------
    ap : float
        Average Precision score (area under PR curve).
    balanced_acc : float
        Balanced Accuracy score across both classes.
    """

    # 🧮 Convert probabilities into class labels using threshold
    preds = (y_probs > threshold).astype(int)

    # 📝 Generate human-readable classification report
    report = classification_report(y_test, preds, target_names=["Not FastTracked", "FastTracked"])
    print(f"\n🧾 Classification Report ({name}):\n{report}")

    # 📐 Calculate precision-recall AUC and balanced accuracy
    ap = average_precision_score(y_test, y_probs)
    balanced_acc = balanced_accuracy_score(y_test, preds)
    print(f"🔍 Test PR AUC: {ap:.4f} | Balanced Accuracy: {balanced_acc:.4f}")

    # 📊 Plot and save the precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    plt.plot(recall, precision, label=f"{name} PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(f"charts/{name}_pr_curve.png")

    # ✅ Calculate Class 1 metrics for more granular business KPIs
    precision_class1 = precision_score(y_test, preds)
    recall_class1 = recall_score(y_test, preds)
    f1_class1 = f1_score(y_test, preds)

    # 📦 Append metrics to global summary list (used later for comparison)
    model_metrics_summary.append({
        "Model": name,
        "PR AUC": round(ap, 4),
        "Balanced Accuracy": round(balanced_acc, 4),
        "Precision (Class 1)": round(precision_class1, 2),
        "Recall (Class 1)": round(recall_class1, 2),
        "F1-Score (Class 1)": round(f1_class1, 2)
    })

    # 📚 Track the run in MLflow experiment tracker
    mlflow.set_experiment("MotorClaims FastTrack Classifier")  # Set experiment name
    with mlflow.start_run(run_name=f"{name}_run"):
        mlflow.log_params(model.get_params())                    # Log model parameters
        mlflow.log_metric("Test_PR_AUC", ap)                     # Log PR AUC
        mlflow.log_metric("Balanced_Accuracy", balanced_acc)     # Log Balanced Accuracy
        mlflow.log_metric("Optimized_Threshold", threshold)      # Log threshold
        mlflow.log_artifact(f"charts/{name}_pr_curve.png")       # Save PR curve
        mlflow.sklearn.log_model(model, "model")                 # Log model artifact
        print("📊 Logged to MLflow")

    # Return summary metrics for further use (e.g. comparison)
    return ap, balanced_acc

def compare_base_models(X_train, y_train, X_test, y_test):
    """
    Trains and evaluates a set of baseline models on training and test data.

    Parameters:
    - X_train (array): Feature matrix for training set
    - y_train (array): Labels for training set
    - X_test (array): Feature matrix for test set
    - y_test (array): Labels for test set

    Returns:
    - scores (dict): Dictionary mapping model names to a tuple of (PR AUC, Balanced Accuracy, trained model)
    """
    
    # Dictionary of models to compare, each with predefined hyperparameters and class balancing
    models = {
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=2, random_state=42),
        "LightGBM": LGBMClassifier(class_weight="balanced", random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42, auto_class_weights="Balanced"),
        "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=42),
        "LogisticRegression": LogisticRegression(class_weight="balanced", random_state=42)
    }

    scores = {}  # Dictionary to store evaluation scores for each model

    # Iterate over each model
    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        model.fit(X_train, y_train)  # Train the model

        # Predict probabilities for class 1
        y_probs = model.predict_proba(X_test)[:, 1]

        # Find optimal threshold based on F1 score
        threshold, _ = optimize_threshold(y_test, y_probs)

        # Evaluate and store results
        pr_auc, bal_acc = evaluate_model(name, model, X_test, y_test, y_probs, threshold)
        scores[name] = (pr_auc, bal_acc, model)

    return scores

def tune_best_model(name, base_model, X_train, y_train):
    """
    Tunes the best performing base model using Optuna and returns the optimized model.

    Parameters:
    - name (str): Model name to tune (e.g., 'XGBoost', 'LightGBM', etc.)
    - base_model: The initial model object before tuning
    - X_train: Feature matrix for training set
    - y_train: Labels for training set

    Returns:
    - tuned_model: Trained model with best parameters found
    """
    
    def objective(trial):
        # Define hyperparameter search space per model
        if name == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
            model = XGBClassifier(eval_metric="logloss", scale_pos_weight=2, use_label_encoder=False, **params)

        elif name == "LightGBM":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 31, 128),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            }
            model = LGBMClassifier(class_weight="balanced", **params)

        elif name == "CatBoost":
            params = {
                'iterations': trial.suggest_int("iterations", 100, 500),
                'depth': trial.suggest_int("depth", 4, 10),
                'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3),
            }
            model = CatBoostClassifier(verbose=0, random_state=42, auto_class_weights="Balanced", **params)

        elif name == "RandomForest":
            params = {
                'n_estimators': trial.suggest_int("n_estimators", 100, 500),
                'max_depth': trial.suggest_int("max_depth", 5, 20),
                'max_features': trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            }
            model = RandomForestClassifier(class_weight="balanced", random_state=42, **params)

        elif name == "MLP":
            params = {
                'hidden_layer_sizes': trial.suggest_categorical("hidden_layer_sizes", [(64,), (128,), (128, 64)]),
                'alpha': trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
                'learning_rate_init': trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
            }
            model = MLPClassifier(max_iter=300, random_state=42, **params)

        else:
            raise ValueError(f"Tuning not supported for model: {name}")

        model.fit(X_train, y_train)  # Train the model with current trial params
        y_probs = model.predict_proba(X_train)[:, 1]  # Predict on train set to evaluate
        return average_precision_score(y_train, y_probs)  # Use PR AUC for optimization

    # Run Optuna optimization
    print(f"🔍 Running Optuna tuning for {name}...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)  # Run for 25 trials

    best_params = study.best_params
    print(f"✅ Best params for {name}: {best_params}")

    # Re-train model on best params from Optuna
    if name == "XGBoost":
        tuned_model = XGBClassifier(eval_metric="logloss", scale_pos_weight=2, use_label_encoder=False, **best_params)
    elif name == "LightGBM":
        tuned_model = LGBMClassifier(class_weight="balanced", **best_params)
    elif name == "CatBoost":
        tuned_model = CatBoostClassifier(verbose=0, random_state=42, auto_class_weights="Balanced", **best_params)
    elif name == "RandomForest":
        tuned_model = RandomForestClassifier(class_weight="balanced", random_state=42, **best_params)
    elif name == "MLP":
        tuned_model = MLPClassifier(max_iter=300, random_state=42, **best_params)

    tuned_model.fit(X_train, y_train)  # Fit the tuned model on full training data
    return tuned_model

def generate_shap(model, X_train_raw, feature_names, model_name="Model"):
    """
    Generate SHAP summary plot to interpret the feature importance 
    and contribution for a given trained model.

    Parameters:
    ----------
    model : sklearn-compatible model
        Trained classifier to explain using SHAP.
    X_train_raw : array or DataFrame
        Raw training feature data (unscaled or scaled) used for explanation.
    feature_names : list
        List of feature names to display in SHAP summary plot.
    model_name : str, optional
        Name of the model (used in print/log messages and plot title).

    Returns:
    -------
    None. Saves SHAP plot as 'shap_summary.png'.
    """

    import shap
    import matplotlib.pyplot as plt

    print(f"🔎 Generating SHAP summary for {model_name}...")

    # ⚡ Sample 500 rows for performance; SHAP is slow on large data
    X_sample = pd.DataFrame(X_train_raw, columns=feature_names).sample(n=500, random_state=42)

    try:
        # 🌳 Use TreeExplainer for tree-based models (fast & accurate)
        if model_name.startswith("XGBoost") or model_name.startswith("LightGBM") or model_name.startswith("CatBoost"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            # 🧠 Fallback for generic models like MLP or RandomForest
            explainer = shap.Explainer(model.predict, X_sample)
            shap_values = explainer(X_sample)

        # 📊 Generate SHAP summary plot (bar + beeswarm)
        shap.summary_plot(shap_values, features=X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig("charts/shap_summary.png")
        print("📊 SHAP summary plot saved to shap_summary.png")

    except Exception as e:
        print(f"❌ SHAP generation failed: {e}")

def print_model_comparison_table():
    """
    Display and save a visual comparison of model performance metrics 
    (PR AUC, Balanced Accuracy, Precision/Recall/F1 for class 1).

    Uses `model_metrics_summary` global list populated by `evaluate_model`.

    Returns:
    -------
    None. Saves the table image as 'model_comparison_table.png'.
    """

    # 📘 Convert collected model metrics to DataFrame
    df = pd.DataFrame(model_metrics_summary)

    print("\n📋 Model Comparison Summary:\n")

    # 📄 Pretty-print the table in terminal using tabulate
    from tabulate import tabulate
    print(tabulate(df, headers="keys", tablefmt="github"))

    # 📷 Export the table as an image for reporting
    import dataframe_image as dfi
    dfi.export(df, "charts/model_comparison_table.png")
    print("📷 Model comparison table saved as model_comparison_table.png")

def main():
    """
    Main orchestration function that:
    1. Loads data
    2. Splits and scales it
    3. Trains multiple base models
    4. Identifies and tunes the best model
    5. Evaluates and explains the final model
    6. Saves the best model and scaler for reuse

    Returns:
    -------
    None. Outputs visual reports, model files, and logs to MLflow.
    """

    # 📥 Step 1: Load processed data with selected features from RDS
    df = load_data()

    # ✂️ Step 2: Split into train/test sets
    X_train, X_test, y_train, y_test = prepare_data(df)

    # ⚖️ Step 3: Scale features using StandardScaler
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # 📊 Step 4: Train all base models and evaluate them
    print("\n📊 Step 1: Train Base Models")
    scores = compare_base_models(X_train_scaled, y_train, X_test_scaled, y_test)

    # 🏆 Step 5: Select best model by highest PR AUC
    print("\n✅ Step 2: Identify Best Model")
    best_model_name = max(scores.items(), key=lambda x: x[1][0])[0]
    base_model = scores[best_model_name][2]
    print(f"\n🎯 Best Model Identified: {best_model_name}, proceeding to tuning...")

    # 🛠️ Step 6: Tune best model with Optuna if supported
    if best_model_name in ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "MLP"]:
        tuned_model = tune_best_model(best_model_name, base_model, X_train_scaled, y_train)

        # 🎯 Predict using tuned model and evaluate
        y_probs = tuned_model.predict_proba(X_test_scaled)[:, 1]
        threshold, _ = optimize_threshold(y_test, y_probs)
        evaluate_model(f"{best_model_name}_Tuned", tuned_model, X_test_scaled, y_test, y_probs, threshold)

        # 📈 Step 7: Explain model with SHAP
        print("\n📈 Step 4: SHAP Explainability")
        generate_shap(
            tuned_model,
            X_train_scaled,
            df.drop(columns="is_fast_tracked").columns,
            model_name=f"{best_model_name}_Tuned"
        )
    else:
        print(f"❌ Skipping tuning — no tuning logic implemented for {best_model_name}")

    # 📋 Step 8: Print final model comparison
    print_model_comparison_table()

    # 💾 Step 9: Save best model and scaler to disk for reuse
    joblib.dump(tuned_model, f"models/best_model.pkl")
    print(f"💾 Tuned model saved as best_model.pkl")

    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Scaler saved to scaler.pkl")

# 🟢 Entrypoint for script execution
if __name__ == "__main__":
    main()