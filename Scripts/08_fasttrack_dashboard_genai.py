# -----------------------------------------------------------------------------------------------
# 🧾 File: streamlit_dashboard.py
# 📌 Purpose: Streamlit-based interactive dashboard for fast-track motor claims prediction,
#             risk tagging, and Amazon Bedrock GenAI summary generation.
#             This top section loads models, sets up AWS + RDS connections, and imports modules.
# -----------------------------------------------------------------------------------------------

# ✅ Core Streamlit Web App framework
import streamlit as st

# ✅ Data processing libraries
import pandas as pd                          # For working with tabular data (DataFrames)
import numpy as np                          # For numerical operations

# ✅ ML & Vectorization utilities
import joblib                               # For loading serialized models and transformers
import json, re, time, random               # Built-in libraries: parsing, regex, delays, randomness

# ✅ AWS integration for GenAI and model inference
import boto3                                # AWS SDK (used here for calling Bedrock)

# ✅ Database support
import sqlalchemy                           # ORM and SQL connection (used here for AWS RDS)

# ✅ Visualization
import plotly.express as px                 # For interactive visualizations in the dashboard

# ✅ Date/time management
from datetime import datetime               # Used for timestamping records

# ✅ Text embeddings model
from sentence_transformers import SentenceTransformer  # For converting text into embeddings

# ✅ Scalers
from sklearn.preprocessing import MinMaxScaler  # For feature scaling (if needed in preprocessing)

# -----------------------------------------------------------------------------------------------
# 🎯 Load ML and NLP artifacts saved during training
# -----------------------------------------------------------------------------------------------

# 🔍 Main classifier model (e.g., RandomForest or XGBoost)
model = joblib.load("models/best_model.pkl")

# 🧠 Pre-trained TF-IDF vectorizer for converting descriptions to sparse text features
tfidf = joblib.load("models/tfidf_vectorizer.joblib")

# 🔄 PCA transformer used to reduce dimensionality of embeddings
pca = joblib.load("models/pca_transformer.joblib")

# 📊 KMeans clustering model for segmenting similar damage descriptions
kmeans = joblib.load("models/kmeans_cluster.joblib")

# 🔧 Feature scaler used during model training for normalization
scaler = joblib.load("models/scaler.pkl")

# 📋 Final list of selected features after feature selection pipeline
selected_features = joblib.load("models/selected_features.pkl")

# 🔤 Sentence-level embedding model (MiniLM) for dense semantic understanding
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------------------------------------------------------------------
# 🌩️ Set up Amazon Bedrock runtime client
# Used for GenAI inference (Mistral 7B in production) — summaries, tags, recommendations.
# -----------------------------------------------------------------------------------------------
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-2")

# -----------------------------------------------------------------------------------------------
# 🛢️ AWS RDS database connection (MySQL)
# Used to load or save batch claims, predictions, summaries, and logs.
# -----------------------------------------------------------------------------------------------
rds_uri = (
    "mysql+pymysql://admin:mExmuk-kitqim-jodza9@"
    "suncorp.ct2ykcc82vni.ap-southeast-2.rds.amazonaws.com/Suncorp"
)
engine = sqlalchemy.create_engine(rds_uri)
# ---------------------------------------------------------------------------------------------------- #
# 🔧 Function: feature_engineering(df)
# 📌 Purpose: Apply domain-specific, statistical, text-based, and interaction-based transformations
#     to enrich the input claims data before feeding it into the trained model.
# 📥 Input:
#     df (pd.DataFrame): Raw claims DataFrame uploaded via the UI or read from database.
# 📤 Output:
#     df (pd.DataFrame): Enhanced DataFrame with all engineered features ready for prediction.
# ---------------------------------------------------------------------------------------------------- #
def feature_engineering(df):
    # ----------------------- Numerical Transformations -----------------------
    
    # Calculate vehicle age based on the assumption it's 2025 now
    df["vehicle_age"] = 2025 - df["vehicle_year"]
    
    # Derive average mileage per year to normalize usage
    df["mileage_per_year"] = df["vehicle_mileage"] / df["vehicle_age"].replace(0, 1)
    
    # Flag if mileage is unusually high (possible old or frequently used vehicle)
    df["high_mileage_flag"] = (df["vehicle_mileage"] > 150000).astype(int)
    
    # Log-transform mileage for better model learning (reduce skew)
    df["log_mileage"] = np.log1p(df["vehicle_mileage"])

    # ----------------------- Customer Profile Features -----------------------

    # Claims filed per year of being a customer — proxy for claim frequency
    df["prior_claim_rate"] = df["historical_claims_count"] / df["customer_tenure"].replace(0, 1)
    
    # Flag if customer has made >1 past claim — potentially risky
    df["is_high_claim_history"] = (df["historical_claims_count"] > 1).astype(int)
    
    # Group tenure into bins to reduce granularity
    df["tenure_group"] = pd.cut(df["customer_tenure"], bins=[0, 2, 5, 10, 20], labels=["<2y", "2-5y", "5-10y", "10+y"])
    
    # Loyalty indicator if customer tenure > 7 years
    df["is_loyal_customer"] = (df["customer_tenure"] > 7).astype(int)
    
    # Log-transform tenure to reduce skew
    df["log_tenure"] = np.log1p(df["customer_tenure"])

    # ----------------------- Text Description Features -----------------------

    # Calculate number of words in damage description
    df["desc_length"] = df["damage_description"].str.split().apply(len)
    
    # Flag short descriptions (which may be low-effort or less detailed)
    df["low_effort_text"] = (df["desc_length"] < 12).astype(int)

    # Manually crafted keyword features based on domain knowledge
    keywords = {
        "has_safety_keywords": ["fluid", "leaks", "alignment", "safety", "confirmed", "concern"],
        "mentions_minor": ["minor", "scuff", "surface", "bending"]
    }

    # Search for each keyword in the damage_description column
    for colname, terms in keywords.items():
        df[colname] = df["damage_description"].apply(
            lambda x: int(any(re.search(rf"\b{term}\b", str(x).lower()) for term in terms))
        )

    # ----------------------- TF-IDF Vectorization -----------------------

    # Transform text using pre-trained TF-IDF vectorizer
    tfidf_matrix = tfidf.transform(df["damage_description"])
    
    # Convert sparse matrix to dense DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{t}" for t in tfidf.get_feature_names_out()])
    
    # Merge TF-IDF features back to original data
    df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    # ----------------------- Sentence Embeddings -----------------------

    # Use pre-loaded MiniLM model to encode text as dense embeddings
    embeddings = embedder.encode(df["damage_description"].tolist(), show_progress_bar=False)
    
    # Convert embeddings into DataFrame
    emb_df = pd.DataFrame(embeddings, columns=[f"text_emb_{i}" for i in range(embeddings.shape[1])])
    
    # Append embedding features to main DataFrame
    df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    # ----------------------- PCA Reduction on Embeddings -----------------------

    # Reduce high-dimensional embeddings into 5 principal components
    pca_comps = pca.transform(embeddings)
    for i in range(pca_comps.shape[1]):
        df[f"text_pca_{i+1}"] = pca_comps[:, i]

    # ----------------------- Clustering -----------------------

    # Predict semantic cluster using pre-trained KMeans
    df["text_cluster"] = kmeans.predict(embeddings)

    # ----------------------- Binary & Interaction Flags -----------------------

    # Convert severe damage into binary flag
    df["is_severe_damage"] = (df["damage_level_reported"] == "Severe").astype(int)
    
    # Convert garage_estimate_provided from boolean to int
    df["is_garage_provided"] = df["garage_estimate_provided"].astype(int)

    # Create logical interaction between severity and estimate
    df["est_provided_x_severity"] = df["is_garage_provided"] * df["is_severe_damage"]
    
    # Combine age and mileage to detect wear/tear risks
    df["vehicle_age_x_mileage"] = df["vehicle_age"] * df["mileage_per_year"]
    
    # Combine delay and severity to capture urgency
    df["claim_delay_x_severity"] = df["days_between_accident_and_claim"] * df["is_severe_damage"]

    # Combine loyalty and prior claims to model risk over time
    df["tenure_x_claim_rate"] = df["customer_tenure"] * df["prior_claim_rate"]

    # Flag claims that were delayed more than 2 weeks
    df["delay_flag"] = (df["days_between_accident_and_claim"] > 14).astype(int)

    # ----------------------- Domain Risk Scoring -----------------------

    # Heuristic-based scoring: high risk if more severe, delayed, or history of many claims
    df["risk_score_proxy"] = (
        df["is_severe_damage"] * 3 +
        df["is_high_claim_history"] * 2 +
        df["delay_flag"] +
        df["has_safety_keywords"]
    )

    # ✅ Return enhanced DataFrame
    return df

# ------------------------------------------------------------------------------------------------
# 🤖 GenAI Summarization & Reasoning via Amazon Bedrock (Mistral 7B)
# ------------------------------------------------------------------------------------------------

def invoke_bedrock(prompt, max_retries=3):
    """
    Calls the Amazon Bedrock Mistral model with a given prompt and handles throttling retries.

    Parameters:
        prompt (str): Instruction text to send to the Mistral model (e.g., claim summary request).
        max_retries (int): Number of retry attempts if throttled (default: 3).

    Returns:
        str: The generated response from the Mistral model, or an error message if failed.
    """
    for attempt in range(max_retries):
        try:
            # 🔁 Send the prompt to Amazon Bedrock's Mistral 7B model
            response = bedrock.invoke_model(
                modelId="mistral.mistral-7b-instruct-v0:2",  # ✅ Mistral 7B via Bedrock
                contentType="application/json",              # 📤 Input format
                accept="application/json",                   # 📥 Output format
                body=json.dumps({
                    "prompt": prompt,                        # 🧠 Text instruction to the model
                    "max_tokens": 512,                       # ✂️ Max tokens in output
                    "temperature": 0.3                       # 🎲 Lower value = more deterministic
                })
            )

            # 📦 Read and decode JSON response
            output = json.loads(response["body"].read())
            
            # ✅ Return the first generated output
            return output["outputs"][0]["text"]

        except Exception as e:
            # 🔄 If rate-limited (ThrottlingException), wait and retry
            if "ThrottlingException" in str(e):
                time.sleep(random.uniform(1.5, 3.0))  # ⏱️ Backoff between 1.5–3.0s
            else:
                # ❌ Return error message for other exceptions
                return f"❌ Error: {str(e)}"
    
    # ❌ If all retries fail
    return "❌ Max retries reached"

def get_genai_outputs(record):
    """
    Generates three GenAI outputs for a given motor insurance claim record:
    1. Executive claim summary
    2. Risk tags
    3. Recommended next action

    Parameters:
        record (dict): A single claim record (row) as a dictionary after preprocessing.

    Returns:
        summary (str): A business-readable summary of the claim.
        tags (str): A short list of 3 risk-related tags.
        next_step (str): A single next-step recommendation for the claim.
    """
    # 📄 Convert dictionary record to formatted JSON string
    record_json = json.dumps(record, indent=2)

    # 🧠 Build base prompt context for all queries
    base_prompt = f"Summarise the following motor insurance claim:\n{record_json}\n"

    # 1️⃣ Generate claim summary + explanation of fast-track reason
    summary = invoke_bedrock(base_prompt + "\nWhy is it fast-tracked or not?")

    # 2️⃣ Generate short risk factor tags (for display or analysis)
    tags = invoke_bedrock(base_prompt + "\nGive 3 short risk factor tags for this claim.")

    # 3️⃣ Generate recommended next step action (e.g., inspection, payout, etc.)
    next_step = invoke_bedrock(base_prompt + "\nSuggest one next step action for this claim.")

    # ✅ Return all outputs
    return summary, tags, next_step

# ------------------------------------------------------------------------------------------------
# 📊 Streamlit Dashboard UI – Fast-Track Claim Predictor with GenAI Summaries
# ------------------------------------------------------------------------------------------------

# 🌐 Set Streamlit page configuration (title and layout)
st.set_page_config(page_title="🚗 Fast-Track Claim Predictor", layout="wide")

# 🏷️ Title of the application (centered, with HTML formatting)
st.markdown("<h1 style='text-align:center;'>🚗 Motor Claims Fast-Track Predictor</h1>", unsafe_allow_html=True)

# Horizontal line separator
st.markdown("<hr>", unsafe_allow_html=True)

# 📤 CSV Upload section for users to input raw claims
uploaded = st.file_uploader("📤 Upload motor claims CSV", type="csv")

# ------------------------------------------------------------------------------------------
# 📁 Once file is uploaded, begin processing pipeline
# ------------------------------------------------------------------------------------------
if uploaded:
    # 🧾 Load CSV file into DataFrame
    raw = pd.read_csv(uploaded)

    # ℹ️ Notify user that feature engineering has started
    st.info("🔄 Running Feature Engineering...")

    # ⚙️ Apply domain + NLP + vectorization transformations
    processed = feature_engineering(raw.copy())

    # ✅ Notify completion of feature pipeline
    st.success("✅ Feature Engineering Complete")

    # 🧪 Select and scale only the chosen features
    X = processed[selected_features]
    X_scaled = scaler.transform(X)

    # 🤖 Run classification model to generate predictions and probabilities
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    # 📊 Append predictions and timestamp to raw DataFrame
    raw["fast_track_prediction"] = preds                         # Binary class (0 or 1)
    raw["confidence_score"] = probs.round(4)                     # Probability score (0–1)
    raw["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Timestamp of prediction

    # ✅ Notify that ML predictions are completed
    st.success("✅ Predictions Complete")

    # --------------------------------------------------------------------------------------
    # 🧠 GenAI Integration: Generate claim summaries, risk tags, and next steps
    # --------------------------------------------------------------------------------------
    st.info("🧠 Generating LLM Summaries + Tags + Next Steps...")

    summaries, tags, steps = [], [], []

    # 🔁 Loop over each record to invoke GenAI for insights
    for i, row in raw.iterrows():
        summary, tag, step = get_genai_outputs(row.to_dict())  # 🤖 Prompt Bedrock
        summaries.append(summary)
        tags.append(tag)
        steps.append(step)
        time.sleep(1)  # 💤 Small delay to avoid throttling Bedrock API

    # 🧾 Append GenAI columns to raw DataFrame
    raw["genai_summary"] = summaries       # 📄 Executive summary of claim
    raw["risk_tags"] = tags                # 🏷️ Risk factor tags
    raw["next_step"] = steps               # 🚦 Recommended action

    # ✅ Notify that GenAI tasks are completed
    st.success("🧠 GenAI Processing Done")

    # --------------------------------------------------------------------------------------
    # 📊 Dashboard Metrics + Class Distribution Chart
    # --------------------------------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        # 📦 Total records and Fast-track counts
        st.metric("📦 Total Claims", len(raw))
        st.metric("✅ Fast-Tracked", (raw["fast_track_prediction"] == 1).sum())

    with col2:
        # 📊 Bar chart: Distribution of prediction classes
        chart_df = raw["fast_track_prediction"].value_counts().reset_index()
        chart_df.columns = ["Prediction", "Count"]
        fig = px.bar(
            chart_df,
            x="Prediction",
            y="Count",
            color=chart_df["Prediction"].astype(str),
            title="Fast-Track Class Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------------------------------
    # 📋 Results Table + Download Button
    # --------------------------------------------------------------------------------------
    st.dataframe(raw.head(10))  # 👁️ Show first 10 rows in preview

    # 📥 Allow users to download full predictions
    st.download_button(
        "📥 Download Full Output",
        raw.to_csv(index=False),
        file_name="predictions_with_summary.csv"
    )

    # --------------------------------------------------------------------------------------
    # 💾 Save Results to Database (MySQL RDS)
    # --------------------------------------------------------------------------------------
    try:
        raw.to_sql("motor_claims_predictions", con=engine, if_exists="append", index=False)
        st.success("💾 Results saved to database successfully.")
    except Exception as e:
        st.error(f"❌ DB Save Failed: {str(e)}")

    # --------------------------------------------------------------------------------------
    # 📊 Executive Batch Summary using LLM (Optional)
    # --------------------------------------------------------------------------------------
    st.subheader("📊 Executive Batch Summary")

    # Use GenAI to generate a batch summary over the top 20 claims
    batch_summary = invoke_bedrock(
        f"Summarise this batch of motor insurance predictions:\n{raw.head(20).to_dict(orient='records')}"
    )
    st.markdown(batch_summary)

    # --------------------------------------------------------------------------------------
    # 📊 Dashboard Visual Overview Section
    # Shows multiple charts to help users interpret prediction results across dimensions.
    # --------------------------------------------------------------------------------------

    # 🔹 Horizontal divider
    st.markdown("---")

    # 📢 Section header
    st.subheader("📊 Visual Overview of Prediction Results")

    # Make a copy of the full dataset (with predictions and probabilities)
    view_df = raw.copy()

    # --------------------------------------------------------------------------------------
    # 1️⃣ Fast-Track Prediction Distribution (Bar Chart)
    # --------------------------------------------------------------------------------------

    # ✅ Count how many claims are classified as fast-tracked (1) vs not (0)
    prediction_counts = view_df["fast_track_prediction"].value_counts().reset_index()
    prediction_counts.columns = ["Prediction", "Count"]

    # 📊 Bar chart to show binary classification results
    fig1 = px.bar(
        prediction_counts,
        x="Prediction",
        y="Count",
        color="Prediction",
        title="✅ Fast-Track Prediction Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --------------------------------------------------------------------------------------
    # 2️⃣ Confidence Score Distribution (Pie Chart)
    # --------------------------------------------------------------------------------------

    # 📈 Bin probability scores into categories: Low, Medium, High, Very High
    conf_bins = pd.cut(
        view_df["confidence_score"],
        bins=[0, 0.5, 0.7, 0.85, 1],
        labels=["Low", "Medium", "High", "Very High"]
    )

    # 🔢 Count how many predictions fall in each confidence band
    conf_counts = conf_bins.value_counts().reset_index()
    conf_counts.columns = ["Confidence Level", "Count"]

    # 🥧 Pie chart to visualize confidence level distribution
    fig2 = px.pie(
        conf_counts,
        names="Confidence Level",
        values="Count",
        title="📍 Confidence Score Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --------------------------------------------------------------------------------------
    # 3️⃣ Vehicle Year vs Prediction (Box Plot)
    # --------------------------------------------------------------------------------------

    # 📦 Box plot to visualize relationship between vehicle year and fast-track label
    fig4 = px.box(
        view_df,
        x="fast_track_prediction",
        y="vehicle_year",
        points="all",  # Show outliers and data points
        title="🚘 Vehicle Year vs Prediction"
    )
    st.plotly_chart(fig4, use_container_width=True)

    # --------------------------------------------------------------------------------------
    # 4️⃣ Accident Location Type vs Prediction (Grouped Bar Chart)
    # --------------------------------------------------------------------------------------

    # 📍 If location type data exists, show breakdown by accident location type
    if "accident_location_type" in view_df.columns:
        # 📊 Count number of predictions for each location type + fast-track combo
        grouped = view_df.groupby(
            ["accident_location_type", "fast_track_prediction"]
        ).size().reset_index(name="count")

        # 🧱 Grouped bar chart for each location type and prediction class
        fig5 = px.bar(
            grouped,
            x="accident_location_type",
            y="count",
            color="fast_track_prediction",
            barmode="group",
            title="📍 Location Type vs Prediction"
        )
        st.plotly_chart(fig5, use_container_width=True)