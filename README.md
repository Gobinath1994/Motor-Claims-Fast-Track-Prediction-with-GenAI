
# 🚗 Motor Claims Fast-Track Prediction with GenAI

This repository provides a complete end-to-end machine learning and GenAI pipeline to predict whether a motor insurance claim should be fast-tracked. It includes raw data ingestion, preprocessing, NLP-based feature engineering, ensemble feature selection, model training & evaluation, and a live Streamlit dashboard powered by Amazon Bedrock's Mistral 7B.

---

## 📁 Project Structure

```
.
├── 01_data_ingestion_to_rds.py         # Load raw claims data into AWS RDS
├── 02_data_cleaning_pipeline.py        # Clean and preprocess claims
├── 03_data_exploration_pipeline.ipynb  # Visual data exploration and profiling
├── 04_feature_engineering_pipeline.py  # Add domain features, TF-IDF, embeddings, PCA, KMeans
├── 05_feature_selection_pipeline.py    # Select top features using ensemble techniques
├── 06_model_training_pipeline.py       # Train, tune and evaluate ML models
├── 07_selected_features_dump.py        # Save selected features list to disk
├── 08_fasttrack_dashboard_genai.py     # Streamlit dashboard for prediction and GenAI reasoning
├── models/                             # Stores trained models and transformers
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

## 🚀 Local Execution Guide (End-to-End)

### 1️⃣ Create Conda Environment

```bash
conda create -p env_Motor_Claim python=3.11 -y
conda activate /Users/gobinathsindhuja/Suncorp_task/env_Motor_Claim
pip install -r requirements.txt
```

### 2️⃣ Configure AWS RDS Connection

Update MySQL RDS credentials in scripts that use:

```python
sqlalchemy.create_engine("mysql+pymysql://<user>:<password>@<host>/<db>")
```

### 3️⃣ Run the Data + Modeling Pipeline in Order

```bash
python Scripts/01_data_ingestion_to_rds.py
python Scripts/02_data_cleaning_pipeline.py
python Scripts/04_feature_engineering_pipeline.py
python Scripts/05_feature_selection_pipeline.py
python Scripts/06_model_training_pipeline.py
python Scripts/07_selected_features_dump.py
```

### 4️⃣ Launch Streamlit Dashboard (Locally)

```bash
streamlit run Scripts/08_fasttrack_dashboard_genai.py
```

Access: http://localhost:8501

---

## 🧠 GenAI Integration (Amazon Bedrock - Mistral 7B)

The dashboard connects to Amazon Bedrock to generate:

- 📋 Executive Claim Summary  
- ⚠️ Risk Tags  
- 💡 Next-Step Recommendation

### Setup Instructions

```bash
aws configure
```

Or set credentials in `~/.aws/credentials`.

Required permission: `bedrock:InvokeModel`

---

## ☁️ EC2 Deployment Instructions

### 1️⃣ Launch EC2 Instance

- Choose Ubuntu 22.04 LTS
- Open ports: `22` (SSH) and `8501` (Streamlit)

### 2️⃣ SSH into EC2

```bash
chmod 400 Motor_claim_key.pem
ssh -i Motor_claim_key.pem ubuntu@<ec2-public-ip>
```

### 3️⃣ Install Requirements on EC2

```bash
sudo apt update && sudo apt install -y python3-pip git docker.io
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Transfer Project Files from Local

```bash
scp -i Motor_claim_key.pem -r ./models ubuntu@<ec2-ip>:~/aws_fasttrack_dashboard/
scp -i Motor_claim_key.pem *.py requirements.txt ubuntu@<ec2-ip>:~/aws_fasttrack_dashboard/
```

### 5️⃣ Run Streamlit App on EC2

```bash
cd ~/aws_fasttrack_dashboard
streamlit run 08_fasttrack_dashboard_genai.py --server.port 8501 --server.enableCORS false
```

Then visit: http://<your-ec2-ip>:8501

---

## 📦 Output

- ✅ CSV with predictions, probabilities, LLM summaries and tags
- ✅ Records stored in AWS RDS (`motor_claims_predictions`)
- ✅ Visual dashboard with SHAP insights and GenAI explanations

---

## 🧪 Techniques Used

- SMOTE Oversampling for class imbalance
- PCA for dimensionality reduction
- KMeans clustering on embeddings
- Sentence-BERT (MiniLM) for semantic text features
- TF-IDF for sparse textual features
- Ensemble feature voting (MI, Boruta, LGBM, L1 Logistic)
- Model tuning using Optuna
- Explainability via SHAP
- Amazon Bedrock for LLM summarization and tagging

---

## 🛡️ .gitignore Setup

To avoid uploading unnecessary data or environment artifacts, use:

```
# Ignore raw data and sensitive files
data/
*.csv
*.xlsx
*.docx

# Model outputs and logs
catboost_info/
*.joblib
*.pkl
*.png

# Python envs
env_*/
__pycache__/
.ipynb_checkpoints/

# AWS credentials
*.pem
```

---

## 📬 Author

**Gobinath Subramain**  
Email: g.subramani@uqconnect.edu.au
Use Case: Automating motor claims triage using ML + GenAI   
