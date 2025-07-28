# 🚗 Motor Claims Fast-Track Prediction with GenAI

This repository provides a full end-to-end machine learning and GenAI pipeline for predicting whether a motor insurance claim should be fast-tracked. It covers everything from data ingestion to model deployment and an interactive Streamlit dashboard powered by Amazon Bedrock (Mistral 7B).

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

## 🚀 Local Setup & Execution (End-to-End)

### 1️⃣ Create Python Environment

```bash
conda create -p env_Motor_Claim python=3.11 -y
conda activate /Users/gobinathsindhuja/Suncorp_task/env_Motor_Claim
pip install -r requirements.txt
```

### 2️⃣ Configure AWS RDS

Update MySQL RDS credentials in the scripts wherever `sqlalchemy.create_engine()` is used. Example:

```python
rds_uri = "mysql+pymysql://<user>:<password>@<host>/<db>"
engine = sqlalchemy.create_engine(rds_uri)
```

### 3️⃣ Run Data Pipeline in Order

```bash
python Scripts/01_data_ingestion_to_rds.py
python Scripts/02_data_cleaning_pipeline.py
python Scripts/04_feature_engineering_pipeline.py
python Scripts/05_feature_selection_pipeline.py
python Scripts/06_model_training_pipeline.py
python Scripts/07_selected_features_dump.py
```

### 4️⃣ Launch the Streamlit Dashboard

```bash
streamlit run Scripts/08_fasttrack_dashboard_genai.py
```

This will open a browser window at `http://localhost:8501`.

---

## 🧠 GenAI Integration (Amazon Bedrock - Mistral 7B)

The Streamlit dashboard connects to Amazon Bedrock to generate:

- Executive Claim Summary
- Risk Tags
- Next-Step Recommendation

### Setup

Ensure your AWS CLI or boto3 credentials are configured:

```bash
aws configure
# Or via ~/.aws/credentials or environment variables
```

Required permissions: `bedrock:InvokeModel`

---

## ☁️ EC2 Deployment Guide

### 1️⃣ Launch EC2 Instance

- Use Ubuntu 22.04 LTS
- Allow port 22 (SSH) and 8501 (Streamlit) in security group

### 2️⃣ SSH Into EC2

```bash
chmod 400 Motor_claim_key.pem
ssh -i Motor_claim_key.pem ubuntu@<ec2-public-ip>
```

### 3️⃣ Install Requirements

```bash
sudo apt update && sudo apt install -y python3-pip git docker.io
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Transfer Files from Local

```bash
scp -i Motor_claim_key.pem -r ./models ubuntu@<ec2-ip>:~/aws_fasttrack_dashboard/
scp -i Motor_claim_key.pem *.py requirements.txt ubuntu@<ec2-ip>:~/aws_fasttrack_dashboard/
```

### 5️⃣ Run Streamlit App

```bash
cd aws_fasttrack_dashboard
streamlit run 08_fasttrack_dashboard_genai.py --server.port 8501 --server.enableCORS false
```

Visit `http://<ec2-public-ip>:8501` in your browser.

---

## 📦 Output

- CSV with predictions, probabilities, LLM summaries and recommendations
- Records saved to RDS (`motor_claims_predictions` table)
- Interactive visual dashboard

---

## 📬 Author

Developed by **Gobinath Subramain**  
Use case: Automating motor claims triage using ML + GenAI.