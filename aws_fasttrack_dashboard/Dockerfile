# Dockerfile
FROM python:3.11

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "08_fasttrack_dashboard_genai.py", "--server.port=8501", "--server.address=0.0.0.0"]