FROM python:3.9-slim

WORKDIR ~/mlops/lab9

COPY redditApp.py ./
COPY reddit_model_pipeline.joblib ./
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "redditApp:app", "--host", "0.0.0.0", "--port", "8000"]
