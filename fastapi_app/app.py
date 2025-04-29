# fastapi_app/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import lightgbm as lgb
import pandas as pd
import boto3
import tempfile
import os
import pickle
from fastapi import Body
from typing import List
from datetime import datetime

app = FastAPI()

# Config MinIO
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minio_admin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minio_admin')
BUCKET_NAME = "respaldo2"
PREFIX = "best_model/"

# Modelo global
model = None

class PredictRequest(BaseModel):
    TempOut: float
    DewPt.: float
    WSpeed: float
    WHSpeed: float
    Bar: float
    Rain: float
    ET: float
    WDir_deg: float
    Date_num: float

def load_latest_model():
    global model
    s3 = boto3.client('s3',
                      endpoint_url=f'http://{MINIO_ENDPOINT}',
                      aws_access_key_id=MINIO_ACCESS_KEY,
                      aws_secret_access_key=MINIO_SECRET_KEY)
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
    files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.pkl')]
    if not files:
        raise Exception("No model files found.")
    latest_file = sorted(files)[-1]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, os.path.basename(latest_file))
        s3.download_file(BUCKET_NAME, latest_file, tmp_path)
        with open(tmp_path, 'rb') as f:
            model = pickle.load(f)

@app.on_event("startup")
def startup_event():
    load_latest_model()

@app.post("/predict")
def predict(data: PredictRequest):
    df = pd.DataFrame([data.dict()])
    
    # Usamos predict_proba en lugar de predict
    proba = model.predict_proba(df)
    
    # Nos quedamos solo con la columna de la clase positiva (índice 1)
    proba_positive_class = proba[:, 1]
    
    return {"probability": proba_positive_class.tolist()}

@app.post("/predict_batch")
def predict_batch(data: List[PredictRequest]):
    df = pd.DataFrame([d.dict() for d in data])

    preds_proba = model.predict_proba(df)[:, 1]  # Clase positiva

    # Convertir Date_num a fecha
    dates = df['Date_num'].apply(lambda ts: datetime.fromtimestamp(ts).strftime('%d-%m-%Y'))

    # Preparar respuesta
    results = []
    for date, proba in zip(dates, preds_proba):
        results.append({
            "date": date,
            "probability": round(float(proba), 4)
        })

    return {"predictions": results}