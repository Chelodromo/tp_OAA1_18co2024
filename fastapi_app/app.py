# fastapi_app/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import boto3
import tempfile
import os
import pickle
from typing import List, Any
from datetime import datetime
from sklearn.base import BaseEstimator

app = FastAPI()

# Configuración de MinIO
MINIO_ENDPOINT   = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minio_admin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minio_admin')
BUCKET_NAME      = "respaldo2"
PREFIX           = "modelos/"

# Variables globales
model: Any = None
expected_features: List[str] = []

class PredictRequest(BaseModel):
    TempOut:   float
    DewPt:     float
    WSpeed:    float
    WHSpeed:   float
    Bar:       float
    Rain:      float
    ET:        float
    WDir_deg:  float
    Date_num:  float

def load_latest_model():
    global model
    s3 = boto3.client(
        's3',
        endpoint_url=f'http://{MINIO_ENDPOINT}',
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY
    )
    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
    files = [o['Key'] for o in resp.get('Contents', []) if o['Key'].endswith('.pkl')]
    if not files:
        raise RuntimeError("No se encontraron modelos en MinIO.")
    latest = sorted(files)[-1]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, os.path.basename(latest))
        s3.download_file(BUCKET_NAME, latest, tmp_path)
        with open(tmp_path, 'rb') as f:
            model = pickle.load(f)

@app.on_event("startup")
def startup_event():
    load_latest_model()
    # Capturamos los nombres de características que el modelo vio en fit()
    global expected_features
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
    print(f"✅ Modelo cargado: {type(model)}")
    print(f"🔧 Features esperadas: {expected_features}")

def align_and_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra columnas que coinciden tras quitar/poner un punto final y
    las reordena según expected_features.
    """
    df2 = df.copy()
    for feat in expected_features:
        if feat not in df2.columns:
            # buscar columna candidata con/ sin punto
            for col in df2.columns:
                if col.rstrip('.') == feat.rstrip('.'):
                    df2.rename(columns={col: feat}, inplace=True)
                    break
    # finalmente reordeno y devuelvo solo las esperadas
    return df2[expected_features]

def predict_proba_model(m: BaseEstimator, df: pd.DataFrame) -> pd.Series:
    """
    Unifica inferencia: siempre predict_proba[:,1] de sklearn estimators.
    """
    probs = m.predict_proba(df)
    return pd.Series(probs[:, 1], index=df.index)

@app.post("/predict")
def predict(data: PredictRequest):
    # 1) Cargo en DataFrame
    df = pd.DataFrame([data.dict()])
    # 2) Alineo nombres y orden
    df_aligned = align_and_order(df)
    # 3) Infiero
    proba = predict_proba_model(model, df_aligned)
    return {"probability": round(float(proba.iloc[0]), 4)}

@app.post("/predict_batch")
def predict_batch(data: List[PredictRequest]):
    df = pd.DataFrame([d.dict() for d in data])
    df_aligned = align_and_order(df)
    proba = predict_proba_model(model, df_aligned)
    dates = df_aligned["Date_num"].apply(lambda ts: datetime.fromtimestamp(ts).strftime("%d-%m-%Y"))
    results = [
        {"date": date, "probability": round(float(p), 4)}
        for date, p in zip(dates, proba)
    ]
    return {"predictions": results}
