import boto3
import io
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd

#  Configuración fija para MinIO
minio_endpoint = "localhost:9000"
minio_access_key = "minio_admin"
minio_secret_key = "minio_admin"
bucket_name = "respaldo"
modelo_key = "modelos/modelo_svm.pkl"

#  Intentar conectar a MinIO y cargar el modelo
modelo = None
try:
    s3_client = boto3.client(
        's3',
        endpoint_url=f"http://{minio_endpoint}",
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        region_name="us-east-1",
    )

    response = s3_client.get_object(Bucket=bucket_name, Key=modelo_key)
    modelo_bytes = response['Body'].read()

    buffer = io.BytesIO(modelo_bytes)
    modelo = joblib.load(buffer)

    print(f"✅ Modelo cargado exitosamente desde MinIO ({minio_endpoint})")

except Exception as e:
    print(f"❌ Error cargando modelo desde MinIO: {e}")

# --- 🚀 Inicializar FastAPI
app = FastAPI()

class Caracteristica(BaseModel):
    TempOut: float
    OutHum: float
    DewPt_: float
    WSpeed: float
    WRun: float
    WHSpeed: float
    WChill: float
    HeatIx: float
    ThwIx: float
    ThswI: float
    Bar: float
    Rain: float
    HiSlE: float
    Rad_: float
    uvIndex: float
    uDose: float
    hiUV: float
    hetD_D: float
    colD_D: float
    inTemp: float
    inHum: float
    inDew: float
    iHeat: float
    ET: float
    WSamp: float
    iRecept: float
    WDir_deg: float
    HWDir_deg: float
    Date_num: float

class Entrada(BaseModel):
    caracteristicas: List[Caracteristica]

@app.post("/predecir")
def predict(entrada: Entrada):
    if modelo is None:
        return {"error": "Modelo no disponible. No se pudo cargar desde MinIO."}
    
    datos = [carac.dict() for carac in entrada.caracteristicas]
    df = pd.DataFrame(datos)
    predicciones = modelo.predict(df)
    return {"predicciones": predicciones.tolist()}
