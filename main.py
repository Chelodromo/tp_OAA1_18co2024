import boto3
import io
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd

#  Configuración fija para MinIO
minio_endpoint = "localhost:9000"
minio_access_key = "minioadmin"
minio_secret_key = "minioadmin"
bucket_name = "respaldo"
modelo_key = "modelos/modelo_svm.pkl"

#  Conectarse a MinIO y cargar el modelo
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

# --- 🚀 Inicializar FastAPI
app = FastAPI()

# Estructura de un solo registro
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

# Entrada de muchos registros
class Entrada(BaseModel):
    caracteristicas: List[Caracteristica]

# Ruta para predicción
@app.post("/predecir")
def predict(entrada: Entrada):
    datos = [carac.dict() for carac in entrada.caracteristicas]
    df = pd.DataFrame(datos)
    predicciones = modelo.predict(df)
    return {"predicciones": predicciones.tolist()}
