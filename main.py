from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import io
import boto3  #  usamos boto3 directo para conectar a MinIO

# Conectarse a MinIO y cargar modelo ---

# Parámetros de conexión (ajustalos a tus valores reales si cambia algo)
minio_endpoint = "localhost:9000"  # o IP:puerto de tu MinIO
minio_access_key = "minioadmin"    # usuario
minio_secret_key = "minioadmin"    # contraseña
bucket_name = "respaldo"
modelo_key = "modelos/modelo_svm.pkl"

# Conexión boto3
s3_client = boto3.client(
    's3',
    endpoint_url=f"http://{minio_endpoint}",
    aws_access_key_id=minio_access_key,
    aws_secret_access_key=minio_secret_key,
    region_name="us-east-1",  # No importa la región en MinIO pero es requerido
)

# Descargar el modelo
response = s3_client.get_object(Bucket=bucket_name, Key=modelo_key)
modelo_bytes = response['Body'].read()

# Cargar modelo desde memoria
buffer = io.BytesIO(modelo_bytes)
modelo = joblib.load(buffer)

print("✅ Modelo cargado exitosamente desde MinIO para FastAPI.")

#  Inicializar FastAPI
app = FastAPI()

# Definir estructura de entrada para un solo registro
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

# Entrada completa: una lista de registros
class Entrada(BaseModel):
    caracteristicas: List[Caracteristica]

# Ruta para realizar predicción
@app.post("/predecir")
def predict(entrada: Entrada):
    # Convertir la entrada en un DataFrame
    datos = [carac.dict() for carac in entrada.caracteristicas]
    df = pd.DataFrame(datos)

    # Realizar predicción
    predicciones = modelo.predict(df)

    # Devolver predicciones como lista
    return {"predicciones": predicciones.tolist()}
