import boto3
import io
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
from fastapi.responses import StreamingResponse

#  Configuración fija para MinIO
minio_endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minio_admin")
minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minio_admin")
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

# Definir el esquema de entrada para /predecir
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

# --- Endpoint 1️⃣: predicción normal
@app.post("/predecir")
def predict(entrada: Entrada):
    if modelo is None:
        return {"error": "Modelo no disponible. No se pudo cargar desde MinIO."}
    
    datos = [carac.dict() for carac in entrada.caracteristicas]
    df = pd.DataFrame(datos)
    predicciones = modelo.predict(df)
    return {"predicciones": predicciones.tolist()}

# --- Endpoint 2️⃣: predicción batch leyendo df_nuevo.csv desde MinIO
@app.get("/batch_predict")
def batch_predict():
    if modelo is None:
        return {"error": "Modelo no disponible. No se pudo cargar desde MinIO."}
    
    try:
        # 1️⃣ Parámetros fijos
        file_key = "modelos/df_nuevo.csv"

        # 2️⃣ Descargar el archivo desde MinIO
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response['Body'].read()

        # 3️⃣ Cargar CSV en un DataFrame
        df = pd.read_csv(io.BytesIO(file_content))
        
        # 4️⃣ Realizar las predicciones
        predicciones = modelo.predict(df)

        # 5️⃣ Agregar las predicciones al DataFrame
        df['prediccion'] = predicciones

        # 6️⃣ Preparar el CSV para enviar como respuesta
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        # 7️⃣ Devolver el CSV para descargar
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=df_nuevo_predicho.csv"}
        )

    except Exception as e:
        return {"error": f"Error en batch_predict: {str(e)}"}
