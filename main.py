import boto3
import io
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import os

# --- 🔧 Configuración fija para MinIO
minio_endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minio_admin")
minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minio_admin")
bucket_name = "respaldo2"
modelo_folder = "best_model/"

# --- 🧠 Intentar conectar a MinIO y cargar el modelo más nuevo
modelo = None
try:
    s3_client = boto3.client(
        's3',
        endpoint_url=f"http://{minio_endpoint}",
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        region_name="us-east-1",
    )

    # Listar archivos en best_model/
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=modelo_folder)
    objetos = response.get('Contents', [])

    if not objetos:
        raise Exception(f"No se encontraron modelos en {modelo_folder}")

    # Ordenar por fecha de modificación (el más nuevo primero)
    objetos = sorted(objetos, key=lambda obj: obj['LastModified'], reverse=True)
    modelo_key = objetos[0]['Key']  # El más reciente

    # Descargar y cargar el modelo
    response = s3_client.get_object(Bucket=bucket_name, Key=modelo_key)
    modelo_bytes = response['Body'].read()

    buffer = io.BytesIO(modelo_bytes)
    modelo = joblib.load(buffer)

    print(f"✅ Modelo más nuevo cargado: {modelo_key}")

except Exception as e:
    print(f"❌ Error cargando modelo desde MinIO: {e}")

# --- 🚀 Inicializar FastAPI
app = FastAPI()

# --- 🧩 Esquema de Entrada
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

# --- 🔮 Servicio de predicción manual
@app.post("/predecir")
def predict(entrada: Entrada):
    if modelo is None:
        return {"error": "Modelo no disponible. No se pudo cargar desde MinIO."}
    
    datos = [carac.dict() for carac in entrada.caracteristicas]
    df = pd.DataFrame(datos)
    predicciones = modelo.predict(df)
    return {"predicciones": predicciones.tolist()}

# --- 📂 Servicio de batch_predict leyendo df_nuevo.json de MinIO
@app.get("/batch_predict")
def batch_predict():
    try:
        if modelo is None:
            return {"error": "Modelo no disponible."}

        s3_client = boto3.client(
            's3',
            endpoint_url=f"http://{minio_endpoint}",
            aws_access_key_id=minio_access_key,
            aws_secret_access_key=minio_secret_key,
            region_name="us-east-1",
        )

        # Leer el archivo df_nuevo.json
        response = s3_client.get_object(Bucket=bucket_name, Key="modelos/df_nuevo.json")
        df = pd.read_json(io.BytesIO(response['Body'].read()))

        # Predicción
        predicciones = modelo.predict(df)
        df['prediccion'] = predicciones

        # Guardar nuevo_predicciones.json
        buffer = io.BytesIO()
        df.to_json(buffer, orient="records")
        buffer.seek(0)

        s3_client.put_object(
            Bucket=bucket_name,
            Key="modelos/nuevo_predicciones.json",
            Body=buffer.getvalue()
        )

        return {"mensaje": "✅ Predicciones realizadas y archivo 'nuevo_predicciones.json' guardado en MinIO."}

    except Exception as e:
        return {"error": f"Error en batch_predict: {e}"}

