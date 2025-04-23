from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib

# Cargar el modelo desde la carpeta datalake
modelo = joblib.load("datalake/modelo_svm.pkl")

# Inicializar FastAPI
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
