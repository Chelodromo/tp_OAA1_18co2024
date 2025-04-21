from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

def hacer_predicciones():
    import pandas as pd
    import joblib

    path = "/opt/airflow/datalake"
    modelo_path = os.path.join(path, "modelo_svm.pkl")
    nuevo_csv_path = os.path.join(path, "df_nuevo.csv")
    predicciones_csv_path = os.path.join(path, "nuevo_predicciones.csv")

    # Leer CSV excluyendo columnas problemáticas
    columnas_excluir = ['WDir', 'HWDir', 'Polvo_PM10']
    df_nuevo = pd.read_csv(
        nuevo_csv_path,
        usecols=lambda col: col not in columnas_excluir,
        parse_dates=['Date']
    )

    # Homogeneizar nombre de fecha
    if 'date' in df_nuevo.columns:
        df_nuevo = df_nuevo.rename(columns={'date': 'Date'})

    # Convertir fecha a timestamp numérico
    df_nuevo['Date_num'] = df_nuevo['Date'].apply(lambda x: x.timestamp())
    df_nuevo['Date_num'] = pd.to_numeric(df_nuevo['Date_num'], errors='coerce')

    # Eliminar columnas innecesarias para el modelo
    columnas_a_eliminar = ['Date', 'Punto', 'HiTemp', 'LowTemp', 'WTx', 'SolRate', 'SolRad.', 'arcInt']
    df_nuevo = df_nuevo.drop(columns=[c for c in columnas_a_eliminar if c in df_nuevo.columns])

    # Imputar valores nulos en columnas numéricas
    numeric_cols = df_nuevo.select_dtypes(include=['number']).columns
    df_nuevo[numeric_cols] = df_nuevo[numeric_cols].fillna(df_nuevo[numeric_cols].mean())

    # Cargar el modelo entrenado y predecir
    modelo = joblib.load(modelo_path)
    predicciones = modelo.predict(df_nuevo)

    # Guardar las predicciones en CSV
    df_nuevo['prediccion'] = predicciones
    df_nuevo.to_csv(predicciones_csv_path, index=False)
    print(f"✅ Archivo de predicciones guardado en: {predicciones_csv_path}")

# Definición del DAG
with DAG(
    dag_id="predicciones_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    prediccion_task = PythonOperator(
        task_id='realizar_predicciones',
        python_callable=hacer_predicciones
    )
