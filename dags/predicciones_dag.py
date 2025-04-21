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

    # Cargar modelo entrenado
    modelo = joblib.load(modelo_path)

    # Cargar datos nuevos
    df_nuevo = pd.read_csv(nuevo_csv_path, parse_dates=['Date'])

    # Procesamiento igual que en entrenamiento
    if 'date' in df_nuevo.columns:
        df_nuevo = df_nuevo.rename(columns={'date': 'Date'})

    df_nuevo['Date_num'] = df_nuevo['Date'].apply(lambda x: x.timestamp())
    df_nuevo['Date_num'] = pd.to_numeric(df_nuevo['Date_num'], errors='coerce')

    columnas_a_eliminar = ['Date', 'Punto', 'HiTemp', 'LowTemp', 'WTx', 'SolRate', 'SolRad.', 'arcInt']
    df_nuevo = df_nuevo.drop(columns=[c for c in columnas_a_eliminar if c in df_nuevo.columns])

    # Predecir
    predicciones = modelo.predict(df_nuevo)
    df_nuevo['prediccion'] = predicciones

    # Guardar resultado
    df_nuevo.to_csv(predicciones_csv_path, index=False)
    print(f"✅ Archivo de predicciones guardado en {predicciones_csv_path}")

# DAG definition
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
