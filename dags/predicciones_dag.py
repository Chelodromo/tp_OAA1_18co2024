from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import joblib
import os

def realizar_predicciones():
    input_path = "/opt/airflow/datalake/df_nuevo.csv"
    model_path = "/opt/airflow/datalake/modelo_svm.pkl"
    output_path = "/opt/airflow/datalake/predicciones.csv"

    modelo = joblib.load(model_path)
    df_nuevo = pd.read_csv(input_path)

    if 'Date' in df_nuevo.columns:
        df_nuevo['Date_num'] = pd.to_datetime(df_nuevo['Date']).apply(lambda x: x.timestamp())
        df_nuevo = df_nuevo.drop(columns=['Date'])

    columnas_a_eliminar = ['Punto', 'HiTemp', 'LowTemp', 'WTx', 'SolRate', 'SolRad.', 'arcInt']
    df_nuevo = df_nuevo.drop(columns=[col for col in columnas_a_eliminar if col in df_nuevo.columns])

    predicciones = modelo.predict(df_nuevo)
    df_nuevo['Prediccion'] = predicciones

    df_nuevo.to_csv(output_path, index=False)
    print(f"✅ Predicciones guardadas en {output_path}")

with DAG(
    dag_id="predicciones_svm",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    predicciones_task = PythonOperator(
        task_id="realizar_predicciones",
        python_callable=realizar_predicciones
    )
