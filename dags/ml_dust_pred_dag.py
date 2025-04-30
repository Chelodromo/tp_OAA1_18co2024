# dags/flujo_completo_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
# Importamos las tareas desde plugins/tasks/
from tasks.s3_utils import ejemplo_conexion_s3, descargar_dataset
from tasks.procesamiento_utils import leer_y_loguear_minio, split_dataset_minio
from tasks.entrenamiento_utils import (
    simple_mlflow_run,
    train_lightgbm_optuna_minio,
    train_randomforest_optuna_minio,
    train_logisticregression_optuna_minio,
    train_knn_optuna_minio
)
from tasks.prediccion_utils import seleccionar_mejor_modelo, predict_datos_actuales, test_endpoints_predict

def notificar_api_reload():
    url = "http://fastapi_app:8000/reload"
    try:
        response = requests.post(url)
        print(f"🔁 Respuesta de la API: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error al intentar recargar modelo: {e}")



# Definimos el DAG
with DAG(
    dag_id="flujo_completo_prediccion_polvo",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["pipeline", "mlops", "polvo"]
) as dag:

    conectar_minio = PythonOperator(
        task_id="conectar_minio",
        python_callable=ejemplo_conexion_s3
    )

    descargar_dataset_task = PythonOperator(
        task_id="descargar_dataset",
        python_callable=descargar_dataset
    )

    procesar_dataset_minio_task = PythonOperator(
        task_id="procesar_dataset_minio",
        python_callable=leer_y_loguear_minio
    )

    split_dataset_minio_task = PythonOperator(
        task_id="split_dataset_minio",
        python_callable=split_dataset_minio
    )

    mlflow_test_run = PythonOperator(
        task_id="mlflow_test_run",
        python_callable=simple_mlflow_run
    )

    train_lightgbm = PythonOperator(
        task_id="train_lightgbm",
        python_callable=train_lightgbm_optuna_minio
    )

    train_randomforest = PythonOperator(
        task_id="train_randomforest",
        python_callable=train_randomforest_optuna_minio
    )

    train_logisticregression = PythonOperator(
        task_id="train_logisticregression",
        python_callable=train_logisticregression_optuna_minio
    )

    train_knn = PythonOperator(
        task_id="train_knn",
        python_callable=train_knn_optuna_minio
    )

    seleccionar_modelo = PythonOperator(
        task_id="seleccionar_mejor_modelo",
        python_callable=seleccionar_mejor_modelo
    )

    predict_actual = PythonOperator(
        task_id="predict_datos_actuales",
        python_callable=predict_datos_actuales
    )

    test_endpoints = PythonOperator(
        task_id="test_fastapi_endpoints",
        python_callable=test_endpoints_predict
    )
    
    recargar_modelo_api = PythonOperator(
    task_id='recargar_modelo_api',
    python_callable=notificar_api_reload,
    dag=dag,
    )


    # Definimos el flujo de dependencias
    conectar_minio >> descargar_dataset_task >> procesar_dataset_minio_task >> split_dataset_minio_task >> mlflow_test_run
    mlflow_test_run >> [train_lightgbm, train_randomforest, train_logisticregression, train_knn]
    [train_lightgbm, train_randomforest, train_logisticregression, train_knn] >> seleccionar_modelo
    seleccionar_modelo >> recargar_modelo_api >> predict_actual >> test_endpoints
