# dags/descargar_dataset_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from tasks.s3_utils import ejemplo_conexion_s3, descargar_dataset

with DAG(
    dag_id="descargar_dataset_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["dataset", "minio"]
) as dag:

    conectar_minio = PythonOperator(
        task_id="conectar_minio",
        python_callable=ejemplo_conexion_s3
    )

    descargar_dataset_task = PythonOperator(
        task_id="descargar_dataset",
        python_callable=descargar_dataset
    )

    conectar_minio >> descargar_dataset_task
