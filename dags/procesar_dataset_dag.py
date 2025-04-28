# dags/procesar_dataset_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from tasks.procesamiento_utils import leer_y_loguear, split_dataset, svm_modeling
from tasks.s3_utils import leer_y_loguear_minio, split_dataset_minio

with DAG(
    dag_id="procesar_dataset_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["procesamiento", "split"]
) as dag:

    mostrar_head = PythonOperator(
        task_id="mostrar_head",
        python_callable=leer_y_loguear
    )

    split_dataset_task = PythonOperator(
        task_id="split_dataset",
        python_callable=split_dataset
    )

    svm_modeling_task = PythonOperator(
        task_id="svm_modeling",
        python_callable=svm_modeling
    )

    procesar_dataset_minio = PythonOperator(
        task_id="procesar_dataset_minio",
        python_callable=leer_y_loguear_minio
    )

    split_dataset_minio_task = PythonOperator(
        task_id="split_dataset_minio",
        python_callable=split_dataset_minio
    )

    mostrar_head >> split_dataset_task >> svm_modeling_task
    procesar_dataset_minio >> split_dataset_minio_task
