# dags/seleccion_modelo_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from tasks.prediccion_utils import seleccionar_mejor_modelo

with DAG(
    dag_id="seleccion_modelo_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["seleccion", "modelo", "mlflow"]
) as dag:

    seleccionar_modelo_task = PythonOperator(
        task_id="seleccionar_mejor_modelo",
        python_callable=seleccionar_mejor_modelo
    )
