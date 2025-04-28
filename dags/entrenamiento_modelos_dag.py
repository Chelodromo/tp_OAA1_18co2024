# dags/entrenamiento_modelos_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from tasks.entrenamiento_utils import (
    simple_mlflow_run,
    train_lightgbm_optuna_minio,
    train_randomforest_optuna_minio,
    train_logisticregression_optuna_minio,
    train_knn_optuna_minio
)

with DAG(
    dag_id="entrenamiento_modelos_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["entrenamiento", "optuna", "mlflow"]
) as dag:

    mlflow_test = PythonOperator(
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

    mlflow_test >> [train_lightgbm, train_randomforest, train_logisticregression, train_knn]
