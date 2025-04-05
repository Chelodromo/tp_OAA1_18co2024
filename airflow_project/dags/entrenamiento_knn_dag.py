import datetime
from airflow.decorators import dag, task

default_args = {
    'start_date': datetime.datetime(2024, 1, 1),
}

@dag(
    dag_id="entrenamiento_knn",
    default_args=default_args,
    schedule_interval=None,
    catchup=False
)
def pipeline_knn():

    @task()
    def cargar_y_preparar_datos():
        import pandas as pd
        from DA_KNN import preparar_datos_knn

        df_polvo = pd.read_csv('/opt/airflow/dags/data/df_merged.csv')
        df_polvo['Date'] = pd.to_datetime(df_polvo['Date'])
        df_polvo.drop(columns=['WDir','HWDir', 'Polvo_PM10' ], inplace=True)
        numerical_cols = df_polvo.select_dtypes(include=['number']).columns
        df_polvo[numerical_cols] = df_polvo[numerical_cols].fillna(df_polvo[numerical_cols].mean())

        X_train, X_test, y_train, y_test = preparar_datos_knn(df_polvo)

        # Guardar datos procesados en CSV
        X_train.to_csv('/opt/airflow/dags/data/X_train.csv', index=False)
        X_test.to_csv('/opt/airflow/dags/data/X_test.csv', index=False)
        y_train.to_csv('/opt/airflow/dags/data/y_train.csv', index=False)
        y_test.to_csv('/opt/airflow/dags/data/y_test.csv', index=False)

        return "ok"

    @task()
    def entrenar_y_evaluar_modelo(flag):
        if flag == "ok":
            import pandas as pd
            from KNN import KNNTrainer

            # Cargar los datos escalados
            X_train = pd.read_csv('/opt/airflow/dags/data/X_train.csv')
            X_test = pd.read_csv('/opt/airflow/dags/data/X_test.csv')
            y_train = pd.read_csv('/opt/airflow/dags/data/y_train.csv').squeeze()
            y_test = pd.read_csv('/opt/airflow/dags/data/y_test.csv').squeeze()

            # Entrenar
            trainer = KNNTrainer()
            best_params = trainer.train_with_random_search(X_train, y_train, {
                "n_neighbors": range(1, 15),
                "weights": ["uniform", "distance"],
                "p": range(1, 10)
            })
            print("Mejores parámetros:", best_params)

            # Evaluar
            trainer.evaluar_modelo(X_test, y_test, nombre="KNN Random Search")

    # Encadenar tareas
    datos_ok = cargar_y_preparar_datos()
    entrenar_y_evaluar_modelo(datos_ok)

dag = pipeline_knn()
