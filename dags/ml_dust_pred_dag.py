from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd

# Ruta al archivo dentro del contenedor
DATA_PATH = "/opt/airflow/datalake/df_merged.csv"
backup_path = "/opt/airflow/datalake/df_procesado.json"


# Función para leer y loguear el head
def leer_y_loguear():
    columnas_excluir = ['WDir', 'HWDir', 'Polvo_PM10']
    df = pd.read_csv(
        DATA_PATH,
        usecols=lambda col: col not in columnas_excluir,
        parse_dates=['Date']
    )
    print("\n🧠 Primeras filas del dataset:")
    print(df.head())
    print(df.info())
    # Imputar los valores nulos únicamente en las columnas numéricas
    numerical_cols = df.select_dtypes(include=['number']).columns  # Seleccionar solo columnas numéricas
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    print("\n🧠 Check Nulos:")
    print(df.info())
    # Guardar respaldo como JSON
    df.to_json(backup_path, orient="records", date_format="iso")
    print(f"\n💾 Dataset respaldado como JSON en: {backup_path}")


def split_dataset():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    json_path = "/opt/airflow/datalake/df_procesado.json"

    # Leer JSON
    df_polvo_svm = pd.read_json(json_path)

    # Asegurar que la columna 'date' esté en datetime y renombrarla a 'Date' si fuera necesario
    if 'date' in df_polvo_svm.columns:
        df_polvo_svm = df_polvo_svm.rename(columns={'date': 'Date'})

    df_polvo_svm['Date_num'] = df_polvo_svm['Date'].apply(lambda x: x.timestamp())
    df_polvo_svm['Date_num'] = pd.to_numeric(df_polvo_svm['Date_num'], errors='coerce')

    # Drop columnas innecesarias
    columnas_a_eliminar = ['Date', 'Punto', 'HiTemp', 'LowTemp', 'WTx', 'SolRate', 'SolRad.', 'arcInt']
    df_polvo_svm = df_polvo_svm.drop(columns=[c for c in columnas_a_eliminar if c in df_polvo_svm.columns])

    print(df_polvo_svm.info())
    # Separar features y target
    X = df_polvo_svm.drop(columns=['clase'])
    print(X.info())
    y = df_polvo_svm.iloc[:, -2]
    print(y)

    # Split con stratify
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    print("✅ Split realizado con éxito.")
    print("📊 Tamaño del conjunto de entrenamiento:", len(X_train_svm))
    print("📈 Tamaño del conjunto de prueba:", len(X_test_svm))



# DAG definition
with DAG(
    dag_id="descargar_y_ver_dataset",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    descargar_csv = BashOperator(
        task_id='descargar_csv',
        bash_command=(
            f"mkdir -p /opt/airflow/datalake && "
            f"curl -L -o {DATA_PATH} "
            "'https://docs.google.com/uc?export=download&id=1gT8k90Iisd-sZVXWtS6Exl1ZFwwTd_WM'"
        )
    )

    mostrar_head = PythonOperator(
        task_id='mostrar_head',
        python_callable=leer_y_loguear
    )
    
    split_dataset_task = PythonOperator(
    task_id='split_dataset',
    python_callable=split_dataset
)


    descargar_csv >> mostrar_head >> split_dataset_task
