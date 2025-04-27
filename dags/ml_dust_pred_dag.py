from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from botocore.exceptions import ClientError
import os

## Minio bucket
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime

import tempfile
import requests
import os
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from sklearn.model_selection import train_test_split

import os
import tempfile
import pandas as pd
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


def ejemplo_conexion_s3():
    bucket_name = 'respaldo2'
    hook = S3Hook(aws_conn_id='minio_s3')
    s3_client = hook.get_conn()
    # Verificar si el bucket ya existe
    if not hook.check_for_bucket(bucket_name):
        try:
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"🪣 Bucket '{bucket_name}' creado correctamente.")
        except ClientError as e:
            print(f"❌ Error al crear bucket: {e}")
    else:
        print(f"✅ El bucket '{bucket_name}' ya existe. No se necesita crear.")
    # Subir archivo de prueba
    s3_client.put_object(
        Bucket=bucket_name,
        Key='prueba.txt',
        Body='Desde Airflow por variable de entorno'
    )
    print(f"📄 Archivo 'prueba.txt' subido a bucket '{bucket_name}'.")



# Ruta al archivo dentro del contenedor
DATA_PATH = "/opt/airflow/datalake/df_merged.csv"
backup_path = "/opt/airflow/datalake/df_procesado.json"

output_dir = "/opt/airflow/datalake"

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
    
    # Guardar como JSON
    X_train_svm.to_json(os.path.join(output_dir, "X_train.json"), orient="records")
    X_test_svm.to_json(os.path.join(output_dir, "X_test.json"), orient="records")
    y_train_svm.to_json(os.path.join(output_dir, "y_train.json"), orient="records")
    y_test_svm.to_json(os.path.join(output_dir, "y_test.json"), orient="records")

    print("✅ Split realizado y archivos guardados en /opt/airflow/datalake.")
    print("📦 Archivos creados: X_train.json, X_test.json, y_train.json, y_test.json")


def svm_modeling():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import confusion_matrix

    # Cargar los datos desde los JSON
    path = "/opt/airflow/datalake"
    X_train_svm = pd.read_json(f"{path}/X_train.json")
    X_test_svm = pd.read_json(f"{path}/X_test.json")
    y_train_svm = pd.read_json(f"{path}/y_train.json", typ='series')
    y_test_svm = pd.read_json(f"{path}/y_test.json", typ='series')

    # Escalar
    scaler = StandardScaler()
    X_train_svm = scaler.fit_transform(X_train_svm)
    X_test_svm = scaler.transform(X_test_svm)

    # Modelo SVM
    svm_linear = SVC(C=0.001, kernel='linear')
    svm_linear.fit(X_train_svm, y_train_svm)

    # Validación cruzada
    scores = cross_val_score(svm_linear, X_train_svm, y_train_svm, cv=5, scoring='accuracy')
    print("\n🔁 Cross-validation scores:", scores)
    print("📊 Cross-validation mean accuracy:", scores.mean())

    # Predicciones
    y_pred_svm = svm_linear.predict(X_test_svm)

    # Matriz de confusión
    print("\n🔍 Confusion Matrix:")
    print(confusion_matrix(y_test_svm, y_pred_svm))


def descargar_dataset(**kwargs):
    # URL del archivo a descargar
    url = 'https://docs.google.com/uc?export=download&id=1gT8k90Iisd-sZVXWtS6Exl1ZFwwTd_WM'
    nombre_archivo_local = 'dataset.csv'
    bucket_name = 'respaldo2'
    s3_key = 'dataset.csv'  # Nombre que tendrá en MinIO

    # Crear directorio temporal
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_path = os.path.join(tmpdirname, nombre_archivo_local)

        # Descargar archivo
        print(f"📥 Descargando archivo desde {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Levanta error si falla
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Archivo descargado en {local_path}")

        # Conectar con MinIO usando conexión configurada en Airflow
        print(f"🚀 Subiendo {nombre_archivo_local} al bucket '{bucket_name}' en MinIO...")
        hook = S3Hook(aws_conn_id='minio_s3')
        hook.load_file(
            filename=local_path,
            key=s3_key,
            bucket_name=bucket_name,
            replace=True  # Reemplaza si ya existe
        )
        print(f"✅ Archivo {nombre_archivo_local} subido correctamente a {bucket_name}/{s3_key}")


def leer_y_loguear_minio(**kwargs):
    bucket_name = 'respaldo2'
    input_key = 'dataset.csv'
    output_key = 'dataset_backup.json'
    aws_conn_id = 'minio_s3'

    hook = S3Hook(aws_conn_id=aws_conn_id)

    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f"📁 Usando directorio temporal: {tmpdirname}")

        # Definimos ruta completa donde vamos a guardar el CSV
        input_local_path = os.path.join(tmpdirname, os.path.basename(input_key))

        # 📥 Descargar el contenido como string y guardarlo manualmente
        file_content = hook.read_key(key=input_key, bucket_name=bucket_name)

        # Guardarlo en el archivo temporal
        with open(input_local_path, 'w', encoding='utf-8') as f:
            f.write(file_content)

        print(f"✅ CSV descargado manualmente en: {input_local_path}")

        # Leer CSV
        columnas_excluir = ['WDir', 'HWDir', 'Polvo_PM10']
        df = pd.read_csv(
            input_local_path,
            usecols=lambda col: col not in columnas_excluir,
            parse_dates=['Date']
        )

        print("\n🧠 Primeras filas del dataset:")
        print(df.head())
        print(df.info())

        # Imputar nulos
        numerical_cols = df.select_dtypes(include=['number']).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

        print("\n🧠 Dataset luego de imputar nulos:")
        print(df.info())

        # Guardar backup
        output_local_path = os.path.join(tmpdirname, 'backup.json')
        df.to_json(output_local_path, orient="records", date_format="iso")
        print(f"\n💾 Backup generado en {output_local_path}")

        # Subir backup a MinIO
        hook.load_file(
            filename=output_local_path,
            key=output_key,
            bucket_name=bucket_name,
            replace=True
        )
        print(f"✅ Backup subido a MinIO en {bucket_name}/{output_key}")

def split_dataset_minio(**kwargs):
    bucket_name = 'respaldo2'  # <-- Tu bucket en MinIO
    input_key = 'dataset_backup.json'  # <-- JSON procesado que ya está guardado
    aws_conn_id = 'minio_s3'  # Conexión de Airflow a MinIO

    hook = S3Hook(aws_conn_id=aws_conn_id)

    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f"📁 Usando directorio temporal: {tmpdirname}")

        # Descargar el JSON de MinIO
        input_local_path = os.path.join(tmpdirname, os.path.basename(input_key))
        file_content = hook.read_key(key=input_key, bucket_name=bucket_name)
        
        with open(input_local_path, 'w', encoding='utf-8') as f:
            f.write(file_content)

        print(f"✅ JSON descargado en: {input_local_path}")

        # Leer el JSON como DataFrame
        df_polvo_svm = pd.read_json(input_local_path)

        # Procesar columnas
        if 'date' in df_polvo_svm.columns:
            df_polvo_svm = df_polvo_svm.rename(columns={'date': 'Date'})
        
        df_polvo_svm['Date_num'] = df_polvo_svm['Date'].apply(lambda x: pd.to_datetime(x).timestamp())
        df_polvo_svm['Date_num'] = pd.to_numeric(df_polvo_svm['Date_num'], errors='coerce')

        # Eliminar columnas irrelevantes
        columnas_a_eliminar = ['Date', 'Punto', 'HiTemp', 'LowTemp', 'WTx', 'SolRate', 'SolRad.', 'arcInt']
        df_polvo_svm = df_polvo_svm.drop(columns=[c for c in columnas_a_eliminar if c in df_polvo_svm.columns])

        print(df_polvo_svm.info())

        # Separar features y target
        X = df_polvo_svm.drop(columns=['clase'])
        y = df_polvo_svm['clase']

        # Realizar el split
        X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
            X, y, stratify=y, test_size=0.3, random_state=42
        )

        print("✅ Split realizado con éxito.")
        print("📊 Tamaño del conjunto de entrenamiento:", len(X_train_svm))
        print("📈 Tamaño del conjunto de prueba:", len(X_test_svm))

        # Guardar los splits en archivos temporales
        splits = {
            "X_train.json": X_train_svm,
            "X_test.json": X_test_svm,
            "y_train.json": y_train_svm,
            "y_test.json": y_test_svm
        }

        for filename, df_split in splits.items():
            local_split_path = os.path.join(tmpdirname, filename)
            df_split.to_json(local_split_path, orient="records")

            # Subir a MinIO
            hook.load_file(
                filename=local_split_path,
                key=f'splits/{filename}',
                bucket_name=bucket_name,
                replace=True
            )
            print(f"✅ Archivo {filename} subido a MinIO en splits/{filename}")
            
def svm_modeling_minio(**kwargs):
    bucket_name = 'respaldo2'  # Tu bucket en MinIO
    splits_path = 'splits'     # Carpeta donde están los splits
    aws_conn_id = 'minio_s3'    # Conexión Airflow-MinIO

    hook = S3Hook(aws_conn_id=aws_conn_id)

    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f"📁 Usando directorio temporal: {tmpdirname}")

        # Archivos a descargar
        archivos = ['X_train.json', 'X_test.json', 'y_train.json', 'y_test.json']
        rutas_locales = {}

        for archivo in archivos:
            s3_key = f'{splits_path}/{archivo}'
            local_path = os.path.join(tmpdirname, archivo)

            file_content = hook.read_key(key=s3_key, bucket_name=bucket_name)
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(file_content)
            rutas_locales[archivo] = local_path

            print(f"✅ Archivo {archivo} descargado en {local_path}")

        # Cargar los datos
        X_train_svm = pd.read_json(rutas_locales['X_train.json'])
        X_test_svm = pd.read_json(rutas_locales['X_test.json'])
        y_train_svm = pd.read_json(rutas_locales['y_train.json'], typ='series')
        y_test_svm = pd.read_json(rutas_locales['y_test.json'], typ='series')

        # Escalar features
        scaler = StandardScaler()
        X_train_svm = scaler.fit_transform(X_train_svm)
        X_test_svm = scaler.transform(X_test_svm)

        # Crear y entrenar el modelo SVM
        svm_linear = SVC(C=0.001, kernel='linear')
        svm_linear.fit(X_train_svm, y_train_svm)

        # Validación cruzada
        scores = cross_val_score(svm_linear, X_train_svm, y_train_svm, cv=5, scoring='accuracy')
        print("\n🔁 Cross-validation scores:", scores)
        print("📊 Cross-validation mean accuracy:", scores.mean())

        # Predicciones en test
        y_pred_svm = svm_linear.predict(X_test_svm)

        # Matriz de confusión
        print("\n🔍 Confusion Matrix:")
        print(confusion_matrix(y_test_svm, y_pred_svm))           

# DAG definition
with DAG(
    dag_id="descargar_y_ver_dataset",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    
    probar_minio =  PythonOperator(
        task_id='conectar_minio',
        python_callable=ejemplo_conexion_s3)


    descargar_csv = BashOperator(
        task_id='descargar_csv',
        bash_command=(
            f"mkdir -p /opt/airflow/datalake && "
            f"curl -L -o {DATA_PATH} "
            "'https://docs.google.com/uc?export=download&id=1gT8k90Iisd-sZVXWtS6Exl1ZFwwTd_WM'"
        )
    )

    descargar_dataset = PythonOperator(
        task_id='descargar_dataset',
        python_callable=descargar_dataset,
        provide_context=True  # Si usás **kwargs, esto es necesario
    )

    mostrar_head = PythonOperator(
        task_id='mostrar_head',
        python_callable=leer_y_loguear
    )
    
    split_dataset_task = PythonOperator(
    task_id='split_dataset',
    python_callable=split_dataset
)
    svm_modeling_task = PythonOperator(
    task_id='svm_modeling',
    python_callable=svm_modeling
)
    
    procesar_dataset = PythonOperator(
        task_id='procesar_dataset',
        python_callable=leer_y_loguear_minio,
        provide_context=True,
    )
    
    split_dataset_task_minio = PythonOperator(
    task_id='split_dataset_minio',
    python_callable=split_dataset_minio,
    provide_context=True,
    dag=dag
)
    svm_modeling_task_minio = PythonOperator(
    task_id='svm_modeling_minio',
    python_callable=svm_modeling_minio,
    provide_context=True,
    dag=dag
)

    probar_minio >> descargar_dataset >> mostrar_head >> split_dataset_task >> svm_modeling_task >> procesar_dataset >> split_dataset_task_minio >> svm_modeling_task_minio
    
