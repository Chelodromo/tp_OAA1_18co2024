# plugins/tasks/procesamiento_utils.py
import os
import tempfile
import pandas as pd
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from sklearn.model_selection import train_test_split

# Rutas locales
DATA_PATH = "/opt/airflow/datalake/df_merged.csv"
backup_path = "/opt/airflow/datalake/df_procesado.json"
output_dir = "/opt/airflow/datalake"

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

    numerical_cols = df.select_dtypes(include=['number']).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    print("\n🧠 Check Nulos después de imputar:")
    print(df.info())

    df.to_json(backup_path, orient="records", date_format="iso")
    print(f"\n💾 Dataset respaldado como JSON en: {backup_path}")

def leer_y_loguear_minio(**kwargs):
    bucket_name = 'respaldo2'
    input_key = 'dataset.csv'
    output_key = 'dataset_backup.json'
    aws_conn_id = 'minio_s3'

    hook = S3Hook(aws_conn_id=aws_conn_id)

    with tempfile.TemporaryDirectory() as tmpdirname:
        input_local_path = os.path.join(tmpdirname, os.path.basename(input_key))
        file_content = hook.read_key(key=input_key, bucket_name=bucket_name)

        with open(input_local_path, 'w', encoding='utf-8') as f:
            f.write(file_content)

        df = pd.read_csv(
            input_local_path,
            usecols=lambda col: col not in ['WDir', 'HWDir', 'Polvo_PM10'],
            parse_dates=['Date']
        )

        print("\n🧠 Primeras filas del dataset descargado:")
        print(df.head())
        print(df.info())

        numerical_cols = df.select_dtypes(include=['number']).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

        output_local_path = os.path.join(tmpdirname, 'backup.json')
        df.to_json(output_local_path, orient="records", date_format="iso")

        hook.load_file(
            filename=output_local_path,
            key=output_key,
            bucket_name=bucket_name,
            replace=True
        )
        print(f"✅ Backup generado y subido como {output_key}")

def split_dataset_minio(**kwargs):
    bucket_name = 'respaldo2'
    input_key = 'dataset_backup.json'
    aws_conn_id = 'minio_s3'

    hook = S3Hook(aws_conn_id=aws_conn_id)

    with tempfile.TemporaryDirectory() as tmpdirname:
        input_local_path = os.path.join(tmpdirname, os.path.basename(input_key))
        file_content = hook.read_key(key=input_key, bucket_name=bucket_name)

        with open(input_local_path, 'w', encoding='utf-8') as f:
            f.write(file_content)

        df_polvo_svm = pd.read_json(input_local_path)

        if 'date' in df_polvo_svm.columns:
            df_polvo_svm = df_polvo_svm.rename(columns={'date': 'Date'})

        df_polvo_svm['Date_num'] = df_polvo_svm['Date'].apply(lambda x: pd.to_datetime(x).timestamp())
        df_polvo_svm['Date_num'] = pd.to_numeric(df_polvo_svm['Date_num'], errors='coerce')

        columnas_a_eliminar = [
            'Date','Punto', 'HiTemp', 'LowTemp', 'WTx', 'SolRate', 'SolRad.', 'arcInt',
            'OutHum', 'WRun', 'WChill', 'HeatIx', 'ThwIx', 'ThswI', 'HiSlE', 'Rad.',
            'uvIndex', 'uDose', 'hiUV', 'hetD-D', 'colD-D', 'inTemp', 'inHum',
            'inDew', 'iHeat', 'WSamp', 'iRecept', 'HWDir_deg'
        ]
        df_polvo_svm = df_polvo_svm.drop(columns=[c for c in columnas_a_eliminar if c in df_polvo_svm.columns])

        X = df_polvo_svm.drop(columns=['clase'])
        y = df_polvo_svm['clase']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.3, random_state=42
        )

        splits = {
            "X_train.json": X_train,
            "X_test.json": X_test,
            "y_train.json": y_train,
            "y_test.json": y_test
        }

        for filename, df_split in splits.items():
            local_split_path = os.path.join(tmpdirname, filename)
            df_split.to_json(local_split_path, orient="records")

            hook.load_file(
                filename=local_split_path,
                key=f'splits/{filename}',
                bucket_name=bucket_name,
                replace=True
            )
            print(f"✅ Split {filename} subido a MinIO")
