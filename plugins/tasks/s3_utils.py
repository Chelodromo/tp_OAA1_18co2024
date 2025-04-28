# plugins/tasks/s3_utils.py
import os
import tempfile
import pandas as pd
import requests
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from botocore.exceptions import ClientError

def ejemplo_conexion_s3():
    bucket_name = 'respaldo2'
    hook = S3Hook(aws_conn_id='minio_s3')
    s3_client = hook.get_conn()

    if not hook.check_for_bucket(bucket_name):
        try:
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"🪣 Bucket '{bucket_name}' creado correctamente.")
        except ClientError as e:
            print(f"❌ Error al crear bucket: {e}")
    else:
        print(f"✅ El bucket '{bucket_name}' ya existe.")
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key='prueba.txt',
        Body='Desde Airflow por variable de entorno'
    )
    print(f"📄 Archivo 'prueba.txt' subido a bucket '{bucket_name}'.")

def descargar_dataset(**kwargs):
    url = 'https://docs.google.com/uc?export=download&id=1gT8k90Iisd-sZVXWtS6Exl1ZFwwTd_WM'
    nombre_archivo_local = 'dataset.csv'
    bucket_name = 'respaldo2'
    s3_key = 'dataset.csv'
    hook = S3Hook(aws_conn_id='minio_s3')

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_path = os.path.join(tmpdirname, nombre_archivo_local)
        response = requests.get(url)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(response.content)
        hook.load_file(
            filename=local_path,
            key=s3_key,
            bucket_name=bucket_name,
            replace=True
        )
        print(f"✅ Dataset subido a MinIO en {bucket_name}/{s3_key}")

def leer_y_loguear_minio(**kwargs):
    # [código igual que tu versión actual]
    pass

def split_dataset_minio(**kwargs):
    # [código igual que tu versión actual]
    pass
