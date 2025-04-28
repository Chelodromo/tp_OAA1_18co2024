# plugins/tasks/prediccion_utils.py
import os
import tempfile
import pickle
import requests
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

def seleccionar_mejor_modelo(**kwargs):
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()

    experiment_names = [
        "lightgbm_experiment",
        "randomforest_experiment",
        "logisticregression_experiment",
        "knn_experiment"
    ]

    best_score = -1
    best_experiment_name = None
    best_run_id = None

    for exp_name in experiment_names:
        experiment = client.get_experiment_by_name(exp_name)
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.recall DESC"],
                max_results=1
            )
            if runs:
                top_run = runs[0]
                recall = top_run.data.metrics.get('recall', 0)
                print(f"🔍 {exp_name}: Recall = {recall:.4f}")
                if recall > best_score:
                    best_score = recall
                    best_experiment_name = exp_name
                    best_run_id = top_run.info.run_id

    if not best_run_id:
        raise Exception("❌ No se encontró ningún modelo entrenado.")

    print(f"\n🏆 Mejor modelo: {best_experiment_name} con Recall: {best_score:.4f}")
    print(f"📦 Run ID: {best_run_id}")

    bucket_name = 'respaldo2'
    hook = S3Hook(aws_conn_id='minio_s3')

    with tempfile.TemporaryDirectory() as tmpdir:
        modelo_name = best_experiment_name.replace('_experiment', '')
        key_modelo = f"modelos/{modelo_name}_model.pkl"
        local_model_path = os.path.join(tmpdir, f"{modelo_name}_model.pkl")

        obj = hook.get_key(key=key_modelo, bucket_name=bucket_name)
        with open(local_model_path, 'wb') as f:
            f.write(obj.get()['Body'].read())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_model_key = f"best_model/{modelo_name}_{timestamp}.pkl"

        hook.load_file(
            filename=local_model_path,
            key=best_model_key,
            bucket_name=bucket_name,
            replace=True
        )
        print(f"🚀 Mejor modelo subido como {best_model_key}")

def predict_datos_actuales(**kwargs):
    user = "ricardoq"
    pswd = "eLxdr3FZ51DE"
    auth_url = 'https://tca-ssrm.com/api/auth'
    payload = {'username': user, 'password': pswd}

    auth_response = requests.post(auth_url, data=payload)
    auth_response.raise_for_status()
    auth_token = auth_response.json()['token']
    headers = {"Authorization": f"Token {auth_token}"}

    base_url = "https://tca-ssrm.com/api"
    report_url = f"{base_url}/estaciones/registros/reporte?estacion_id=164144&fecha_de_inicio=2025-04-01T00:00:00&periodo=1%20Mes&page_size=50&page=1&order_by=fecha&mode=hi"

    response = requests.get(report_url, headers=headers)
    response.raise_for_status()

    df_raw = pd.DataFrame(response.json()['data']['rows'], columns=response.json()['data']['header']).reset_index(drop=True)
    df_raw = df_raw.iloc[:-1]
    fechas = df_raw['Date'].tolist()

    columnas_a_mantener = [
        'Date', 'Avg Temp ºC', 'Avg DEW PT ºC', 'Avg Wind Speed km/h',
        'Max wind Speed km/h', 'Pressure HPA', 'Precip. mm', 'ET mm', 'Wind dir'
    ]
    df = df_raw[columnas_a_mantener].copy()
    df['Date_num'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce').apply(lambda x: x.timestamp())
    df = df.drop(columns=['Date'])
    df = df.rename(columns={
        'Avg Temp ºC': 'TempOut',
        'Avg DEW PT ºC': 'DewPt.',
        'Avg Wind Speed km/h': 'WSpeed',
        'Max wind Speed km/h': 'WHSpeed',
        'Pressure HPA': 'Bar',
        'Precip. mm': 'Rain',
        'ET mm': 'ET',
        'Wind dir': 'WDir_deg'
    })

    hook = S3Hook(aws_conn_id='minio_s3')
    bucket_name = 'respaldo2'

    all_models = hook.list_keys(bucket_name=bucket_name, prefix='best_model/')
    model_files = [k for k in all_models if k.endswith('.pkl')]
    latest_model = sorted(model_files, reverse=True)[0]

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_model_path = os.path.join(tmpdirname, os.path.basename(latest_model))
        hook.get_conn().download_file(
            Bucket=bucket_name,
            Key=latest_model,
            Filename=local_model_path
        )

        with open(local_model_path, 'rb') as f:
            model = pickle.load(f)

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[:, 1]
        else:
            proba = model.predict(df)

        for fecha, p in zip(fechas, proba):
            print(f"📅 Fecha: {fecha} - 🔮 Probabilidad de clase positiva: {p:.4f}")

def test_endpoints_predict(**kwargs):
    user = "ricardoq"
    pswd = "eLxdr3FZ51DE"
    url_auth = 'https://tca-ssrm.com/api/auth'
    payload = {'username': user, 'password': pswd}

    auth_response = requests.post(url_auth, data=payload)
    token = auth_response.json().get('token')
    headers = {"Authorization": f"Token {token}"}
    base_url = "https://tca-ssrm.com/api"
    report_url = f"{base_url}/estaciones/registros/reporte?estacion_id=164144&fecha_de_inicio=2025-04-01T00:00:00&periodo=1%20Mes&page_size=50&page=1&order_by=fecha&mode=hi"

    data_response = requests.get(report_url, headers=headers)
    data_json = data_response.json()

    df = pd.DataFrame(data_json['data']['rows'], columns=data_json['data']['header']).iloc[:-1]
    columnas_a_mantener = [
        'Date', 'Avg Temp ºC', 'Avg DEW PT ºC', 'Avg Wind Speed km/h',
        'Max wind Speed km/h', 'Pressure HPA', 'Precip. mm', 'ET mm', 'Wind dir'
    ]
    df = df[columnas_a_mantener]
    df['Date_num'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce').apply(lambda x: x.timestamp())
    df = df.drop(columns=['Date'])
    df = df.rename(columns={
        'Avg Temp ºC': 'TempOut',
        'Avg DEW PT ºC': 'DewPt_',
        'Avg Wind Speed km/h': 'WSpeed',
        'Max wind Speed km/h': 'WHSpeed',
        'Pressure HPA': 'Bar',
        'Precip. mm': 'Rain',
        'ET mm': 'ET',
        'Wind dir': 'WDir_deg'
    })
    df = df.dropna()

    sample_row = df.sample(1).to_dict(orient="records")[0]
    df_batch = df.to_dict(orient="records")

    url_predict = "http://fastapi_app:8000/predict"
    url_predict_batch = "http://fastapi_app:8000/predict_batch"

    response_single = requests.post(url_predict, json=sample_row)
    print(f"✅ Predict individual: {response_single.status_code} - {response_single.json()}")

    response_batch = requests.post(url_predict_batch, json=df_batch)
    print(f"✅ Predict batch: {response_batch.status_code} - {response_batch.json()}")
