from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from botocore.exceptions import ClientError
import os
import requests
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
import mlflow
import pickle

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
        #columnas_a_eliminar = ['Date', 'Punto', 'HiTemp', 'LowTemp', 'WTx', 'SolRate', 'SolRad.', 'arcInt']
        columnas_a_eliminar =['Date','Punto', 'HiTemp', 'LowTemp', 'WTx', 'SolRate', 'SolRad.', 'arcInt',  # columnas viejas que sobraban
            'OutHum', 'WRun', 'WChill', 'HeatIx', 'ThwIx', 'ThswI', 'HiSlE', 'Rad.', 
            'uvIndex', 'uDose', 'hiUV', 'hetD-D', 'colD-D', 'inTemp', 'inHum',
            'inDew', 'iHeat', 'WSamp', 'iRecept', 'HWDir_deg']
        df_polvo_svm = df_polvo_svm.drop(columns=[c for c in columnas_a_eliminar if c in df_polvo_svm.columns])

        print(df_polvo_svm.info())

        # Separar features y target
        X = df_polvo_svm.drop(columns=['clase'])
        y = df_polvo_svm['clase']

        # Realizar el split
        X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
            X, y, stratify=y, test_size=0.3, random_state=42
        )
        print(X_train_svm.info())
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
            

def simple_mlflow_run(**kwargs):
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("test_airflow_experiment")

    with mlflow.start_run(run_name="test_run"):
        mlflow.log_param("param1", 5)
        mlflow.log_metric("accuracy", 0.87)
        print("✅ MLflow test: parámetro y métrica registrados.")

def train_lightgbm_optuna_minio(**kwargs):
    import os
    import tempfile
    import pandas as pd
    import pickle
    import lightgbm as lgb
    import mlflow
    import mlflow.lightgbm
    import optuna
    from sklearn.metrics import recall_score
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook

    bucket_name = 'respaldo2'
    splits_path = 'splits'
    aws_conn_id = 'minio_s3'

    mlflow.set_tracking_uri("http://mlflow:5000")

    hook = S3Hook(aws_conn_id=aws_conn_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"📁 Usando directorio temporal: {tmpdir}")

        archivos = ['X_train.json', 'X_test.json', 'y_train.json', 'y_test.json']
        paths_locales = {}

        for archivo in archivos:
            content = hook.read_key(key=f"{splits_path}/{archivo}", bucket_name=bucket_name)
            local_path = os.path.join(tmpdir, archivo)
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)
            paths_locales[archivo] = local_path

        X_train = pd.read_json(paths_locales['X_train.json'])
        X_test = pd.read_json(paths_locales['X_test.json'])
        y_train = pd.read_json(paths_locales['y_train.json'], typ='series')
        y_test = pd.read_json(paths_locales['y_test.json'], typ='series')

        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            }

            lgb_train = lgb.Dataset(X_train, y_train)
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=100,
                callbacks=[lgb.log_evaluation(0)]
            )

            y_pred = model.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)

            return recall_score(y_test, y_pred_binary)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)

        print("✅ Mejor hiperparámetros encontrados:", study.best_params)

        best_params = study.best_params
        best_params.update({
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
        })

        lgb_train = lgb.Dataset(X_train, y_train)
        model_final = lgb.train(
            best_params,
            lgb_train,
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(0)]
        )

        experiment_name = "lightgbm_experiment"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="lightgbm_run") as run:
            mlflow.log_params(best_params)

            y_pred = model_final.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)
            recall = recall_score(y_test, y_pred_binary)

            mlflow.log_metric("recall", recall)

            mlflow.lightgbm.log_model(
                model_final,
                artifact_path="lightgbm_model"
            )

        print(f"✅ Modelo registrado en MLflow (experimento: {experiment_name})")

        model_path = os.path.join(tmpdir, "lightgbm_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_final, f)

        hook.load_file(
            filename=model_path,
            key="modelos/lightgbm_model.pkl",
            bucket_name=bucket_name,
            replace=True
        )
        print(f"✅ Modelo subido a MinIO: modelos/lightgbm_model.pkl")



def train_randomforest_optuna_minio(**kwargs):
    import os
    import tempfile
    import pandas as pd
    import pickle
    import mlflow
    import mlflow.sklearn
    import optuna
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import recall_score
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook

    bucket_name = 'respaldo2'
    splits_path = 'splits'
    aws_conn_id = 'minio_s3'

    hook = S3Hook(aws_conn_id=aws_conn_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"📁 Usando directorio temporal: {tmpdir}")

        archivos = ['X_train.json', 'X_test.json', 'y_train.json', 'y_test.json']
        paths_locales = {}

        for archivo in archivos:
            content = hook.read_key(key=f"{splits_path}/{archivo}", bucket_name=bucket_name)
            local_path = os.path.join(tmpdir, archivo)
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)
            paths_locales[archivo] = local_path
            print(f"✅ Descargado: {archivo}")

        X_train = pd.read_json(paths_locales['X_train.json'])
        X_test = pd.read_json(paths_locales['X_test.json'])
        y_train = pd.read_json(paths_locales['y_train.json'], typ='series')
        y_test = pd.read_json(paths_locales['y_test.json'], typ='series')

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            }
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            recall = recall_score(y_test, preds)
            return recall

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)

        print("✅ Mejor hiperparámetros encontrados:", study.best_params)

        best_params = study.best_params
        model_final = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        model_final.fit(X_train, y_train)

        mlflow.set_tracking_uri("http://mlflow:5000")
        experiment_name = "randomforest_experiment"

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="randomforest_run") as run:
            mlflow.log_params(best_params)

            preds = model_final.predict(X_test)
            recall = recall_score(y_test, preds)

            mlflow.log_metric("recall", recall)

            mlflow.sklearn.log_model(model_final, artifact_path="randomforest_model")

        print(f"✅ Modelo registrado en MLflow (experiment: {experiment_name})")

        model_path = os.path.join(tmpdir, "randomforest_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_final, f)

        hook.load_file(
            filename=model_path,
            key="modelos/randomforest_model.pkl",
            bucket_name=bucket_name,
            replace=True
        )
        print(f"✅ Modelo subido a MinIO en modelos/randomforest_model.pkl")


def train_logisticregression_optuna_minio(**kwargs):
    import os
    import tempfile
    import pandas as pd
    import pickle
    import mlflow
    import mlflow.sklearn
    import optuna
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import recall_score
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook

    bucket_name = 'respaldo2'
    splits_path = 'splits'
    aws_conn_id = 'minio_s3'

    hook = S3Hook(aws_conn_id=aws_conn_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"📁 Usando directorio temporal: {tmpdir}")

        archivos = ['X_train.json', 'X_test.json', 'y_train.json', 'y_test.json']
        paths_locales = {}

        for archivo in archivos:
            content = hook.read_key(key=f"{splits_path}/{archivo}", bucket_name=bucket_name)
            local_path = os.path.join(tmpdir, archivo)
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)
            paths_locales[archivo] = local_path
            print(f"✅ Descargado: {archivo}")

        X_train = pd.read_json(paths_locales['X_train.json'])
        X_test = pd.read_json(paths_locales['X_test.json'])
        y_train = pd.read_json(paths_locales['y_train.json'], typ='series')
        y_test = pd.read_json(paths_locales['y_test.json'], typ='series')

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 1e-4, 10.0, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
            }
            model = LogisticRegression(**params)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            recall = recall_score(y_test, y_pred)
            return recall

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)

        print("✅ Mejor hiperparámetros encontrados:", study.best_params)

        best_params = study.best_params
        model_final = LogisticRegression(**best_params)
        model_final.fit(X_train_scaled, y_train)

        mlflow.set_tracking_uri("http://mlflow:5000")
        experiment_name = "logisticregression_experiment"

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="logisticregression_run") as run:
            mlflow.log_params(best_params)

            y_pred = model_final.predict(X_test_scaled)
            recall = recall_score(y_test, y_pred)

            mlflow.log_metric("recall", recall)

            mlflow.sklearn.log_model(
                sk_model=model_final,
                artifact_path="logisticregression_model"
            )

        print(f"✅ Modelo registrado en MLflow (experiment: {experiment_name})")

        model_path = os.path.join(tmpdir, "logisticregression_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_final, f)

        hook.load_file(
            filename=model_path,
            key="modelos/logisticregression_model.pkl",
            bucket_name=bucket_name,
            replace=True
        )
        print(f"✅ Modelo subido a MinIO en modelos/logisticregression_model.pkl")

def train_knn_optuna_minio(**kwargs):
    import os
    import tempfile
    import pandas as pd
    import pickle
    import mlflow
    import mlflow.sklearn
    import optuna
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook

    # Parámetros generales
    bucket_name = 'respaldo2'
    splits_path = 'splits'
    aws_conn_id = 'minio_s3'

    hook = S3Hook(aws_conn_id=aws_conn_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"📁 Usando directorio temporal: {tmpdir}")

        # Descargar splits
        archivos = ['X_train.json', 'X_test.json', 'y_train.json', 'y_test.json']
        paths_locales = {}

        for archivo in archivos:
            content = hook.read_key(key=f"{splits_path}/{archivo}", bucket_name=bucket_name)
            local_path = os.path.join(tmpdir, archivo)
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)
            paths_locales[archivo] = local_path
            print(f"✅ Descargado: {archivo}")

        # Cargar datasets
        X_train = pd.read_json(paths_locales['X_train.json'])
        X_test = pd.read_json(paths_locales['X_test.json'])
        y_train = pd.read_json(paths_locales['y_train.json'], typ='series')
        y_test = pd.read_json(paths_locales['y_test.json'], typ='series')

        # Escalar features (importante para KNN)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Definición de la función de Optuna
        def objective(trial):
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 50),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_categorical('p', [1, 2]),
            }
            model = KNeighborsClassifier(**params)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            recall = recall_score(y_test, y_pred)
            return recall  # 🔥 Buscamos maximizar recall

        # Ejecutar Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)

        print("✅ Mejor hiperparámetros encontrados:", study.best_params)

        # Entrenar modelo final
        best_params = study.best_params
        model_final = KNeighborsClassifier(**best_params)
        model_final.fit(X_train_scaled, y_train)

        # MLflow Tracking
        mlflow.set_tracking_uri("http://mlflow:5000")
        experiment_name = "knn_experiment"

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="knn_run") as run:
            mlflow.log_params(best_params)

            y_pred = model_final.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            try:
                roc_auc = roc_auc_score(y_test, model_final.predict_proba(X_test_scaled)[:, 1])
            except AttributeError:
                roc_auc = 0.0  # En caso de que predict_proba no esté disponible (por configuración)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("roc_auc", roc_auc)

            # Registrar el modelo
            mlflow.sklearn.log_model(
                sk_model=model_final,
                artifact_path="knn_model"
            )

        print(f"✅ Modelo registrado en MLflow (experimento: {experiment_name})")

        # Guardar el modelo como pickle
        model_path = os.path.join(tmpdir, "knn_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_final, f)
        print(f"💾 Modelo guardado como {model_path}")

        # Subir el modelo a MinIO
        hook.load_file(
            filename=model_path,
            key="modelos/knn_model.pkl",
            bucket_name=bucket_name,
            replace=True
        )
        print(f"✅ Modelo subido a MinIO en modelos/knn_model.pkl")
        
def seleccionar_mejor_modelo(**kwargs):
    import os
    import tempfile
    import mlflow
    import pickle
    from datetime import datetime
    from airflow.providers.amazon.aws.hooks.s3 import S3Hook
    from mlflow.tracking import MlflowClient

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

    # Configuración de MinIO
    bucket_name = 'respaldo2'
    hook = S3Hook(aws_conn_id='minio_s3')

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"📁 Usando directorio temporal: {tmpdir}")

        # Preparar nombre del modelo
        modelo_name = best_experiment_name.replace('_experiment', '')
        key_modelo = f"modelos/{modelo_name}_model.pkl"

        # Descargar modelo
        local_model_path = os.path.join(tmpdir, f"{modelo_name}_model.pkl")
        obj = hook.get_key(key=key_modelo, bucket_name=bucket_name)
        with open(local_model_path, 'wb') as f:
            f.write(obj.get()['Body'].read())

        print(f"✅ Modelo descargado: {local_model_path}")

        # Armar timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_model_key = f"best_model/{modelo_name}_{timestamp}.pkl"

        # Subir modelo con timestamp
        hook.load_file(
            filename=local_model_path,
            key=best_model_key,
            bucket_name=bucket_name,
            replace=True
        )

        print(f"🚀 Mejor modelo subido como {best_model_key}")

def predict_datos_actuales(**kwargs):
    # 🔐 Autenticarse
    user = "ricardoq"
    pswd = "eLxdr3FZ51DE"
    auth_url = 'https://tca-ssrm.com/api/auth'
    payload = {'username': user, 'password': pswd}

    auth_response = requests.post(auth_url, data=payload)
    auth_response.raise_for_status()
    auth_token = auth_response.json()['token']

    headers = {"Authorization": f"Token {auth_token}"}

    # 🌐 Traer datos actuales
    base_url = "https://tca-ssrm.com/api"
    report_url = f"{base_url}/estaciones/registros/reporte?estacion_id=164144&fecha_de_inicio=2025-04-01T00:00:00&periodo=1%20Mes&page_size=50&page=1&order_by=fecha&mode=hi"

    response = requests.get(report_url, headers=headers)
    response.raise_for_status()

    df_raw = pd.DataFrame(response.json()['data']['rows'], columns=response.json()['data']['header']).reset_index(drop=True)

    # ⚠️ Eliminar última fila (AVERAGE)
    df_raw = df_raw.iloc[:-1]

    # 🗓️ Guardar las fechas para luego
    fechas = df_raw['Date'].tolist()

    # ✨ Preprocesar
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

    print("\n✅ Datos actuales listos para predicción:")
    print(df.head())

    # 📦 Descargar último modelo
    hook = S3Hook(aws_conn_id='minio_s3')
    bucket_name = 'respaldo2'

    all_models = hook.list_keys(bucket_name=bucket_name, prefix='best_model/')
    model_files = [k for k in all_models if k.endswith('.pkl')]
    latest_model = sorted(model_files, reverse=True)[0]

    print(f"📦 Último modelo encontrado: {latest_model}")

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Ruta final donde guardarlo
        local_model_path = os.path.join(tmpdirname, os.path.basename(latest_model))
        
        # Usar hook para descargar explícitamente en esa ruta
        hook.get_conn().download_file(
            Bucket=bucket_name,
            Key=latest_model,
            Filename=local_model_path
        )

        with open(local_model_path, 'rb') as f:
            model = pickle.load(f)

        # 🔮 Predecir
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[:, 1]
        else:
            proba = model.predict(df)

        # 📅 Mostrar resultados
        for fecha, p in zip(fechas, proba):
            print(f"📅 Fecha: {fecha} - 🔮 Probabilidad de clase positiva (polvo): {p:.4f}")

def test_endpoints_predict(**kwargs):
    import pandas as pd
    import requests

    # 🔥 Pido los datos actuales a la API de la estación
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

    df = pd.DataFrame(data_json['data']['rows'], columns=data_json['data']['header'])
    df = df.iloc[:-1]

    # 🔥 Preprocesado igual que en predict_datos_actuales
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

    # 🔥 Elegir uno aleatorio
    sample_row = df.sample(1).to_dict(orient="records")[0]
    df_batch = df.to_dict(orient="records")

    # 🔥 Hacer requests a la API FastAPI
    url_predict = "http://fastapi_app:8000/predict"
    url_predict_batch = "http://fastapi_app:8000/predict_batch"

    response_single = requests.post(url_predict, json=sample_row)
    print(f"✅ Resultado predict individual: {response_single.status_code} - {response_single.json()}")

    response_batch = requests.post(url_predict_batch, json=df_batch)
    print(f"✅ Resultado predict batch: {response_batch.status_code} - {response_batch.json()}")



# DAG definition
with DAG(
    dag_id="mlops_prediccion-polvo",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    
    probar_minio =  PythonOperator(
        task_id='conectar_minio',
        python_callable=ejemplo_conexion_s3)


    descargar_dataset = PythonOperator(
        task_id='descargar_dataset',
        python_callable=descargar_dataset,
        provide_context=True  # Si usás **kwargs, esto es necesario
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

    run_test = PythonOperator(
        task_id="mlflow_test_run",
        python_callable=simple_mlflow_run,
        provide_context=True
    )
    
    train_lightgbm_task = PythonOperator(
    task_id='train_lightgbm_optuna_minio',
    python_callable=train_lightgbm_optuna_minio,
    provide_context=True,
    dag=dag
)
    train_randomforest_task = PythonOperator(
    task_id='train_randomforest_optuna_minio',
    python_callable=train_randomforest_optuna_minio,
    provide_context=True,
    dag=dag
)
    train_logisticregression_task = PythonOperator(
    task_id='train_logisticregression_optuna_minio',
    python_callable=train_logisticregression_optuna_minio,
    provide_context=True,
    dag=dag
)

    train_knn_optuna_minio_task = PythonOperator(
    task_id='train_knn_optuna_minio',
    python_callable=train_knn_optuna_minio,
    provide_context=True,
    dag=dag
)

    seleccionar_mejor_modelo_task = PythonOperator(
    task_id='seleccionar_mejor_modelo',
    python_callable=seleccionar_mejor_modelo,
    provide_context=True,
    dag=dag
)

    predict_datos_actuales_task = PythonOperator(
    task_id='predict_datos_actuales',
    python_callable=predict_datos_actuales,
    dag=dag,
)

    test_fastapi_endpoints = PythonOperator(
    task_id="test_fastapi_endpoints",
    python_callable=test_endpoints_predict,
    provide_context=True,
    dag=dag
)

probar_minio >> descargar_dataset >> procesar_dataset >> split_dataset_task_minio >> run_test >> [train_lightgbm_task, train_randomforest_task, train_logisticregression_task , train_knn_optuna_minio_task] >> seleccionar_mejor_modelo_task >> predict_datos_actuales_task >> test_fastapi_endpoints

