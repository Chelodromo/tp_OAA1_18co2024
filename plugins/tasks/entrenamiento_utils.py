# plugins/tasks/entrenamiento_utils.py
import os
import tempfile
import pandas as pd
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import optuna
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

def simple_mlflow_run(**kwargs):
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("test_airflow_experiment")
    with mlflow.start_run(run_name="test_run"):
        mlflow.log_param("param1", 5)
        mlflow.log_metric("accuracy", 0.87)
    print("✅ MLflow test: parámetro y métrica registrados.")

def train_lightgbm_optuna_minio(**kwargs):
    _train_generic_model_lightgbm(bucket_name="respaldo2", splits_path="splits")

def train_randomforest_optuna_minio(**kwargs):
    _train_generic_model_randomforest(bucket_name="respaldo2", splits_path="splits")

def train_logisticregression_optuna_minio(**kwargs):
    _train_generic_model_logistic(bucket_name="respaldo2", splits_path="splits")

def train_knn_optuna_minio(**kwargs):
    _train_generic_model_knn(bucket_name="respaldo2", splits_path="splits")

# Funciones privadas (helper functions)

def _load_splits(hook, bucket_name, splits_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        archivos = ['X_train.json', 'X_test.json', 'y_train.json', 'y_test.json']
        paths = {}
        for archivo in archivos:
            content = hook.read_key(f"{splits_path}/{archivo}", bucket_name)
            path_local = os.path.join(tmpdir, archivo)
            with open(path_local, 'w') as f:
                f.write(content)
            paths[archivo] = path_local
        X_train = pd.read_json(paths['X_train.json'])
        X_test = pd.read_json(paths['X_test.json'])
        y_train = pd.read_json(paths['y_train.json'], typ='series')
        y_test = pd.read_json(paths['y_test.json'], typ='series')
        return X_train, X_test, y_train, y_test

def _train_generic_model_lightgbm(bucket_name, splits_path):
    hook = S3Hook(aws_conn_id="minio_s3")
    X_train, X_test, y_train, y_test = _load_splits(hook, bucket_name, splits_path)

    mlflow.set_tracking_uri("http://mlflow:5000")

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
        model = lgb.train(params, lgb_train, num_boost_round=100, callbacks=[lgb.log_evaluation(0)])
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        return recall_score(y_test, y_pred_binary)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    best_params.update({'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt'})

    final_model = lgb.train(best_params, lgb.Dataset(X_train, y_train), num_boost_round=100, callbacks=[lgb.log_evaluation(0)])

    with tempfile.NamedTemporaryFile(delete=False) as tmp_model:
        pickle.dump(final_model, tmp_model)
        tmp_model_path = tmp_model.name

    hook.load_file(
        filename=tmp_model_path,
        key="modelos/lightgbm_model.pkl",
        bucket_name=bucket_name,
        replace=True
    )

def _train_generic_model_randomforest(bucket_name, splits_path):
    hook = S3Hook(aws_conn_id="minio_s3")
    X_train, X_test, y_train, y_test = _load_splits(hook, bucket_name, splits_path)

    mlflow.set_tracking_uri("http://mlflow:5000")

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
        return recall_score(y_test, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    model_final = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    model_final.fit(X_train, y_train)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_model:
        pickle.dump(model_final, tmp_model)
        tmp_model_path = tmp_model.name

    hook.load_file(
        filename=tmp_model_path,
        key="modelos/randomforest_model.pkl",
        bucket_name=bucket_name,
        replace=True
    )

def _train_generic_model_logistic(bucket_name, splits_path):
    hook = S3Hook(aws_conn_id="minio_s3")
    X_train, X_test, y_train, y_test = _load_splits(hook, bucket_name, splits_path)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlflow.set_tracking_uri("http://mlflow:5000")

    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 1e-4, 10.0, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'lbfgs']),
        }
        model = LogisticRegression(**params)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        return recall_score(y_test, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    model_final = LogisticRegression(**best_params)
    model_final.fit(X_train_scaled, y_train)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_model:
        pickle.dump(model_final, tmp_model)
        tmp_model_path = tmp_model.name

    hook.load_file(
        filename=tmp_model_path,
        key="modelos/logisticregression_model.pkl",
        bucket_name=bucket_name,
        replace=True
    )

def _train_generic_model_knn(bucket_name, splits_path):
    hook = S3Hook(aws_conn_id="minio_s3")
    X_train, X_test, y_train, y_test = _load_splits(hook, bucket_name, splits_path)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlflow.set_tracking_uri("http://mlflow:5000")

    def objective(trial):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 50),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_categorical('p', [1, 2]),
        }
        model = KNeighborsClassifier(**params)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        return recall_score(y_test, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    model_final = KNeighborsClassifier(**best_params)
    model_final.fit(X_train_scaled, y_train)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_model:
        pickle.dump(model_final, tmp_model)
        tmp_model_path = tmp_model.name

    hook.load_file(
        filename=tmp_model_path,
        key="modelos/knn_model.pkl",
        bucket_name=bucket_name,
        replace=True
    )
