# plugins/tasks/entrenamiento_utils.py
import os
import tempfile
import pandas as pd
import pickle
import mlflow
import optuna
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

def simple_mlflow_run(**kwargs):
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("test_airflow_experiment")

    with mlflow.start_run(run_name="test_run"):
        mlflow.log_param("param1", 5)
        mlflow.log_metric("accuracy", 0.87)
        print("✅ MLflow test: parámetro y métrica registrados.")

def train_lightgbm_optuna_minio(**kwargs):
    import lightgbm as lgb
    from mlflow import lightgbm as mlflow_lgb

    bucket_name = 'respaldo2'
    splits_path = 'splits'
    aws_conn_id = 'minio_s3'
    hook = S3Hook(aws_conn_id=aws_conn_id)

    with tempfile.TemporaryDirectory() as tmpdir:
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
            model = lgb.train(params, lgb_train, num_boost_round=100, callbacks=[lgb.log_evaluation(0)])
            y_pred = model.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)
            return recall_score(y_test, y_pred_binary)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)

        best_params = study.best_params
        best_params.update({'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt'})

        model_final = lgb.train(best_params, lgb.Dataset(X_train, y_train), num_boost_round=100)

        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("lightgbm_experiment")
        with mlflow.start_run(run_name="lightgbm_run"):
            mlflow.log_params(best_params)
            y_pred = model_final.predict(X_test)
            recall = recall_score(y_test, (y_pred > 0.5).astype(int))
            mlflow.log_metric("recall", recall)
            mlflow_lgb.log_model(model_final, artifact_path="lightgbm_model")

        model_path = os.path.join(tmpdir, "lightgbm_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_final, f)

        hook.load_file(filename=model_path, key="modelos/lightgbm_model.pkl", bucket_name=bucket_name, replace=True)

def train_randomforest_optuna_minio(**kwargs):
    from sklearn.ensemble import RandomForestClassifier
    from mlflow import sklearn as mlflow_sklearn

    bucket_name = 'respaldo2'
    splits_path = 'splits'
    aws_conn_id = 'minio_s3'
    hook = S3Hook(aws_conn_id=aws_conn_id)

    with tempfile.TemporaryDirectory() as tmpdir:
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

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)

        best_params = study.best_params
        model_final = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        model_final.fit(X_train, y_train)

        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("randomforest_experiment")
        with mlflow.start_run(run_name="randomforest_run"):
            mlflow.log_params(best_params)
            preds = model_final.predict(X_test)
            recall = recall_score(y_test, preds)
            mlflow.log_metric("recall", recall)
            mlflow_sklearn.log_model(model_final, artifact_path="randomforest_model")

        model_path = os.path.join(tmpdir, "randomforest_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_final, f)

        hook.load_file(filename=model_path, key="modelos/randomforest_model.pkl", bucket_name=bucket_name, replace=True)

# (Continúan train_logisticregression_optuna_minio y train_knn_optuna_minio)
def train_logisticregression_optuna_minio(**kwargs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from mlflow import sklearn as mlflow_sklearn

    bucket_name = 'respaldo2'
    splits_path = 'splits'
    aws_conn_id = 'minio_s3'
    hook = S3Hook(aws_conn_id=aws_conn_id)

    with tempfile.TemporaryDirectory() as tmpdir:
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
            return recall_score(y_test, y_pred)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)

        best_params = study.best_params
        model_final = LogisticRegression(**best_params)
        model_final.fit(X_train_scaled, y_train)

        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("logisticregression_experiment")
        with mlflow.start_run(run_name="logisticregression_run"):
            mlflow.log_params(best_params)
            y_pred = model_final.predict(X_test_scaled)
            recall = recall_score(y_test, y_pred)
            mlflow.log_metric("recall", recall)
            mlflow_sklearn.log_model(sk_model=model_final, artifact_path="logisticregression_model")

        model_path = os.path.join(tmpdir, "logisticregression_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_final, f)

        hook.load_file(filename=model_path, key="modelos/logisticregression_model.pkl", bucket_name=bucket_name, replace=True)

def train_knn_optuna_minio(**kwargs):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
    from mlflow import sklearn as mlflow_sklearn

    bucket_name = 'respaldo2'
    splits_path = 'splits'
    aws_conn_id = 'minio_s3'
    hook = S3Hook(aws_conn_id=aws_conn_id)

    with tempfile.TemporaryDirectory() as tmpdir:
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

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        def objective(trial):
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 50),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_categorical('p', [1, 2]),
            }
            model = KNeighborsClassifier(**params)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            return recall_score(y_test, y_pred)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)

        best_params = study.best_params
        model_final = KNeighborsClassifier(**best_params)
        model_final.fit(X_train_scaled, y_train)

        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("knn_experiment")
        with mlflow.start_run(run_name="knn_run"):
            mlflow.log_params(best_params)
            y_pred = model_final.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            try:
                roc_auc = roc_auc_score(y_test, model_final.predict_proba(X_test_scaled)[:, 1])
            except AttributeError:
                roc_auc = 0.0

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow_sklearn.log_model(sk_model=model_final, artifact_path="knn_model")

        model_path = os.path.join(tmpdir, "knn_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_final, f)

        hook.load_file(filename=model_path, key="modelos/knn_model.pkl", bucket_name=bucket_name, replace=True)

