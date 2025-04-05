from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import joblib
import os

# Ruta segura dentro del DAG para almacenar datos y resultados
BASE_PATH = os.path.join(os.path.dirname(__file__), 'output/')
os.makedirs(BASE_PATH, exist_ok=True)

def cargar_datos():
    df = pd.read_csv(f'{BASE_PATH}df_merged.csv', parse_dates=['Date'])
    df.to_pickle(f'{BASE_PATH}df_raw.pkl')

def preprocesar_datos():
    df = pd.read_pickle(f'{BASE_PATH}df_raw.pkl')

    # Crear nueva columna con fecha numérica
    df['Date_num'] = df['Date'].apply(lambda x: x.timestamp())

    # Eliminar columnas irrelevantes
    columnas_a_eliminar = ['Date', 'Punto', 'HiTemp', 'LowTemp', 'WTx', 'SolRate', 'SolRad.', 'arcInt']
    df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns], inplace=True)

    # Convertir a numérico y eliminar filas no válidas
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("El DataFrame quedó vacío después del preprocesamiento. Verifica los datos de entrada.")

    print(f"[INFO] Filas después del preprocesamiento: {len(df)}")
    df.to_pickle(f'{BASE_PATH}df_preprocessed.pkl')

def dividir_escalar():
    df = pd.read_pickle(f'{BASE_PATH}df_preprocessed.pkl')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -2].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    np.save(f'{BASE_PATH}X_train.npy', X_train)
    np.save(f'{BASE_PATH}X_test.npy', X_test)
    np.save(f'{BASE_PATH}y_train.npy', y_train)
    np.save(f'{BASE_PATH}y_test.npy', y_test)
    joblib.dump(scaler, f'{BASE_PATH}scaler.pkl')

def entrenar_con_gridsearch():
    X_train = np.load(f'{BASE_PATH}X_train.npy')
    y_train = np.load(f'{BASE_PATH}y_train.npy')

    model = SVC(kernel='rbf')
    grid = GridSearchCV(model, {"C": [0.001, 0.01, 0.1, 1, 5, 10, 100],
                                "gamma": [0.5, 1, 2, 3, 4]}, cv=5, scoring='f1')
    grid.fit(X_train, y_train)
    joblib.dump(grid.best_estimator_, f'{BASE_PATH}svm_rbf_best.pkl')

def evaluar_modelo():
    X_test = np.load(f'{BASE_PATH}X_test.npy')
    y_test = np.load(f'{BASE_PATH}y_test.npy')
    model = joblib.load(f'{BASE_PATH}svm_rbf_best.pkl')

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.grid(False)
    disp.plot(ax=ax)
    plt.savefig(f'{BASE_PATH}confusion_matrix.png')

with DAG('svm_model_avanzado_dag',
         start_date=datetime(2024, 1, 1),
         schedule_interval=None,
         catchup=False) as dag:

    t1 = PythonOperator(task_id='cargar_datos', python_callable=cargar_datos)
    t2 = PythonOperator(task_id='preprocesar_datos', python_callable=preprocesar_datos)
    t3 = PythonOperator(task_id='dividir_y_escalar', python_callable=dividir_escalar)
    t4 = PythonOperator(task_id='entrenar_modelo_gridsearch', python_callable=entrenar_con_gridsearch)
    t5 = PythonOperator(task_id='evaluar_modelo', python_callable=evaluar_modelo)

    t1 >> t2 >> t3 >> t4 >> t5
