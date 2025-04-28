# plugins/tasks/procesamiento_utils.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

DATA_PATH = "/opt/airflow/datalake/df_merged.csv"
BACKUP_PATH = "/opt/airflow/datalake/df_procesado.json"
OUTPUT_DIR = "/opt/airflow/datalake"

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

    df.to_json(BACKUP_PATH, orient="records", date_format="iso")
    print(f"💾 Dataset respaldado como JSON en: {BACKUP_PATH}")

def split_dataset():
    df = pd.read_json(BACKUP_PATH)

    if 'date' in df.columns:
        df = df.rename(columns={'date': 'Date'})

    df['Date_num'] = df['Date'].apply(lambda x: pd.to_datetime(x).timestamp())
    df = df.drop(columns=['Date'])

    columnas_a_eliminar = ['Punto', 'HiTemp', 'LowTemp', 'WTx', 'SolRate', 'SolRad.', 'arcInt']
    df = df.drop(columns=[c for c in columnas_a_eliminar if c in df.columns])

    X = df.drop(columns=['clase'])
    y = df['clase']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    X_train.to_json(os.path.join(OUTPUT_DIR, "X_train.json"), orient="records")
    X_test.to_json(os.path.join(OUTPUT_DIR, "X_test.json"), orient="records")
    y_train.to_json(os.path.join(OUTPUT_DIR, "y_train.json"), orient="records")
    y_test.to_json(os.path.join(OUTPUT_DIR, "y_test.json"), orient="records")

    print("✅ Split realizado y archivos guardados en /opt/airflow/datalake.")

def svm_modeling():
    X_train = pd.read_json(f"{OUTPUT_DIR}/X_train.json")
    X_test = pd.read_json(f"{OUTPUT_DIR}/X_test.json")
    y_train = pd.read_json(f"{OUTPUT_DIR}/y_train.json", typ='series')
    y_test = pd.read_json(f"{OUTPUT_DIR}/y_test.json", typ='series')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(C=0.001, kernel='linear')
    model.fit(X_train, y_train)

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print("\n🔁 Cross-validation scores:", scores)
    print("📊 Cross-validation mean accuracy:", scores.mean())

    y_pred = model.predict(X_test)
    print("\n🔍 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
