
# 🛠 Proyecto Airflow + MinIO + ML Pipeline

Este proyecto orquesta un flujo completo de procesamiento de datos y entrenamiento de modelos de Machine Learning usando **Apache Airflow**, con almacenamiento de datos en **MinIO (S3 compatible)** y tracking de experimentos en **MLflow**. Todo está dockerizado y configurado para correr automáticamente.

---

## 📦 Estructura del Proyecto

```
.
├── dags/                       # DAGs de Airflow
├── scripts/                    # Scripts auxiliares
├── logs/                       # Carpeta ignorada por Git (Airflow logs)
├── datalake/                   # Carpeta local para almacenamiento temporal
├── docker-compose.yml          # Definición de servicios
├── mlflow/                     # Artefactos y base de MLflow
└── .gitignore                  # Ignora logs, datalake, mlflow, etc.
```

---

## 🔁 Flujo del DAG principal (`descargar_y_ver_dataset`)

1. **`probar_minio`**
   - Crea bucket `respaldo2` si no existe en MinIO.

2. **`descargar_dataset`**
   - Descarga dataset desde Google Drive.
   - Lo sube a MinIO en `respaldo2/dataset.csv`.

3. **`procesar_dataset`**
   - Limpieza y preprocesamiento del dataset descargado.
   - Carga el backup como JSON en MinIO (`dataset_backup.json`).

4. **`split_dataset_minio`**
   - Realiza el split `train/test` desde el JSON de MinIO.
   - Guarda `X_train.json`, `X_test.json`, `y_train.json`, `y_test.json` en `respaldo2/splits/`.

5. **Modelado y entrenamiento**:
   - **`train_lightgbm_optuna_minio`**: entrena usando Optuna y sube el modelo.
   - **`train_randomforest_optuna_minio`**: entrena usando Optuna y sube el modelo.
   - **`train_logisticregression_optuna_minio`**: entrena usando Optuna y sube el modelo.
   - **`train_knn_optuna_minio`**: entrena usando Optuna y sube el modelo.

6. **`seleccionar_mejor_modelo`**
   - Evalúa todos los experimentos en MLflow según la métrica **Recall**.
   - Selecciona el mejor modelo.
   - Descarga su `pkl` y lo sube a MinIO en la carpeta `best_model/` (con timestamp en el nombre).

---

## 🌐 MinIO (S3 Compatible)

- **Console**: [http://localhost:9001](http://localhost:9001)
- **API**: [http://localhost:9000](http://localhost:9000)
- **Usuario/Contraseña**: `minioadmin/minioadmin`
- **Bucket**: `respaldo2`

Airflow usa esta conexión:

```yaml
AIRFLOW_CONN_MINIO_S3=s3://minioadmin:minioadmin@minio:9000/?endpoint_url=http%3A%2F%2Fminio%3A9000
```

---

## 🧠 MLflow

- **Tracking Server**: [http://localhost:5001](http://localhost:5001)
- **Almacenamiento**: `./mlflow`
- **Base de datos**: SQLite (`mlflow/mlflow.db`)

---

## 🚀 Cómo levantar todo

```bash
docker-compose up -d
```

Accedé a Airflow en: [http://localhost:8080](http://localhost:8080)  
- Usuario: `airflow`
- Contraseña: `airflow`

---

## 🧹 Git

Ignorar carpetas de datos y logs:

```
logs/
datalake/
mlflow/
```

Para limpiar si ya fueron comiteadas:

```bash
git rm -r --cached logs/ datalake/ mlflow/
echo -e "logs/
datalake/
mlflow/" >> .gitignore
git commit -m "Ignorar carpetas de datos temporales y logs"
git push
```

---
