
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
# 🛠️ Proyecto Airflow + MinIO + ML Pipeline + FastAPI

Este proyecto orquesta un flujo completo de procesamiento de datos, entrenamiento y deploy de modelos usando:
- **Apache Airflow** (orquestación)
- **MinIO** (storage tipo S3)
- **MLflow** (tracking de experimentos)
- **FastAPI** (servir el modelo en producción)

Todo el ecosistema corre en **Docker Compose**.

## 📦 Estructura del Proyecto

```
.
├── dags/                   # DAGs de Airflow
├── fastapi_app/             # App de FastAPI para servir el modelo
│   ├── app.py               # API principal
│   ├── schemas.py           # Esquemas de entrada
├── mlflow/                  # Carpeta local para MLflow tracking
├── datalake/                # Datalake local (usado por MinIO)
├── docker-compose.yml       # Definición de servicios
├── Dockerfile.fastapi       # Imagen de la app de FastAPI
├── .env                     # Variables de entorno
└── .gitignore               # Ignorar archivos temporales
```

## 🐍 FastAPI para servir modelos

- **App**: Corre en `http://localhost:8000`
- **Documentación Swagger**: `http://localhost:8000/docs`

### Endpoints disponibles

- **POST** `/predict`
  - Recibe un único registro para predecir.
  - **Ejemplo de input**:
```json
{
    "TempOut": 10.5,
    "DewPt_": -3.2,
    "WSpeed": 5.7,
    "WHSpeed": 30.0,
    "Bar": 622.1,
    "Rain": 0.0,
    "ET": 4.2,
    "WDir_deg": 154.5,
    "Date_num": 1743465600.0
}
```

- **POST** `/predict_batch`
  - Recibe varios registros en formato lista.
  - **Ejemplo de input**:
```json
[
  {
    "TempOut": 10.5,
    "DewPt_": -3.2,
    "WSpeed": 5.7,
    "WHSpeed": 30.0,
    "Bar": 622.1,
    "Rain": 0.0,
    "ET": 4.2,
    "WDir_deg": 154.5,
    "Date_num": 1743465600.0
  },
  {
    "TempOut": 9.8,
    "DewPt_": -2.1,
    "WSpeed": 6.0,
    "WHSpeed": 35.0,
    "Bar": 621.5,
    "Rain": 0.0,
    "ET": 4.5,
    "WDir_deg": 120.5,
    "Date_num": 1743552000.0
  }
]
```

- **Respuesta**:
```json
{
  "prediction": [false, true]
}
```

### 🔄 Actualización Dinámica del Modelo

Cada vez que la app FastAPI se inicia:
- Busca automáticamente el último modelo `.pkl` en el bucket MinIO `respaldo2/best_model/`
- Carga el modelo al inicio (`@app.on_event('startup')`).


## 🚀 Para levantar todo

```bash
docker-compose up --build
```

Accesos:
- **Airflow**: [http://localhost:8080](http://localhost:8080)
- **FastAPI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **MinIO Console**: [http://localhost:9001](http://localhost:9001)
- **MLflow Tracking**: [http://localhost:5001](http://localhost:5001)


## 🔧 Servicios Docker

| Servicio         | Puerto Expuesto | Descripción                  |
|------------------|------------------|-------------------------------|
| Airflow Webserver | 8080             | UI de Airflow                  |
| Airflow Scheduler | 8080             | Scheduler de Airflow           |
| Airflow Worker    | 8080             | Workers de Airflow             |
| MinIO             | 9000, 9001        | API y consola de MinIO         |
| PostgreSQL        | 5432             | Base de datos de Airflow       |
| Redis             | 6379             | Broker de Airflow              |
| MLflow            | 5001             | Tracking server de MLflow      |
| FastAPI           | 8000             | API REST para predicciones     |


## 🚧 Proximamente
- Agregar versionado de modelos (MLflow registry)
- Tests automáticos CI/CD
- Escalabilidad a Kubernetes (opcional)

---

💭 *Proyecto de referencia integrando orquestación, almacenamiento, tracking de modelos y APIs de inferencia en producción.*

