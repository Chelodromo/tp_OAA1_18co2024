# 🛠 Proyecto Airflow + MinIO + ML Pipeline

Este proyecto orquesta un flujo de procesamiento de datos y entrenamiento de un modelo SVM usando **Apache Airflow**, con almacenamiento de datos en **MinIO (S3 compatible)**. Todo está dockerizado y configurado para correr automáticamente.

## 📦 Estructura del Proyecto

```
.
├── dags/                       # DAGs de Airflow
├── scripts/                   # Scripts auxiliares (opcional)
├── logs/                      # Carpeta ignorada por Git (Airflow logs)
├── datalake/                  # Carpeta local (opcional) para almacenamiento temporal
├── docker-compose.yml         # Definición de servicios
└── .gitignore                 # Ignora logs, datalake, etc.
```

## 🔁 Flujo del DAG principal

1. **`crear_bucket_minio`**  
   Conexión automática a MinIO vía `S3Hook`. Verifica si existe el bucket `datalake`, y si no, lo crea.

2. **`check_minio_connection`**  
   Escribe un archivo `conexion_exitosa.txt` en el bucket `datalake` para confirmar la conexión.

3. **`descargar_csv`**  
   Descarga un archivo CSV desde Google Drive a `/opt/airflow/datalake/df_merged.csv`. Este es el dataset usado para la prediccion de polvo de la materia AdM 1. 

4. **`mostrar_head`**  
   Lee el CSV con `pandas`, convierte la columna `date` a datetime, elimina columnas no deseadas y guarda un backup en formato JSON.
   Hace un preproceso de los datos para que queden listos para ingresar el modelo. Respalda el dataset como un JSON.

6. **`split_dataset`**  
   - Convierte la columna `Date` a `timestamp`.
   - Elimina columnas no relevantes para el modelo.
   - Realiza un `train_test_split` y guarda `X_train`, `X_test`, `y_train`, `y_test` como JSON en la carpeta local de respaldo.

7. **`svm_modeling`**  
   - Escala los datos con `StandardScaler`.
   - Entrena un `SVC(kernel='linear')`.
   - Hace `cross_val_score` y muestra la matriz de confusión.
   - Todo se loguea en el contenedor Airflow.

## 🌐 MinIO (S3 compatible)

- Se levanta en el contenedor `minio`.
- Console: [http://localhost:9001](http://localhost:9001)
- API: [http://localhost:9000](http://localhost:9000)
- Usuario/Contraseña: `minioadmin / minioadmin`
- Bucket usado: `datalake`

Airflow se conecta a MinIO automáticamente con:

```yaml
environment:
  AIRFLOW_CONN_MINIO_S3: s3://minioadmin:minioadmin@minio:9000/?endpoint_url=http%3A%2F%2Fminio%3A9000
```

## 🧹 Git

- Se ignoran los logs y archivos temporales:
```
logs/
datalake/
```

Para limpiar logs ya subidos:

```bash
git rm -r --cached logs/
echo "logs/" >> .gitignore
git commit -m "Ignorar carpeta logs"
git push
```

## 🚀 Cómo levantar todo

```bash
docker-compose up -d
```

Una vez levantado, accedé a Airflow:
- [http://localhost:8080](http://localhost:8080)
- Usuario: `airflow`, Password: `airflow` (según imagen oficial)

## Pasos a Seguir
   - Lograr respaldar los archivos en el bucket conectado usando S3.hook()
   - Probar otros modelos
   - Respaldar en el bucket un modelo en plk
   - Tomar con una nueva tarea ese modelo creado y probarlo con un set de datos al azar para hacer un predict
