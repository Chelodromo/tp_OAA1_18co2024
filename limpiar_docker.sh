#!/bin/bash

echo "🛑 Bajando todos los servicios Docker..."
docker-compose down --volumes --remove-orphans

echo "🧹 Eliminando contenedores detenidos y redes no utilizadas..."
docker system prune --volumes -f

echo "🗑 Limpiando contenido de MinIO (si los datos persisten localmente)..."
rm -rf ./minio_data/*

echo "📁 Limpiando directorios de logs de Airflow y su base de datos..."
rm -rf ./airflow/logs/*
rm -rf ./airflow/db.sqlite3
rm -rf ./airflow/dags/__pycache__/*

echo "🧪 Limpiando artefactos de modelos, MLflow y resultados antiguos..."
rm -rf ./mlruns/
rm -rf ./modelos/*
rm -rf ./datalake/*

echo "♻️ Borrando imágenes antiguas (opcional, si querés liberar espacio)..."
# docker image prune -a -f

echo "✅ Entorno limpio. Listo para reiniciar desde cero."

