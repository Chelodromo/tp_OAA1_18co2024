# fastapi_app/Dockerfile

FROM python:3.9-slim

# Instalamos librerías del sistema que faltan (incluyendo libgomp1 para LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==1.22.4 scikit-learn==1.0.2 lightgbm==3.3.5 fastapi uvicorn pandas
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]