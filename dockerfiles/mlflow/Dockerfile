FROM python:3.11-slim

RUN apt-get update && apt-get install
RUN python -m pip install --upgrade pip

COPY requirements.txt /requirements.txt
RUN apt-get update && apt-get install -y wget
RUN python -m pip install -r requirements.txt
RUN pip install --upgrade pip
RUN pip install mlflow[extras] psycopg2-binary boto3 cryptography pymysql
RUN pip install --no-cache-dir -r /requirements.txt
#RUN airflow db upgrade

EXPOSE 5000