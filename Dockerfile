FROM apache/airflow:2.4.1

USER root

RUN apt-get update -o Acquire::AllowInsecureRepositories=true && apt-get install -y libgomp1


USER airflow
