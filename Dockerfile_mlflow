FROM python:3.10

RUN pip install --no-cache-dir \
    mlflow \
    psycopg2-binary

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri postgresql+psycopg2://mlflow:mlflow@mlflow-postgres:5432/mlflow \
    --default-artifact-root /mlflow \
    --host 0.0.0.0
