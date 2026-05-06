#!/usr/bin/env bash
set -euo pipefail

echo "[airflow-init] ⏳ Waiting for airflow-db..."

until nc -z airflow-db 5432; do
  echo "[airflow-init] Waiting for PostgreSQL..."
  sleep 2
done

echo "[airflow-init] Ensuring database exists..."


PGPASSWORD=airflow psql -h airflow-db -U airflow -tc "SELECT 1 FROM pg_database WHERE datname = 'airflow'" | grep -q 1 || \
PGPASSWORD=airflow psql -h airflow-db -U airflow -c "CREATE DATABASE airflow;"

echo "[airflow-init] Migrating metadata DB..."
airflow db migrate

echo "[airflow-init] Creating admin user (idempotent)..."
airflow users create \
  --role Admin \
  --username "${_AIRFLOW_WWW_USER_USERNAME}" \
  --password "${_AIRFLOW_WWW_USER_PASSWORD}" \
  --firstname "${_AIRFLOW_WWW_USER_FIRSTNAME}" \
  --lastname "${_AIRFLOW_WWW_USER_LASTNAME}" \
  --email "${_AIRFLOW_WWW_USER_EMAIL}" \
  || true

echo "[airflow-init] Creating default connections..."
airflow connections create-default-connections || true

echo "[airflow-init] Done."