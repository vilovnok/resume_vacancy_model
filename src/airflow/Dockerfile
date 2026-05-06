ARG AIRFLOW_BASE_IMAGE=apache/airflow:2.9.3-python3.11
FROM ${AIRFLOW_BASE_IMAGE}

USER root

COPY requirements.txt /requirements.txt

USER airflow

RUN if [ -s /requirements.txt ]; then \
	pip install --no-cache-dir -r /requirements.txt; \
fi
