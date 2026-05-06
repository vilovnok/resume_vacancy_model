from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from tasks.pipeline.task_download import download_dataset_from_minio
from tasks.pipeline.extract_module.task_extract_module_resume import extract_resumes
from tasks.pipeline.extract_module.task_extract_module_vacancy import extract_vacancies
from tasks.pipeline.task_sync_postgres_qdrant_dataset import load_metadata_into_postgres
from tasks.pipeline.task_dispatch_vectorization import dispatch_vectorization

from pathlib import Path
import shutil

from config import RUN_ID


BASE_DATA_DIR = Path(f"/opt/airflow/data/{RUN_ID}")


def clean_data_directory() -> None:
    data_dir = Path(BASE_DATA_DIR)

    if data_dir.exists():
        shutil.rmtree(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)


with DAG(
    dag_id="hr_matching_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["hr", "pipeline"],
) as dag:

    task_download = PythonOperator(
        task_id="download_dataset_from_minio",
        python_callable=download_dataset_from_minio,
    )

    task_extract_resumes = PythonOperator(
        task_id="extract_resumes",
        python_callable=extract_resumes,
    )
    
    task_extract_vacancies = PythonOperator(
        task_id="extract_vacancies",
        python_callable=extract_vacancies,
    )

    task_load_postgres = PythonOperator(
        task_id="task_sync_postgres_qdrant_dataset",
        python_callable=load_metadata_into_postgres,
    )

    upload_vectors_to_qdrant = PythonOperator(
        task_id="task_upload_vectors_to_qdrant",
        python_callable=dispatch_vectorization,
    )

    task_clean_data = PythonOperator(
        task_id="clean_data_directory",
        python_callable=clean_data_directory,
    )


    task_download >> [task_extract_resumes, task_extract_vacancies]
    [task_extract_resumes, task_extract_vacancies] >> task_load_postgres
    task_load_postgres >> upload_vectors_to_qdrant
    upload_vectors_to_qdrant >> task_clean_data