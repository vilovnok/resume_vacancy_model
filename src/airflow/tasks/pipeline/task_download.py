import os
from datetime import datetime
import logging
from minio import Minio
from minio.error import S3Error

from config import MINIO_ENDPOINT, MINIO_ROOT_USER, MINIO_ROOT_PASSWORD, RUN_ID

logger = logging.getLogger("hr_matcher")


def get_minio_client():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ROOT_USER,
        secret_key=MINIO_ROOT_PASSWORD,
        secure=False
    )


def download_if_needed(client, bucket, object_name, output_path):
    try:
        stat = client.stat_object(bucket, object_name)
        minio_last_modified = stat.last_modified.replace(tzinfo=None)

        if os.path.exists(output_path):
            local_mtime = datetime.fromtimestamp(os.path.getmtime(output_path))

            if local_mtime.date() == minio_last_modified.date():
                logger.info(f"✅ {object_name} is up-to-date. Skipping.")
                return

            logger.info(f"🔁 Updating {object_name}...")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        client.fget_object(bucket, object_name, output_path)

        logger.info(f"📦 Downloaded {object_name} → {output_path}")

    except S3Error as e:
        logger.error(f"❌ MinIO error while downloading {object_name}: {e}")
        raise


def download_dataset_from_minio():
    client = get_minio_client()
    bucket_name = "dataset"

    files_config = [
        {
            "object": "resumes.zip",
            "path": f"/opt/airflow/data/{RUN_ID}/resumes.zip"
        },
        {
            "object": "vacancies.zip",
            "path": f"/opt/airflow/data/{RUN_ID}/vacancies.zip"
        },
        {
            "object": "resumes_meta.csv",
            "path": f"/opt/airflow/data/{RUN_ID}/resumes_meta.csv"
        },
        {
            "object": "vacancies_meta.csv",
            "path": f"/opt/airflow/data/{RUN_ID}/vacancies_meta.csv"
        },
        {
            "object": "applications.csv",
            "path": f"/opt/airflow/data/{RUN_ID}/applications.csv"
        },
        {
            "object": "irrelevant_jobs.csv",
            "path": f"/opt/airflow/data/{RUN_ID}/irrelevant_jobs.csv"
        }
    ]

    for file in files_config:
        object_name = f"{RUN_ID}/{file['object']}"
        
        download_if_needed(
            client,
            bucket_name,
            object_name,
            file["path"]
        )

    logger.info("✅ All dataset files downloaded successfully")