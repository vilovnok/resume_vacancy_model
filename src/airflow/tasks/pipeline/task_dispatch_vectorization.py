import os
import logging
import requests
from math import ceil
from sqlalchemy import create_engine, text

from config import (
    VECTORIZATION_SERVICE_URL,
    QDRANT_COLLECTION_VACANCIES, QDRANT_COLLECTION_RESUMES)


logger = logging.getLogger("hr_matcher")


BATCH_SIZE = 64


def send_batch(documents, collection_name: str):
    """
    Отправка батча на embedding/vectorization сервис
    """
    payload = {
        "documents": documents,
        "collection_name": collection_name,
    }

    res = requests.post(
        f"{VECTORIZATION_SERVICE_URL}/v1/vectorize",
        json=payload,
        timeout=120
    )


    if not res.ok:
        raise RuntimeError(
            f"Vectorization failed for {collection_name}: {res.text}"
        )


def send_in_batches(documents, collection_name: str):
    total = len(documents)
    batches = ceil(total / BATCH_SIZE)

    logger.info(
        f"📦 Sending {total} docs to `{collection_name}` in {batches} batches"
    )

    for i in range(batches):
        batch = documents[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

        send_batch(batch, collection_name)

        logger.info(f"✅ Batch {i + 1}/{batches} sent to {collection_name}")


def dispatch_vectorization():

    logger.info("📥 Loading data from Postgres...")

    db_url = os.environ["DATABASE_URL"]
    engine = create_engine(db_url)

    with engine.connect() as conn:

        vacancies = conn.execute(text("""
            SELECT id, filename, title, description, skills
            FROM vacancies
        """)).mappings().all()

        resumes = conn.execute(text("""
            SELECT r.id, r.filename, r.title, r.description, r.skills,
                   COALESCE(ARRAY_AGG(a.vacancy_id), '{}') AS vacancy_ids
            FROM resumes r
            LEFT JOIN applications a ON r.id = a.resume_id
            GROUP BY r.id, r.filename, r.title, r.description, r.skills
        """)).mappings().all()

    logger.info(
        f"📊 Loaded {len(vacancies)} vacancies and {len(resumes)} resumes"
    )

    vacancy_documents = [
        {
            "id": str(row["id"]),
            "title": row["title"],
            "description": row["description"],
            "skills": row["skills"],
        }
        for row in vacancies
    ]

    resume_documents = [
        {
            "id": str(row["id"]),
            "title": row["title"],
            "description": row["description"],
            "skills": row["skills"],
            "vacancy_ids": (
                [str(v) for v in row["vacancy_ids"]]
                if row["vacancy_ids"] else []
            ),
        }
        for row in resumes
    ]

    send_in_batches(
        resume_documents,
        QDRANT_COLLECTION_RESUMES,
    )

    send_in_batches(
        vacancy_documents,
        QDRANT_COLLECTION_VACANCIES,
    )

    logger.info("🚀 Vectorization dispatch completed")