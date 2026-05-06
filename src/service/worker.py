import time
from typing import Optional

from celery import Celery
from celery.signals import worker_ready

from datasets import Dataset

from config import (
    MODEL_PATH,
    QDRANT_COLLECTION_RESUMES,
    QDRANT_COLLECTION_VACANCIES,
    QDRANT_HOST,
    QDRANT_PORT,
    REDIS_HOST,
    REDIS_PORT,
)
from engine import InferenceEngine
from metrics import (
    ACTIVE_REQUESTS,
    INFERENCE_LATENCY,
    INFERENCE_REQUESTS,
    MODEL_LOAD_TIME,
)
from models import Document


celery = Celery(
    __name__,
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}",
)


_engine: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    global _engine

    if _engine is None:
        load_start = time.time()
        _engine = InferenceEngine(
            model_path=MODEL_PATH,
            tokenizer_path="deepvk/RuModernBERT-base",
            max_length=512,
            use_cuda=False,
            retriever_host=QDRANT_HOST,
            retriever_port=QDRANT_PORT,
        )
        MODEL_LOAD_TIME.labels(model="modernbert").set(time.time() - load_start)

    return _engine


def _ensure_qdrant_collections(engine: InferenceEngine) -> None:
    embeddings = engine.model.encode(
        sentences=["Hello, world"],
        convert_to_numpy=True,
        batch_size=32,
    )["embeddings"]

    engine.retriever.create_database(
        collection_name=QDRANT_COLLECTION_RESUMES,
        embedding=embeddings,
    )

    engine.retriever.create_database(
        collection_name=QDRANT_COLLECTION_VACANCIES,
        embedding=embeddings,
    )

@worker_ready.connect
def init_qdrant_collections(**kwargs) -> None:
    engine = get_engine()
    _ensure_qdrant_collections(engine)


@celery.task
def vectorize_task(documents: list[dict], collection_name: str):
    ACTIVE_REQUESTS.labels(model="modernbert").inc()

    try:
        INFERENCE_REQUESTS.labels(status="queued", model="modernbert").inc()

        engine = get_engine()
        dataset = build_dataset(documents, collection_name)

        upload_start = time.time()
        engine.retriever.upload_db(
            collection_name=collection_name,
            model=engine.model,
            dataset=dataset,
            batch_size=32,
        )
        INFERENCE_LATENCY.labels(model="modernbert").observe(
            time.time() - upload_start
        )

        INFERENCE_REQUESTS.labels(status="success", model="modernbert").inc()
    except Exception:
        INFERENCE_REQUESTS.labels(status="error", model="modernbert").inc()
        raise
    finally:
        ACTIVE_REQUESTS.labels(model="modernbert").dec()


def build_dataset(
    documents: list[Document],
    collection_name: str,
) -> Dataset:
    data = {
        "id": [],
        "text": [],
        "title": [],
        "skills": [],
        "description": [],
        "vacancy_ids": [],
    }

    for doc in documents:
        text = (
            f"Должность: {doc['title']};\n"
            f"Описание: {doc['description']};\n"
            f"Навыки: {doc['skills']}"
        )

        data["id"].append(str(doc["id"]))
        data["text"].append(text)
        data["title"].append(doc["title"])
        data["skills"].append(doc["skills"])
        data["description"].append(doc["description"])

        vacancy_ids = (
            doc.get("vacancy_ids", [])
            if isinstance(doc, dict)
            else getattr(doc, "vacancy_ids", []) or []
        )

        data["vacancy_ids"].append(vacancy_ids)

    return Dataset.from_dict(data)