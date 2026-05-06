import time
from typing import Any, Sequence

import numpy as np
import requests
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from metrics import RETRIEVER_COUNT, RETRIEVER_LATENCY


class Mixin:
    def retrieve_points(
        self,
        point_ids: Sequence[int | str],
        collection_name: str,
    ) -> list[models.Record]:
        return self._client.retrieve(
            collection_name=collection_name,
            ids=list(point_ids),
        )


class Retriever(Mixin):
    vector_name: str = "deepvk/RuModernBERT-base"

    def __init__(
        self,
        localhost: str = "0.0.0.0",
        port: int = 6333,
    ) -> None:
        self._client = self._setup_database(localhost=localhost, port=port)

    def _setup_database(self, localhost: str, port: int) -> QdrantClient:
        url = f"http://{localhost}:{port}"

        try:
            response = requests.get(url, timeout=3)
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(f"Qdrant server is not running at {url}") from error

        return QdrantClient(url=url)

    @staticmethod
    def _to_vector(embedding: Any) -> list[float]:
        vector = np.asarray(embedding, dtype=np.float32).squeeze()

        if vector.ndim != 1:
            raise ValueError(f"Expected 1D embedding, got shape={vector.shape}")

        return vector.astype(float).tolist()


    def search(
        self,
        embedding: Any,
        collection_name: str,
        topk: int = 10,
        filter_options: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ):
        start_time = time.perf_counter()

        try:
            vector = self._to_vector(embedding)

            query_filter = None
            if filter_options:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                        for key, value in filter_options.items()
                    ]
                )

            result = self._client.query_points(
                collection_name=collection_name,
                query=vector,
                using=self.vector_name,
                with_payload=True,
                limit=topk,
                query_filter=query_filter,
                score_threshold=score_threshold,
            )

            RETRIEVER_COUNT.labels(status="success").inc()
            return result

        except Exception as error:
            RETRIEVER_COUNT.labels(status="error").inc()
            raise RuntimeError(f"Ошибка при поиске: {error}") from error

        finally:
            latency = time.perf_counter() - start_time
            RETRIEVER_LATENCY.observe(latency)


    def create_database(
        self,
        embedding: Any,
        collection_name: str = "HR",
    ) -> None:
        try:
            if self._client.collection_exists(collection_name):
                return

            vector = self._to_vector(embedding)
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    self.vector_name: models.VectorParams(
                        size=len(vector),
                        distance=models.Distance.COSINE,
                    )
                },
            )
        except Exception as error:
            if "already exists" in str(e):
                return
            raise


    def delete_database(self, collection_name: str) -> None:
        try:
            self._client.delete_collection(collection_name=collection_name)
        except Exception as error:
            raise RuntimeError(f"Ошибка при удалении базы: {error}") from error


    @staticmethod
    def _normalize_embeddings(embeddings: Any) -> np.ndarray:
        if isinstance(embeddings, dict) and "embeddings" in embeddings:
            embeddings = embeddings["embeddings"]

        return np.asarray(embeddings, dtype=np.float32)


    def upload_db(
        self,
        collection_name: str,
        model,
        dataset,
        batch_size: int = 4,
    ) -> None:
        for batch in tqdm(
            dataset.iter(batch_size=batch_size),
            total=(len(dataset) + batch_size - 1) // batch_size,
        ):
            try:
                texts = batch["text"]
                embeddings = self._normalize_embeddings(model.encode(texts))

                points = []
                for i, query_text in enumerate(texts):
                    points.append(
                        models.PointStruct(
                            id=str(batch["id"][i]),
                            vector={
                                self.vector_name: embeddings[i].astype(float).tolist(),
                            },
                            payload={
                                "title": batch["title"][i],
                                "skills": batch["skills"][i],
                                "description": batch["description"][i],
                                "vacancy_ids": batch.get("vacancy_ids", [])[i],
                            },
                        )
                    )

                self._client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,
                )

            except Exception as error:
                raise RuntimeError(
                    f"Ошибка при загрузке в базу: {error}"
                ) from error


    def upload_db(
        self,
        collection_name: str,
        model,
        dataset,
        batch_size: int = 4,
    ) -> None:
        for batch in tqdm(
            dataset.iter(batch_size=batch_size),
            total=(len(dataset) + batch_size - 1) // batch_size,
        ):
            try:
                texts = batch["text"]
                embeddings = self._normalize_embeddings(model.encode(texts))

                points = []
                for i, _ in enumerate(texts):
                    payload = {
                        "title": batch["title"][i],
                        "skills": batch["skills"][i],
                        "description": batch["description"][i],
                    }

                    if "vacancy_ids" in batch:
                        payload["vacancy_ids"] = batch["vacancy_ids"][i]

                    points.append(
                        models.PointStruct(
                            id=str(batch["id"][i]),
                            vector={
                                self.vector_name: embeddings[i].astype(float).tolist(),
                            },
                            payload=payload,
                        )
                    )

                self._client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,
                )

            except Exception as error:
                raise RuntimeError(f"Ошибка при загрузке в базу: {error}") from error