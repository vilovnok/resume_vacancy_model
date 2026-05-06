from typing import Optional
import time

from opentelemetry import trace

from .encoder import ONNXEncoder
from .retriever import Retriever
from metrics import (
    INFERENCE_REQUESTS,
    INFERENCE_LATENCY,
    ACTIVE_REQUESTS,
    MODEL_LOAD_TIME,
    GPU_MEMORY_USED,
)

tracer = trace.get_tracer(__name__)


class InferenceEngine:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        max_length: int = 512,
        use_cuda: bool = False,
        retriever_host: str = "localhost",
        retriever_port: int = 6333,
    ):
        self.model_name = tokenizer_path

        start = time.perf_counter()

        self.model = ONNXEncoder(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            max_length=max_length,
            use_cuda=use_cuda,
        )

        load_time = time.perf_counter() - start
        MODEL_LOAD_TIME.labels(model=self.model_name).set(load_time)

        if use_cuda:
            import torch

            if torch.cuda.is_available():
                mem_bytes = torch.cuda.memory_allocated()
                GPU_MEMORY_USED.labels(model=self.model_name, device="cuda").set(mem_bytes)

        self.retriever = Retriever(
            localhost=retriever_host,
            port=retriever_port,
        )

    def vectorize(self, sentences: list[str], batch_size: int = 32):
        embeddings = self.model.encode(
            sentences=sentences,
            convert_to_numpy=True,
            batch_size=batch_size,
        )["embeddings"]
        return embeddings

    def search(
        self,
        text: str,
        collection_name: str,
        topk: int = 5,
        filter_options: Optional[dict] = None,
    ):
        ACTIVE_REQUESTS.labels(model=self.model_name).inc()
        start_time = time.perf_counter()
        status = "success"
        results = None

        try:
            with tracer.start_as_current_span("engine.search") as span:
                span.set_attribute("text.length", len(text))
                span.set_attribute("topk", topk)
                span.set_attribute("collection", collection_name)

                with tracer.start_as_current_span("model.encode") as encode_span:
                    embedding = self.model.encode(
                        sentences=[text],
                        convert_to_numpy=True,
                    )["embeddings"]
                    encode_span.set_attribute("embedding.shape", str(embedding.shape))

                with tracer.start_as_current_span("retriever.search") as retr_span:
                    results = self.retriever.search(
                        embedding=embedding,
                        collection_name=collection_name,
                        topk=topk,
                        filter_options=filter_options,
                    )
                    retr_span.set_attribute("results.count", len(results.points))

            return results

        except Exception:
            status = "error"
            raise

        finally:
            total_latency = time.perf_counter() - start_time
            INFERENCE_LATENCY.labels(model=self.model_name).observe(total_latency)
            INFERENCE_REQUESTS.labels(status=status, model=self.model_name).inc()
            ACTIVE_REQUESTS.labels(model=self.model_name).dec()