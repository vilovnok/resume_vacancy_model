import os
import time
from typing import Any, Optional, Union

import numpy as np
import onnxruntime as ort
import torch
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Gauge, Histogram
from tqdm import tqdm
from transformers import AutoTokenizer

from metrics import (
    ACTIVE_REQUESTS,
    BATCH_SIZE,
    MODEL_LOAD_TIME,
    INFERENCE_REQUESTS,
    INFERENCE_LATENCY,
    TOKENS_PROCESSED,
)

tracer = trace.get_tracer(__name__)



class ONNXEncoder:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        max_length: int = 512,
        use_cuda: bool = True,
        gpu_mem_limit: int = 4 * 1024 * 1024 * 1024,
    ):
        self.model_path = model_path
        self.model_name = "onnx-encoder"
        self.max_length = max_length
        self.session = None
        self.execution_provider = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"

        start_time = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": gpu_mem_limit,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            }),
            "CPUExecutionProvider"
        ] if use_cuda else ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = int(os.getenv("ONNX_INTRA_OP_THREADS", "4"))
        sess_options.inter_op_num_threads = int(os.getenv("ONNX_INTER_OP_THREADS", "2"))

        if os.path.exists(self.model_path):
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 64,
        convert_to_numpy: bool = False,
        show_progress_bar: bool = False,
        device: Optional[torch.device] = None,
        normalize_embeddings: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:

        if len(sentences) == 0:
            if convert_to_numpy:
                return np.empty((0, 0), dtype=np.float32)
            return torch.empty((0, 0), dtype=torch.float32)

        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)

        if show_progress_bar:
            iterator = tqdm(
                iterator,
                desc="Encoding Sentences",
                total=(len(sentences) + batch_size - 1) // batch_size,
            )

        for start_idx in iterator:
            batch_sentences = sentences[start_idx : start_idx + batch_size]

            encoded = self.tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            outputs = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )

            emb = outputs[0].astype(np.float32)

            if normalize_embeddings:
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms = np.clip(norms, a_min=1e-12, a_max=None)
                emb = emb / norms

            all_embeddings.append(emb)

        all_embeddings = np.concatenate(all_embeddings, axis=0)

        if convert_to_numpy:
            return {
                "embeddings":all_embeddings
            }

        return {
            "embeddings":torch.from_numpy(all_embeddings)
        }


    def infer(self, inputs: dict) -> dict:
        ACTIVE_REQUESTS.labels(model=self.model_name).inc()

        with tracer.start_as_current_span("modernbert.inference") as span:
            try:
                start_time = time.time()

                batch_size = len(inputs["input_ids"])
                tokens = int(np.sum(inputs["attention_mask"]))

                span.set_attribute("model.name", self.model_name)
                span.set_attribute("inference.batch_size", batch_size)
                span.set_attribute("inference.tokens", tokens)
                span.set_attribute("inference.execution_provider", self.execution_provider)

                with tracer.start_as_current_span("onnx.session.run") as onnx_span:
                    onnx_span.set_attribute("onnx.provider", self.execution_provider)
                    outputs = self.session.run(
                        None,
                        {
                            "input_ids": inputs["input_ids"],
                            "attention_mask": inputs["attention_mask"],
                        }
                    )

                latency = time.time() - start_time

                result = {
                    "last_hidden_state": outputs[0].tolist(),
                    "pooler_output": outputs[1].tolist() if len(outputs) > 1 else None
                }

                span.set_attribute("inference.latency_ms", latency * 1000)

                INFERENCE_REQUESTS.labels(status="success", model=self.model_name).inc()
                INFERENCE_LATENCY.labels(model=self.model_name).observe(latency)
                TOKENS_PROCESSED.labels(model=self.model_name).inc(tokens)
                BATCH_SIZE.labels(model=self.model_name).observe(batch_size)

                span.set_status(Status(StatusCode.OK))

                return {
                    **result,
                    "latency_ms": latency * 1000,
                    "batch_size": batch_size,
                }

            except Exception as e:
                REQUEST_COUNT.labels(status="error", model=self.model_name).inc()
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            finally:
                ACTIVE_REQUESTS.labels(model=self.model_name).dec()
