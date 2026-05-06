from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)



INFERENCE_REQUESTS = Counter(
    "inference_requests_total",
    "Total number of inference requests",
    ["status", "model"]
)

INFERENCE_LATENCY = Histogram(
    "inference_request_duration_seconds",
    "Inference request latency in seconds",
    ["model"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0]
)

ACTIVE_REQUESTS = Gauge(
    "inference_active_requests",
    "Number of currently active inference requests",
    ["model"]
)

MODEL_LOAD_TIME = Gauge(
    "inference_model_load_seconds",
    "Time taken to load the model",
    ["model"]
)

GPU_MEMORY_USED = Gauge(
    "inference_gpu_memory_bytes",
    "GPU memory used by the model",
    ["model", "device"]
)

TOKENS_PROCESSED = Counter(
    "inference_tokens_processed_total",
    "Total number of tokens processed",
    ["model"]
)

BATCH_SIZE = Histogram(
    "inference_batch_size",
    "Distribution of batch sizes",
    ["model"],
    buckets=[1, 2, 4, 8, 16, 32, 64]
)

QUEUE_SIZE = Gauge(
    "inference_queue_size",
    "Number of requests waiting in queue",
    ["model"]
)

RETRIEVER_COUNT = Counter(
    "retriever_requests_total",
    "Total number of retriever search requests",
    ["status"],
)

RETRIEVER_LATENCY = Histogram(
    "retriever_request_duration_seconds",
    "Retriever search latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0],
)

MATCH_COUNT = Counter(
    "match_requests_total",
    "Total borrower match requests",
    ["status"]
)

MATCH_LATENCY = Histogram(
    "match_request_duration_seconds",
    "Borrower match request latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5]
)