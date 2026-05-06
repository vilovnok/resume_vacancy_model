from dotenv import load_dotenv
import os

load_dotenv()


OTEL_ENDPOINT = os.getenv(
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "http://localhost:4317",
)

SERVICE_NAME = os.getenv(
    "OTEL_SERVICE_NAME",
    "model-inference",
)

OTEL_RESOURCE_ATTRS = os.getenv(
    "OTEL_RESOURCE_ATTRIBUTES",
    "",
)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "./models/model.onnx",
)

QDRANT_HOST = os.getenv(
    "QDRANT_HOST",
    "localhost",
)

QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

QDRANT_COLLECTION_RESUMES = os.getenv(
    "QDRANT_COLLECTION_RESUMES",
    "resumes_db",
)

QDRANT_COLLECTION_VACANCIES = os.getenv(
    "QDRANT_COLLECTION_VACANCIES",
    "vacancies_db",
)


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")