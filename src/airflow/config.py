from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION_RESUMES = os.getenv("QDRANT_COLLECTION_RESUMES", "resumes_db")
QDRANT_COLLECTION_VACANCIES = os.getenv("QDRANT_COLLECTION_VACANCIES", "vacancies_db")

AITUNNEL_API_KEY = os.getenv("AITUNNEL_API_KEY")
AITUNNEL_URL = os.getenv("AITUNNEL_URL")

VECTORIZATION_SERVICE_URL = os.getenv("VECTORIZATION_SERVICE_URL")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", "admin")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", "password123")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

RUN_ID = os.getenv("RUN_ID", "hr_matching_pipeline")