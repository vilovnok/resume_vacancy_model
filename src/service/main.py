import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

import cache as cache_layer
from config import (
    MODEL_PATH,
    OTEL_ENDPOINT,
    OTEL_RESOURCE_ATTRS,
    QDRANT_COLLECTION_RESUMES,
    QDRANT_COLLECTION_VACANCIES,
    QDRANT_HOST,
    QDRANT_PORT,
    SERVICE_NAME,
)
from engine import InferenceEngine
from metrics import (
    MATCH_COUNT, MATCH_LATENCY, 
    ACTIVE_REQUESTS, INFERENCE_REQUESTS, 
    QUEUE_SIZE, TOKENS_PROCESSED, INFERENCE_LATENCY,
    RETRIEVER_COUNT, RETRIEVER_LATENCY
)
from models import (
    IngestRequest,
    IngestResponse,
    MatchRequest,
    MatchResponse,
    SearchByIdRequest,
)
from worker import vectorize_task

from utils import make_cache_key, build_text, build_match_output


resource_attrs = {"service.name": SERVICE_NAME}

if OTEL_RESOURCE_ATTRS:
    for attr in OTEL_RESOURCE_ATTRS.split(","):
        if "=" in attr:
            key, value = attr.split("=", 1)
            resource_attrs[key] = value

resource = Resource.create(resource_attrs)

provider = TracerProvider(resource=resource)
otlp_exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = InferenceEngine(
        model_path=MODEL_PATH,
        tokenizer_path="deepvk/RuModernBERT-base",
        max_length=512,
        use_cuda=False,
        retriever_host=QDRANT_HOST,
        retriever_port=QDRANT_PORT,
    )

    try:
        engine.model.encode(["warmup"])
    except Exception as e:
        print(f"Warmup failed: {e}")

    app.state.engine = engine
    yield
    app.state.engine = None


app = FastAPI(
    title="ModernBert ONNX Inference Server",
    description="ModernBert inference with Prometheus + OpenTelemetry",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

FastAPIInstrumentor.instrument_app(app)


COLLECTION_MAP = {
    "resume": QDRANT_COLLECTION_RESUMES,
    "vacancy": QDRANT_COLLECTION_VACANCIES,
}



@app.get("/v1/health")
async def health(request: Request):
    engine = request.app.state.engine

    return {
        "status": "healthy",
        "model_loaded": bool(engine and engine.model and engine.model.session),
    }


@app.get("/v1/metrics", response_class=PlainTextResponse)
async def metrics():
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/v1/vectorize", response_model=IngestResponse)
async def vectorize(request: IngestRequest, req: Request):
    _ = req.app.state.engine

    ACTIVE_REQUESTS.labels(model="modernbert").inc()
    start_time = time.time()

    try:
        documents = [doc.model_dump() for doc in request.documents]

        QUEUE_SIZE.labels(model="modernbert").set(len(documents))
        INFERENCE_REQUESTS.labels(status="queued", model="modernbert").inc()

        vectorize_task.delay(
            documents,
            request.collection_name,
        )

        return IngestResponse(
            status="queued",
            processed=len(documents),
        )

    except Exception:
        INFERENCE_REQUESTS.labels(status="error", model="modernbert").inc()
        raise

    finally:
        ACTIVE_REQUESTS.labels(model="modernbert").dec()


@app.post("/v1/search_by_text", response_model=MatchResponse)
async def search_by_text(request: MatchRequest, req: Request):
    engine = req.app.state.engine

    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    collection = COLLECTION_MAP.get(request.entity_type)
    if collection is None:
        raise HTTPException(status_code=400, detail="Invalid entity_type")

    cache_key = make_cache_key(
        content=request.content,
        entity_type=request.entity_type,
        top_k=request.top_k
    )

    ACTIVE_REQUESTS.labels(model="modernbert").inc()
    start_time = time.time()

    with tracer.start_as_current_span("search_by_text") as span:
        try:
            cached = cache_layer.get(cache_key, request.top_k)

            if cached:
                latency = time.time() - start_time

                MATCH_COUNT.labels(status="cache_hit").inc()
                MATCH_LATENCY.observe(latency)

                return MatchResponse(
                    matches=cached,
                    latency_ms=latency * 1000,
                    cached=True,
                )

            span.set_attribute("input.length", len(request.content))
            span.set_attribute("top_k", request.top_k)

            encode_start = time.time()

            results = await asyncio.to_thread(
                engine.search,
                request.content,
                collection,
                request.top_k,
            )

            encode_latency = time.time() - encode_start

            INFERENCE_LATENCY.labels(model="modernbert").observe(encode_latency)

            TOKENS_PROCESSED.labels(model="modernbert").inc(
                len(engine.tokenizer.encode(request.content))
            )


            output = build_match_output(results.points)

            latency = time.time() - start_time

            MATCH_LATENCY.observe(latency)
            MATCH_COUNT.labels(status="success").inc()
            RETRIEVER_COUNT.labels(status="success").inc()

            span.set_attribute("results.count", len(output))
            span.set_status(Status(StatusCode.OK))

            cache_layer.set(cache_key, request.top_k, output)

            return MatchResponse(
                matches=output,
                cached=False,
                latency_ms=latency * 1000,
            )

        except Exception as e:
            MATCH_COUNT.labels(status="error").inc()
            RETRIEVER_COUNT.labels(status="error").inc()

            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)

            raise HTTPException(status_code=500, detail=str(e))

        finally:
            ACTIVE_REQUESTS.labels(model="modernbert").dec()


@app.post("/v1/search_by_id", response_model=MatchResponse)
async def search_by_id(request: SearchByIdRequest, req: Request):
    engine = req.app.state.engine

    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    cache_key = make_cache_key(
        content=f"id:{request.id}",
        top_k=request.top_k,
    )

    ACTIVE_REQUESTS.labels(model="modernbert").inc()
    start_time = time.time()

    with tracer.start_as_current_span("search_by_id") as span:
        try:
            cached = cache_layer.get(cache_key, request.top_k)

            if cached:
                latency = time.time() - start_time

                MATCH_COUNT.labels(status="cache_hit").inc()
                MATCH_LATENCY.observe(latency)

                return MatchResponse(
                    matches=cached,
                    cached=True,
                    latency_ms=latency * 1000,
                )

            retriever_start = time.time()

            items = await asyncio.to_thread(
                engine.retriever.retrieve_points,
                [request.id],
                QDRANT_COLLECTION_VACANCIES,
            )

            if not items:
                latency = time.time() - start_time

                RETRIEVER_LATENCY.observe(time.time() - retriever_start)
                MATCH_COUNT.labels(status="success").inc()
                MATCH_LATENCY.observe(latency)

                span.set_attribute("search_by_id_results", 0)
                span.set_status(Status(StatusCode.OK))

                return MatchResponse(
                    matches=[],
                    cached=False,
                    latency_ms=latency * 1000,
                )

            vacancy = items[0].payload
            text = build_text(vacancy)

            inference_start = time.time()
            embedding = engine.model.encode(
                sentences=[text],
                convert_to_numpy=True,
            )["embeddings"]
            INFERENCE_LATENCY.labels(model="modernbert").observe(
                time.time() - inference_start
            )

            filter_options = {"vacancy_ids": request.id}

            results = await asyncio.to_thread(
                engine.retriever.search,
                embedding=embedding,
                collection_name=QDRANT_COLLECTION_RESUMES,
                topk=request.top_k,
                filter_options=filter_options,
            )

            RETRIEVER_LATENCY.observe(time.time() - retriever_start)

            output = build_match_output(results.points)

            latency = time.time() - start_time

            MATCH_LATENCY.observe(latency)
            MATCH_COUNT.labels(status="success").inc()

            cache_layer.set(cache_key, request.top_k, output)

            span.set_attribute("search_by_id_results", len(output))
            span.set_status(Status(StatusCode.OK))

            return MatchResponse(
                matches=output,
                cached=False,
                latency_ms=latency * 1000,
            )

        except Exception as e:
            MATCH_COUNT.labels(status="error").inc()
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            ACTIVE_REQUESTS.labels(model="modernbert").dec()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("SERVER_PORT", "8080")),
        workers=int(os.getenv("SERVER_WORKERS", "1")),
    )