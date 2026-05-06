from pydantic import BaseModel

from typing import Optional



class InferenceRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[list[str]] = None
    inputs: Optional[dict] = None  
    include_embeddings: bool = False


class InferenceResponse(BaseModel):
    embeddings: Optional[list] = None
    pooler_output: Optional[list] = None
    latency_ms: float
    batch_size: int
    tokens_processed: int


class MatchRequest(BaseModel):
    content: str
    entity_type: str
    top_k: int = 5


class MatchResponse(BaseModel):
    matches: list[dict]
    cached: bool
    latency_ms: float


class Document(BaseModel):
    id: str
    title: str
    description: str
    skills: str
    vacancy_ids: Optional[list[str]] = None


class IngestRequest(BaseModel):
    documents: list[Document]
    collection_name: str


class IngestResponse(BaseModel):
    status: str
    processed: int


class SearchByIdRequest(BaseModel):
    id: str
    top_k: int = 5