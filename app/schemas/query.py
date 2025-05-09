from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime


class Query(BaseModel):
    text: str


class FilterCondition(BaseModel):
    """Base class for filter conditions"""

    pass


class EqFilter(FilterCondition):
    """Equality filter condition"""

    eq: Any


class InFilter(FilterCondition):
    """In filter condition"""

    in_: List[Any]


class NinFilter(FilterCondition):
    """Not in filter condition"""

    nin: List[Any]


class AndFilter(FilterCondition):
    """And filter condition"""

    and_: List[Dict[str, Any]]


class OrFilter(FilterCondition):
    """Or filter condition"""

    or_: List[Dict[str, Any]]


class RetrievalRequest(BaseModel):
    """Schema for retrieval requests"""

    query: str
    filter: Optional[Dict[str, Any]] = None
    rerank: Optional[bool] = True


class ScoredChunk(BaseModel):
    """Schema for scored chunks returned by Ragie"""

    text: str
    score: float
    document_id: str
    document_metadata: Dict[str, Any]


class RetrievalResponse(BaseModel):
    """Schema for retrieval responses"""

    scored_chunks: List[ScoredChunk]


class GenerationRequest(BaseModel):
    """Schema for generation requests"""

    query: str
    filter: Optional[Dict[str, Any]] = None
    rerank: Optional[bool] = True
    model: Optional[str] = "gpt-4o"
    system_prompt: Optional[str] = None


class GenerationResponse(BaseModel):
    """Schema for generation responses"""

    response: str


class SyncResponse(BaseModel):
    """Schema for sync responses"""

    message: str


class DocumentMetadata(BaseModel):
    """Schema for document metadata"""

    title: Optional[str] = None
    scope: Optional[str] = None
    # Add any other metadata fields as needed
    # These are just examples from the tutorial


class DocumentStatus(BaseModel):
    """Schema for document status response"""

    id: UUID
    created_at: datetime
    updated_at: datetime
    status: str
    name: str
    metadata: DocumentMetadata
    chunk_count: int
    external_id: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    """Schema for document upload response"""

    id: UUID
    status: str
