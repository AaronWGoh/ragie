from fastapi import APIRouter, HTTPException
from app.schemas.query import (
    Query,
    RetrievalRequest,
    RetrievalResponse,
    GenerationRequest,
    GenerationResponse,
)
from app.core.services import (
    get_ragie_chunks,
    generate_response,
    retrieve_chunks,
    generate_with_retrieval,
)

router = APIRouter()


@router.post("/query")
async def process_query(query: Query):
    """
    Process a query using Ragie's RAG system
    """
    try:
        # Get relevant chunks from Ragie
        chunk_texts = get_ragie_chunks(query.text)

        # Generate response using OpenAI
        response = generate_response(query.text, chunk_texts)

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest):
    """
    Retrieve chunks from Ragie with optional filtering and reranking
    """
    try:
        return retrieve_chunks(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """
    Generate a response using Ragie retrieval and OpenAI generation
    """
    try:
        return generate_with_retrieval(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
