from fastapi import APIRouter, HTTPException
from app.schemas.query import Query
from app.core.services import get_ragie_chunks, generate_response

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
