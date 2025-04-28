from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from schemas.query import (
    Query,
    RetrievalRequest,
    RetrievalResponse,
    GenerationRequest,
    GenerationResponse,
    SyncResponse,
    DocumentMetadata,
    DocumentStatus,
    DocumentUploadResponse,
)
from core.services import (
    get_ragie_chunks,
    generate_response,
    retrieve_chunks,
    generate_with_retrieval,
    sync_connection,
    upload_document,
    get_document_status,
)
from uuid import UUID
import tempfile
import os

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


@router.post("/connections/{connection_id}/sync", response_model=SyncResponse)
async def sync(connection_id: UUID):
    """
    Schedule a connector to sync as soon as possible
    """
    try:
        return sync_connection(connection_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_doc(
    file: UploadFile = File(...), metadata: str = Form(None), mode: str = Form("fast")
):
    """
    Upload a document to Ragie for processing
    """
    try:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Parse metadata if provided
        doc_metadata = None
        if metadata:
            doc_metadata = DocumentMetadata.model_validate_json(metadata)

        # Upload the document
        response = upload_document(
            file_path=temp_file_path, metadata=doc_metadata, mode=mode
        )

        # Clean up the temporary file
        os.unlink(temp_file_path)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}", response_model=DocumentStatus)
async def document_status(document_id: UUID):
    """
    Get the status of a document in Ragie
    """
    try:
        return get_document_status(document_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
