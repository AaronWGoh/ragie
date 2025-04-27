import requests
from openai import OpenAI
from app.core.config import RAGIE_API_KEY, OPENAI_API_KEY
from app.schemas.query import (
    RetrievalRequest,
    RetrievalResponse,
    ScoredChunk,
    GenerationRequest,
    GenerationResponse,
)

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def get_ragie_chunks(query_text: str) -> list[str]:
    """
    Retrieve relevant chunks from Ragie API
    """
    ragie_response = requests.post(
        "https://api.ragie.ai/retrievals",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RAGIE_API_KEY}",
        },
        json={"query": query_text},
    )

    if not ragie_response.ok:
        raise Exception(f"Ragie API error: {ragie_response.text}")

    data = ragie_response.json()
    return [chunk["text"] for chunk in data.get("scored_chunks", [])]


def retrieve_chunks(request: RetrievalRequest) -> RetrievalResponse:
    """
    Retrieve chunks from Ragie API with optional filtering and reranking
    """
    # Prepare the request payload
    payload = {"query": request.query, "rerank": request.rerank}

    # Add filter if provided
    if request.filter:
        payload["filter"] = request.filter

    # Make the API request
    ragie_response = requests.post(
        "https://api.ragie.ai/retrievals",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RAGIE_API_KEY}",
        },
        json=payload,
    )

    if not ragie_response.ok:
        raise Exception(f"Ragie API error: {ragie_response.text}")

    # Parse the response into our schema
    data = ragie_response.json()
    scored_chunks = [
        ScoredChunk(
            text=chunk["text"],
            score=chunk["score"],
            document_id=chunk["document_id"],
            document_metadata=chunk["document_metadata"],
        )
        for chunk in data.get("scored_chunks", [])
    ]

    return RetrievalResponse(scored_chunks=scored_chunks)


def generate_response(
    query_text: str, chunk_texts: list[str], model: str = "gpt-4o"
) -> str:
    """
    Generate response using OpenAI with retrieved chunks
    """
    if not chunk_texts:
        return "No relevant information found for your query."

    system_prompt = f"""You are a helpful AI assistant. Use the following information to answer the user's question.
    If the information is not sufficient to answer the question, say so.

    Retrieved information:
    {' '.join(chunk_texts)}
    """

    completion = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_text},
        ],
    )

    return completion.choices[0].message.content


def generate_with_retrieval(request: GenerationRequest) -> GenerationResponse:
    """
    Generate a response using Ragie retrieval and OpenAI generation
    """
    # First retrieve relevant chunks
    retrieval_request = RetrievalRequest(
        query=request.query, filter=request.filter, rerank=request.rerank
    )
    retrieval_response = retrieve_chunks(retrieval_request)

    if not retrieval_response.scored_chunks:
        return GenerationResponse(
            response="No relevant information found for your query."
        )

    # Extract chunk texts
    chunk_texts = [chunk.text for chunk in retrieval_response.scored_chunks]

    # Use custom system prompt if provided, otherwise use default
    if request.system_prompt:
        system_prompt = request.system_prompt
    else:
        system_prompt = """These are very important to follow:

You are "Ragie AI", a professional but friendly AI chatbot working as an assistant to the user.

Your current task is to help the user based on all of the information available to you shown below.
Answer informally, directly, and concisely without a heading or greeting but include everything relevant.
Use richtext Markdown when appropriate including bold, italic, paragraphs, and lists when helpful.
If using LaTeX, use double $$ as delimiter instead of single $. Use $$...$$ instead of parentheses.
Organize information into multiple sections or points when appropriate.
Don't include raw item IDs or other raw fields from the source.
Don't use XML or other markup unless requested by the user.

Here is all of the information available to answer the user:
===
{chunk_texts}
===

If the user asked for a search and there are no results, make sure to let the user know that you couldn't find anything,
and what they might be able to do to find the information they need.

END SYSTEM INSTRUCTIONS"""

    # Format the system prompt with the chunks
    formatted_system_prompt = system_prompt.format(chunk_texts="\n".join(chunk_texts))

    # Generate the response using OpenAI
    completion = openai_client.chat.completions.create(
        model=request.model,
        messages=[
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": request.query},
        ],
    )

    return GenerationResponse(response=completion.choices[0].message.content)
