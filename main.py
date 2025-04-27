from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Ragie RAG System")

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Get Ragie API key from environment
RAGIE_API_KEY = os.getenv("RAGIE_API_KEY")
if not RAGIE_API_KEY:
    raise ValueError("RAGIE_API_KEY environment variable is not set")


class Query(BaseModel):
    text: str


@app.post("/query")
async def process_query(query: Query):
    """
    Process a query using Ragie's RAG system
    """
    try:
        # Step 1: Get relevant chunks from Ragie
        ragie_response = requests.post(
            "https://api.ragie.ai/retrievals",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {RAGIE_API_KEY}",
            },
            json={"query": query.text},
        )

        if not ragie_response.ok:
            raise HTTPException(
                status_code=ragie_response.status_code,
                detail=f"Ragie API error: {ragie_response.text}",
            )

        data = ragie_response.json()
        chunk_texts = [chunk["text"] for chunk in data.get("scored_chunks", [])]

        if not chunk_texts:
            return {"response": "No relevant information found for your query."}

        # Step 2: Create a system prompt with the retrieved chunks
        system_prompt = f"""You are a helpful AI assistant. Use the following information to answer the user's question.
        If the information is not sufficient to answer the question, say so.

        Retrieved information:
        {' '.join(chunk_texts)}
        """

        # Step 3: Generate response using OpenAI
        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query.text},
            ],
        )

        return {"response": completion.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
