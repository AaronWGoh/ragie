import requests
from openai import OpenAI
from app.core.config import RAGIE_API_KEY, OPENAI_API_KEY

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


def generate_response(query_text: str, chunk_texts: list[str]) -> str:
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
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_text},
        ],
    )

    return completion.choices[0].message.content
