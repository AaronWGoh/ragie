# Ragie RAG System MVP

A simple MVP implementation of a RAG (Retrieval-Augmented Generation) system using Ragie and OpenAI.

## Setup

1. Create and activate a virtual environment using uv:
```bash
# Install uv if you haven't already
pip install uv

# Create a virtual environment
uv venv

# Activate the virtual environment
# On Unix/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install project dependencies
uv pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:
```bash
RAGIE_API_KEY=your_ragie_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

3. Run the server:
```bash
python main.py
```

The server will start on `http://localhost:8000`

## Usage

Send a POST request to `/query` with a JSON body containing your query:

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your question here"}'
```

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## How it Works

1. The system takes your query and sends it to Ragie's retrieval API
2. Ragie returns relevant chunks of information
3. These chunks are combined with your query and sent to OpenAI's GPT-4
4. The final response is returned to you

## Note

Make sure you have:
1. A valid Ragie API key
2. A valid OpenAI API key
3. Documents already uploaded to your Ragie account 