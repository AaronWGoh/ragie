import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys from environment
RAGIE_API_KEY = os.getenv("RAGIE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not RAGIE_API_KEY:
    raise ValueError("RAGIE_API_KEY environment variable is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
