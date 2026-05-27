import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "")
SEARCH_API_URL = os.getenv("SEARCH_API_URL", "https://serpapi.com/search")
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "10"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8000"))
DB_PATH = os.getenv("DB_PATH", "research.db")
