import os
from dotenv import load_dotenv

load_dotenv()

# ─── LLM ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
STACK_MODE   = os.getenv("STACK_MODE", "local")  # "local" = Ollama, "hybrid" = Groq

# ─── Models ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ─── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 200

# ─── Retrieval ───────────────────────────────────────────────────────────────
RAW_K   = 6
FINAL_K = 4

# ─── Generation ──────────────────────────────────────────────────────────────
TEMPERATURE = 0.1

# ─── App ─────────────────────────────────────────────────────────────────────
MAX_QUERY_LEN = 500