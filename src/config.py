from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

DB_URL = (
    "postgresql+psycopg://"
    "rag_user:rag_password"
    "@localhost:5432/"
    "rag_course"
)

EMBEDDING_MODEL = "embeddinggemma"
EMBEDDING_DIMENSION = 768

CHAT_MODEL = "gemma3:4b"

COLLECTION_NAME = "company_knowledge"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

TOP_K = 3