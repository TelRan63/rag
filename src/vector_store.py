from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import (
    DistanceStrategy,
    PGVector,
)

from config import (
    DB_URL,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
)


embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


def get_vector_store(
    recreate: bool = False,
) -> PGVector:

    return PGVector(
        embeddings=embeddings,
        connection=DB_URL,
        collection_name=COLLECTION_NAME,
        embedding_length=EMBEDDING_DIMENSION,
        distance_strategy=DistanceStrategy.COSINE,
        use_jsonb=True,
        pre_delete_collection=recreate,
    )