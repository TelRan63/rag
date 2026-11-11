import ollama
import psycopg

from pgvector import Vector
from pgvector.psycopg import register_vector

MODEL = "embeddinggemma"

DB_URL = (
    "postgresql://"
    "rag_user:rag_password"
    "@localhost:5432/"
    "rag_course"
)


def create_embedding(text: str) -> Vector:
    response = ollama.embed(
        model=MODEL,
        input=text
    )

    return Vector(
        response["embeddings"][0]
    )


def semantic_search(
        query: str,
        limit: int = 3
):
    query_embedding = create_embedding(query)

    with psycopg.connect(DB_URL) as conn:
        register_vector(conn)

        result = conn.execute(
            """
            SELECT title,
                   content,
                   embedding <=> %s
                AS cosine_distance
            FROM documents
            ORDER BY embedding <=> %s
                LIMIT %s
            """,
            (
                query_embedding,
                query_embedding,
                limit
            )
        )

        return result.fetchall()

if __name__ == "__main__":
    query = (
        "How can a Python program do other work "
        "while waiting for network operations?"
    )

    results = semantic_search(query)

    print(f"\nQuery: {query}\n")

    for index, row in enumerate(
            results,
            start=1
    ):
        title, content, distance = row

        similarity = 1 - distance

        print(f"{index}. {title}")
        print(f"   Similarity: {similarity:.4f}")
        print(f"   {content}")
        print()