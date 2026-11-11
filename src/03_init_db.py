import psycopg
from pgvector.psycopg import register_vector

DB_URL = (
    "postgresql://"
    "rag_user:rag_password"
    "@localhost:5432/"
    "rag_course"
)

with psycopg.connect(
        DB_URL,
        autocommit=True
) as conn:
    conn.execute(
        "CREATE EXTENSION IF NOT EXISTS vector"
    )

    register_vector(conn)

    conn.execute(
        "DROP TABLE IF EXISTS documents"
    )

    conn.execute(
        """
        CREATE TABLE documents
        (
            id        BIGSERIAL PRIMARY KEY,
            title     TEXT NOT NULL,
            content   TEXT NOT NULL,
            embedding VECTOR(768) NOT NULL
        )
        """
    )

print("Database initialized.")