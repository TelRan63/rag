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

documents = [
    {
        "title": "Python asyncio",
        "content": (
            "Python's asyncio library coordinates coroutines "
            "using an event loop. Tasks can yield control while "
            "waiting for I/O, allowing other tasks to run."
        )
    },

    {
        "title": "JavaScript Event Loop",
        "content": (
            "JavaScript uses an event loop to coordinate "
            "asynchronous operations and execute queued callbacks."
        )
    },

    {
        "title": "FastAPI",
        "content": (
            "FastAPI is a Python framework for building web APIs. "
            "It supports asynchronous request handlers using "
            "async and await."
        )
    },

    {
        "title": "PostgreSQL Transactions",
        "content": (
            "PostgreSQL transactions group multiple database "
            "operations into a single atomic unit of work."
        )
    },

    {
        "title": "PostgreSQL Indexes",
        "content": (
            "Database indexes can improve query performance by "
            "allowing PostgreSQL to find rows without scanning "
            "the entire table."
        )
    },

    {
        "title": "MongoDB",
        "content": (
            "MongoDB stores information as flexible "
            "document-like records rather than rows in "
            "traditional relational tables."
        )
    },

    {
        "title": "React",
        "content": (
            "React applications are built from components. "
            "Components can manage state and render user "
            "interfaces when data changes."
        )
    },

    {
        "title": "Redux Toolkit",
        "content": (
            "Redux Toolkit provides utilities for managing "
            "shared application state in JavaScript applications."
        )
    },

    {
        "title": "Docker",
        "content": (
            "Docker packages applications and their dependencies "
            "into containers that can run consistently across "
            "different environments."
        )
    },

    {
        "title": "REST API",
        "content": (
            "REST APIs expose resources through HTTP endpoints "
            "and commonly use methods such as GET, POST, PUT, "
            "PATCH, and DELETE."
        )
    }
]

def index_documents():
    texts = [
        document["content"]
        for document in documents
    ]

    response = ollama.embed(
        model=MODEL,
        input=texts
    )

    embeddings = response["embeddings"]

    with psycopg.connect(
            DB_URL,
            autocommit=True
    ) as conn:
        register_vector(conn)

        for document, embedding in zip(
                documents,
                embeddings
        ):
            conn.execute(
                """
                INSERT INTO documents (title,
                                       content,
                                       embedding)
                VALUES (%s, %s, %s)
                """,
                (
                    document["title"],
                    document["content"],
                    Vector(embedding)
                )
            )

    print(
        f"Indexed {len(documents)} documents."
    )

if __name__ == "__main__":
    index_documents()