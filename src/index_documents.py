from knowledge import (
    load_documents,
    split_documents,
)

from vector_store import get_vector_store


documents = load_documents()

print(
    f"Loaded {len(documents)} documents."
)

chunks = split_documents(documents)

print(
    f"Created {len(chunks)} chunks."
)

vector_store = get_vector_store(
    recreate=True
)

ids = vector_store.add_documents(
    documents=chunks
)

print(
    f"Indexed {len(ids)} chunks."
)