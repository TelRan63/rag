import numpy as np
import ollama

MODEL = "embeddinggemma"


def get_embedding(text: str) -> np.ndarray:
    response = ollama.embed(
        model=MODEL,
        input=text
    )

    return np.array(
        response["embeddings"][0],
        dtype = np.float32
    )

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (
        np.linalg.norm(a)  * np.linalg.norm(b)
    )


text_a = "A dog is running through the park."
text_b = "A puppy is running outside."
text_c = "PostgreSQL supports database transactions."

embedding_a = get_embedding(text_a)
embedding_b = get_embedding(text_b)
embedding_c = get_embedding(text_c)

print(
    "A vs B:",
    cosine_similarity(embedding_a, embedding_b)
)

print(
    "A vs C:",
    cosine_similarity(embedding_a, embedding_c)
)