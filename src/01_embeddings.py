import ollama
import numpy as np

MODEL = "embeddinggemma"

text = "Python supports asynchronous programming with asyncio."

response = ollama.embed(
    model=MODEL,
    input=text
)

embedding = response["embeddings"][0]

print(type(embedding))
print("Dimensions:", len(embedding))
print("First 10 values:")
print(embedding[:10])

vector = np.array(embedding)
print(np.linalg.norm(vector))

# Batch embedding

texts = [
    "Python supports asynchronous programming.",
    "PostgreSQL is a relational database.",
    "React is used for building user interfaces."
]

response = ollama.embed(
    model=MODEL,
    input=texts
)

embeddings = response["embeddings"]

print("Number of vectors:", len(embeddings))

for embedding in embeddings:
    print(len(embedding))