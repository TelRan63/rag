from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    KNOWLEDGE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_documents() -> list[Document]:
    documents = []

    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")

        document = Document(
            page_content=content,
            metadata={
                "source": path.name,
            },
        )

        documents.append(document)

    return documents

def split_documents(
    documents: list[Document],
) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    return splitter.split_documents(documents)

if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    print("Documents:", len(documents))
    print("Chunks:", len(chunks))

    for index, chunk in enumerate(chunks, start=1):
        print()
        print("=" * 70)
        print(f"CHUNK {index}")
        print(chunk.metadata)
        print()
        print(chunk.page_content)