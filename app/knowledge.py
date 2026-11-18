from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    KNOWLEDGE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    ALLOWED_EXTENSIONS,
)


def get_knowledge_files():
    return sorted(
        path
        for path in KNOWLEDGE_DIR.iterdir()
        if (
                path.is_file()
                and path.suffix.lower()
                in ALLOWED_EXTENSIONS
        )
    )


def load_documents() -> list[Document]:
    documents = []

    for path in get_knowledge_files():
        content = path.read_text(
            encoding="utf-8"
        )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": path.name,
                },
            )
        )

    return documents


def split_documents(
        documents: list[Document],
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    return splitter.split_documents(
        documents
    )
