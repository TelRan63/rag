from langchain_core.documents import Document
from langchain_core.output_parsers import (
    StrOutputParser,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_ollama import ChatOllama

from .config import (
    CHAT_MODEL,
    KNOWLEDGE_DIR,
    TOP_K,
)

from .knowledge import (
    get_knowledge_files,
    load_documents,
    split_documents,
)

from .vector_store import (
    get_vector_store,
)

llm = ChatOllama(
    model=CHAT_MODEL,
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a company knowledge assistant.

Answer the user's question using only the provided context.

If the context does not contain enough information to answer,
say exactly:

I don't know based on the provided documents.

Do not invent company policies.

Keep the answer concise and factual.
            """.strip(),
        ),
        (
            "human",
            """
Context:

{context}

Question:

{question}
            """.strip(),
        ),
    ]
)

answer_chain = (
        prompt
        | llm
        | StrOutputParser()
)


def format_documents(
        documents: list[Document],
) -> str:
    parts = []

    for index, document in enumerate(
            documents,
            start=1,
    ):
        source = document.metadata.get(
            "source",
            "unknown",
        )

        start_index = (
            document.metadata.get(
                "start_index"
            )
        )

        parts.append(
            f"""
[Source {index}: {source},
 start_index={start_index}]

{document.page_content}
            """.strip()
        )

    return "\n\n---\n\n".join(parts)


def rebuild_index() -> tuple[int, int]:
    documents = load_documents()

    chunks = split_documents(
        documents
    )

    vector_store = get_vector_store(
        recreate=True
    )

    if chunks:
        vector_store.add_documents(
            documents=chunks
        )

    return (
        len(documents),
        len(chunks),
    )


def ask_question(
        question: str,
) -> tuple[str, list[dict]]:
    vector_store = get_vector_store()

    retriever = (
        vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": TOP_K,
            },
        )
    )

    documents = retriever.invoke(
        question
    )

    context = format_documents(
        documents
    )

    answer = answer_chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    sources = []

    seen = set()

    for document in documents:

        source = document.metadata.get(
            "source",
            "unknown",
        )

        start_index = (
            document.metadata.get(
                "start_index"
            )
        )

        key = (
            source,
            start_index,
        )

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            {
                "source": source,
                "start_index": start_index,
            }
        )

    return answer, sources