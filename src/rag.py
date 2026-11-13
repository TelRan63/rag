from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import (
    CHAT_MODEL,
    TOP_K,
)

from vector_store import get_vector_store


vector_store = get_vector_store()

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": TOP_K,
    },
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
    documents,
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

        start_index = document.metadata.get(
            "start_index",
            "unknown",
        )

        parts.append(
            f"""
[Source {index}: {source}, start_index={start_index}]

{document.page_content}
            """.strip()
        )

    return "\n\n---\n\n".join(parts)

def ask(
    question: str,
) -> tuple[str, list]:

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

    return answer, documents

def print_sources(
    documents,
) -> None:

    print("\nSources:")

    for document in documents:
        source = document.metadata.get(
            "source",
            "unknown",
        )

        start_index = document.metadata.get(
            "start_index",
            "unknown",
        )

        print(
            f"- {source} "
            f"(start_index={start_index})"
        )


if __name__ == "__main__":

    print(
        "BluePeak Systems Knowledge Assistant"
    )

    print(
        "Type 'exit' to stop."
    )

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        answer, documents = ask(
            question
        )

        print("\nAnswer:")
        print(answer)

        print_sources(documents)
