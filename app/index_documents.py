from .rag_service import (
    rebuild_index,
)


def main() -> None:

    documents, chunks = (
        rebuild_index()
    )

    print(
        f"Indexed {documents} documents "
        f"and {chunks} chunks."
    )


if __name__ == "__main__":
    main()