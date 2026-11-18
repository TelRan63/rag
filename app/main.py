from pathlib import Path

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from .config import (
    ALLOWED_EXTENSIONS,
    CORS_ORIGINS,
    KNOWLEDGE_DIR,
    MAX_UPLOAD_BYTES,
)

from .knowledge import (
    get_knowledge_files,
)

from .rag_service import (
    ask_question,
    rebuild_index,
)

from .schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    IndexResponse,
)

app = FastAPI(
    title=(
        "BluePeak Knowledge Assistant API"
    )
)

app.add_middleware(
    CORSMiddleware,

    allow_origins=CORS_ORIGINS,

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
    }


@app.get(
    "/api/documents",
    response_model=list[DocumentInfo],
)
def list_documents():
    return [
        DocumentInfo(
            name=path.name,
            size=path.stat().st_size,
        )
        for path
        in get_knowledge_files()
    ]

@app.post(
    "/api/documents",
    response_model=IndexResponse,
)
def upload_document(
    file: UploadFile,
):
    original_name = (
            file.filename or ""
    )

    filename = Path(
        original_name
    ).name

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Missing filename.",
        )
    suffix = (
        Path(filename)
        .suffix
        .lower()
    )

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only .md and .txt "
                "files are supported."
            ),
        )

    raw = file.file.read(
        MAX_UPLOAD_BYTES + 1
    )

    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File is too large.",
        )

    try:
        content = raw.decode(
            "utf-8"
        )
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail=(
                "File must contain "
                "UTF-8 text."
            ),
        )
    path = (
            KNOWLEDGE_DIR
            / filename
    )

    path.write_text(
        content,
        encoding="utf-8",
    )

    documents, chunks = (
        rebuild_index()
    )

    return IndexResponse(
        filename=filename,
        documents=documents,
        chunks=chunks,
    )

@app.delete(
    "/api/documents/{filename}"
)
def delete_document(
    filename: str,
):

    safe_name = Path(filename).name

    path = (
        KNOWLEDGE_DIR
        / safe_name
    )

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    if (
        path.suffix.lower()
        not in ALLOWED_EXTENSIONS
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid document.",
        )

    path.unlink()

    documents, chunks = (
        rebuild_index()
    )

    return {
        "deleted": safe_name,
        "documents": documents,
        "chunks": chunks,
    }

@app.post(
    "/api/ask",
    response_model=AskResponse,
)
def ask(
    request: AskRequest,
):

    answer, sources = (
        ask_question(
            request.question
        )
    )

    return AskResponse(
        answer=answer,
        sources=sources,
    )