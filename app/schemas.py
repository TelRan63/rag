from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
    )


class SourceInfo(BaseModel):
    source: str
    start_index: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


class DocumentInfo(BaseModel):
    name: str
    size: int


class IndexResponse(BaseModel):
    filename: str
    documents: int
    chunks: int