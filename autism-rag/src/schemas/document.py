from pydantic import BaseModel


class DocumentSource(BaseModel):
    source_id: str
    title: str
    document_type: str
    file_path: str


class ExtractedPage(BaseModel):
    source_id: str
    page_number: int
    content: str