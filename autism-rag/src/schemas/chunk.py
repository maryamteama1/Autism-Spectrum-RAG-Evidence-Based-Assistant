from typing import Optional
from pydantic import BaseModel


class TextChunk(BaseModel):
    chunk_id: str
    source_id: str
    document_title: str
    page_number: int
    text: str
    topic: Optional[str] = None
    age_group: Optional[str] = None
    document_type: str