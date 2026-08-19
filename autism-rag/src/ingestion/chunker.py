import uuid
import pymupdf as fitz
from src.schemas.chunk import TextChunk


def chunk_text_by_boundary(
    text: str, chunk_size: int = 500, overlap: int = 50
) -> list[str]:
    """Splits text into chunks prioritizing paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += ("\n\n" if current_chunk else "") + para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Handle paragraphs larger than chunk_size
            if len(para) > chunk_size:
                words = para.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) <= chunk_size:
                        temp_chunk += (" " if temp_chunk else "") + word
                    else:
                        chunks.append(temp_chunk)
                        temp_chunk = word
                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def process_document_to_chunks(
    file_path: str, source_id: str, document_title: str, document_type: str
) -> list[TextChunk]:
    """Extracts pages and generates structured TextChunk objects with preserved metadata."""
    doc = fitz.open(file_path)
    all_chunks = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        if not text:
            continue

        raw_chunks = chunk_text_by_boundary(text)

        for text_segment in raw_chunks:
            chunk_obj = TextChunk(
                chunk_id=str(uuid.uuid4()),
                source_id=source_id,
                document_title=document_title,
                page_number=page_num + 1,
                text=text_segment,
                document_type=document_type,
            )
            all_chunks.append(chunk_obj)

    doc.close()
    return all_chunks


if __name__ == "__main__":
    chunks = process_document_to_chunks(
        file_path="data/raw/document.pdf",
        source_id="nice_cg128",
        document_title="NICE CG128",
        document_type="clinical_guideline",
    )
    print(f"Total Chunks Created: {len(chunks)}")
    print(f"\n--- Chunk 1 Preview (Page {chunks[0].page_number}) ---")
    print(f"ID: {chunks[0].chunk_id}")
    print(f"Text:\n{chunks[0].text[:150]}...")