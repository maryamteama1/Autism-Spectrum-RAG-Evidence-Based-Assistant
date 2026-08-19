import pymupdf as fitz

def extract_pdf_pages(file_path: str) -> list[dict]:
    """Extracts text page-by-page while strictly preserving page numbers.

    Rule: Page numbers must never be lost for accurate citations.
    """
    doc = fitz.open(file_path)
    pages_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        if text.strip():
            pages_data.append(
                {"page_number": page_num + 1, "content": text.strip()}
            )

    doc.close()
    return pages_data


if __name__ == "__main__":
    # Quick sanity check on local document
    sample_path = "data/raw/document.pdf"
    results = extract_pdf_pages(sample_path)
    print(f"Extracted {len(results)} pages successfully!")
    if results:
        print(f"\n--- Page {results[0]['page_number']} Preview ---")
        print(results[0]["content"][:200])