# ==========================================================
# PDF Processor Module
# ----------------------------------------------------------
# Extracts text from PDF files page by page.
# Debug printing is disabled by default to avoid exposing
# sensitive document content in the terminal.
# ==========================================================

from pypdf import PdfReader


def extract_text(pdf_path, debug=False):
    """
    Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        debug: If True, prints raw extracted text per page.

    Returns:
        A list of tuples: (page_number, extracted_text)
    """
    reader = PdfReader(pdf_path)
    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if debug:
            print()
            print("=" * 70)
            print(f"RAW TEXT FROM PAGE {page_number}:")
            print(repr(text))
            print("=" * 70)

        pages_text.append((page_number, text))

    return pages_text