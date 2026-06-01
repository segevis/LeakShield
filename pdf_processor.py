# ==========================================================
# PDF Processing Module
# ----------------------------------------------------------
# This module is responsible for extracting text from PDF files.
#
# It performs:
# - Opening the PDF file
# - Iterating over all pages
# - Extracting textual content from each page
#
# Output:
# A list of tuples → (page_number, page_text)
#
# This is the entry point of the data pipeline.
# ==========================================================

from pypdf import PdfReader

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            print("\n" + "=" * 70)
            print(f"RAW TEXT FROM PAGE {i + 1}:")
            print(repr(text))
            print("=" * 70)
            pages.append((i + 1, text))

    return pages