
import io
import pymupdf4llm
from io import BytesIO
import fitz

def test_extract_pdf_with_bytesio():
    # Create a minimal valid PDF in memory
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Hello World! This is a test PDF.")
    pdf_bytes = doc.tobytes()
    doc.close()

    print("Created sample PDF in memory.")

    try:
        # Simulate the failing code in file_parser.py
        # text = pymupdf4llm.to_markdown(BytesIO(content))
        print("Attempting to extract text using pymupdf4llm with BytesIO...")
        text = pymupdf4llm.to_markdown(BytesIO(pdf_bytes))
        print("Success! Extracted text:")
        print(text)
    except Exception as e:
        print(f"Caught expected error: {e}")

if __name__ == "__main__":
    test_extract_pdf_with_bytesio()
