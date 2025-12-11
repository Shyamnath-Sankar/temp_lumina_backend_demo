
import sys
import os
import fitz

# Add the parent directory to sys.path to import utils.file_parser
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.file_parser import FileParser

def test_file_parser_fix():
    print("Testing FileParser fix...")
    
    # Create a minimal valid PDF in memory
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Hello World! This is a test PDF.")
    pdf_bytes = doc.tobytes()
    doc.close()

    try:
        print("Calling FileParser.extract_text...")
        text = FileParser.extract_text(pdf_bytes, '.pdf')
        
        if text and "Hello World" in text:
            print("SUCCESS: Text extracted successfully.")
            print("-" * 20)
            print(text)
            print("-" * 20)
        else:
            print("FAILURE: Text not extracted or incorrect.")
            print(f"Result: {text}")
            
    except Exception as e:
        print(f"FAILURE: An error occurred: {e}")



if __name__ == "__main__":
    test_file_parser_fix()


