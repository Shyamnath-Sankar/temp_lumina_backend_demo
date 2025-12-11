import os
from typing import Optional
from io import BytesIO
import pymupdf4llm
import fitz
from docx import Document
from bs4 import BeautifulSoup
import markdown
from utils.logger import logger

class FileParser:
    @staticmethod
    def extract_text(content: bytes, ext: str) -> Optional[str]:
        """Extract text from various file formats"""
        try:
            ext = ext.lower()
            if not ext.startswith('.'):
                ext = '.' + ext

            if ext == '.pdf':
                return FileParser._extract_pdf(content)
            elif ext in ['.docx', '.doc']:
                return FileParser._extract_docx(content)
            elif ext == '.txt':
                return FileParser._extract_txt(content)
            elif ext == '.html':
                return FileParser._extract_html(content)
            elif ext == '.md':
                return FileParser._extract_markdown(content)
            else:
                logger.warning(f"Unsupported file type: {ext}")
                return None

        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return None
    
    @staticmethod
    def _extract_pdf(content: bytes) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            with fitz.open(stream=content, filetype="pdf") as doc:
                text = ""
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text += page.get_text("text") + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {str(e)}")
            raise e

    @staticmethod
    def _extract_docx(content: bytes) -> str:
        """Extract text from DOCX"""
        doc = Document(BytesIO(content))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()

    @staticmethod
    def _extract_txt(content: bytes) -> str:
        """Extract text from TXT"""
        return content.decode('utf-8', errors='ignore').strip()

    @staticmethod
    def _extract_html(content: bytes) -> str:
        """Extract text from HTML"""
        soup = BeautifulSoup(content.decode('utf-8', errors='ignore'), 'html.parser')
        return soup.get_text().strip()

    @staticmethod
    def _extract_markdown(content: bytes) -> str:
        """Extract text from Markdown"""
        md_text = content.decode('utf-8', errors='ignore')
        html = markdown.markdown(md_text)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text().strip()
