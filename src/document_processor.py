import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

def process_document(file_path):
    """Process PDF or TXT document and extract text"""
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.pdf':
        return _read_pdf(file_path)
    elif file_path.suffix.lower() == '.txt':
        return _read_text(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or TXT.")

def _read_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise Exception(f"Error reading PDF: {e}")
    return text

def _read_text(file_path):
    """Read text from TXT file"""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        raise Exception(f"Error reading text file: {e}")

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)