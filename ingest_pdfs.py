"""
PDF Ingestion Script
Parses Supreme Court PDFs from Indian Kanoon, chunks text, and indexes into ChromaDB.
"""

import os
import glob
import re
import hashlib
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- Configuration ---
PDF_DIR = "./archive/supreme_court_judgments"
CHROMA_PERSIST_DIR = "./chroma_legal_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 300 


def clean_kanoon_text(text: str) -> str:
    """Removes repetitive Indian Kanoon footers and URLs."""
    text = re.sub(r'Indian Kanoon - http://indiankanoon\.org/doc/\d+/?', '', text, flags=re.IGNORECASE)
    return text.strip()


def extract_metadata_from_kanoon(text: str, filename: str) -> dict:
    """
    Extracts case name, date, and court from Indian Kanoon PDF filename.
    Format: 'Case_Name_vs_Other_Party_on_Date_1.PDF'
    """
    metadata = {
        "case_name": filename.replace(".pdf", "").replace(".PDF", ""),
        "date": 1900,  
        "court": "Unknown Court",
        "summary": "Full text case document extracted from PDF."
    }
    
    # Parse Kanoon filename format
    filename_match = re.search(r'(.+?)_on_(\d{1,2}_[A-Za-z]+_\d{4})', filename, re.IGNORECASE)
    if filename_match:
        metadata["case_name"] = filename_match.group(1).replace("_", " ").strip()
        year_match = re.search(r'(\d{4})', filename_match.group(2))
        if year_match:
            metadata["date"] = int(year_match.group(1))
    else:
        year_match = re.search(r'\b(19\d{2}|20[0-2]\d)\b', text[:1000])
        if year_match:
            metadata["date"] = int(year_match.group(1))
            
    # Find court name from header
    court_match = re.search(r'(IN THE SUPREME COURT OF INDIA|HIGH COURT OF [A-Z\s]+)', text[:1500], re.IGNORECASE)
    if court_match:
        metadata["court"] = " ".join(court_match.group(1).split()).strip().title()
        
    return metadata


def ingest_pdfs():
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        print(f"Created '{PDF_DIR}'. Place your year folders (1950, 1951, etc.) here.")
        return

    pdf_files_raw = glob.glob(os.path.join(PDF_DIR, "**", "*.pdf"), recursive=True)
    pdf_files_raw.extend(glob.glob(os.path.join(PDF_DIR, "**", "*.PDF"), recursive=True))
    
    # Deduplicate (Windows is case-insensitive)
    pdf_files = list(set([os.path.abspath(p) for p in pdf_files_raw]))
    
    if not pdf_files:
        print(f"No PDFs found recursively in {PDF_DIR}.")
        return

    print(f"Found {len(pdf_files)} PDFs across subdirectories. Initializing models...")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, encode_kwargs={'normalize_embeddings': True})
    vector_store = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings, collection_name="legal_precedents")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", ". ", " "])

    batch_chunks = []
    batch_ids = []
    total_added = 0

    for pdf_path in tqdm(pdf_files, desc="Ingesting Kanoon PDFs"):
        try:
            loader = PyMuPDFLoader(pdf_path)
            pages = loader.load()
            
            if not pages:
                continue
                
            full_text = " ".join([page.page_content for page in pages])
            
            if len(full_text.strip()) < 50:
                print(f"\n[SKIP] Scanned/Empty PDF: {pdf_path}")
                continue
                
            cleaned_text = clean_kanoon_text(full_text)
            filename = os.path.basename(pdf_path)
            metadata = extract_metadata_from_kanoon(cleaned_text, filename)
            
            master_doc = Document(page_content=cleaned_text, metadata=metadata)
            chunks = splitter.split_documents([master_doc])
            
            # Deterministic IDs prevent duplicates on re-runs
            file_hash = hashlib.md5(filename.encode()).hexdigest()
            
            for i, chunk in enumerate(chunks):
                batch_chunks.append(chunk)
                batch_ids.append(f"{file_hash}_chunk_{i}")
            
            if len(batch_chunks) >= BATCH_SIZE:
                vector_store.add_documents(documents=batch_chunks, ids=batch_ids)
                total_added += len(batch_chunks)
                batch_chunks, batch_ids = [], []
                
        except Exception as e:
            print(f"\nError processing {pdf_path}: {e}")
            
    if batch_chunks:
        vector_store.add_documents(documents=batch_chunks, ids=batch_ids)
        total_added += len(batch_chunks)

    print(f"\n✅ Ingestion Complete! Added {total_added} vectorized chunks to ChromaDB.")


if __name__ == "__main__":
    load_dotenv()
    ingest_pdfs()
