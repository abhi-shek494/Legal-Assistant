"""
Legal Case Precedent RAG System
================================
RAG pipeline for legal case law analysis using:
  - ChromaDB + BAAI/bge-base-en-v1.5 embeddings
  - BM25 + Vector hybrid search with Cross-Encoder reranking
  - Google Gemini for grounded generation
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import sys
import textwrap
from pathlib import Path

# PyTorch meta-tensor workaround (sentence-transformers + PyTorch >= 2.4 compat)
import torch
_original_module_to = torch.nn.Module.to

def _safe_module_to(self, *args, **kwargs):
    try:
        return _original_module_to(self, *args, **kwargs)
    except NotImplementedError:
        device = args[0] if args else kwargs.get("device", "cpu")
        empty_self = self.to_empty(device="cpu")
        if str(device) != "cpu":
            return _original_module_to(empty_self, *args, **kwargs)
        return empty_self

torch.nn.Module.to = _safe_module_to

import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================================
# CONFIGURATION
# ============================================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
LLM_MODEL_NAME = "gemini-3.6-flash"
LLM_TEMPERATURE = 0.0

CHROMA_PERSIST_DIR = "./chroma_legal_db"
CHROMA_COLLECTION_NAME = "legal_precedents"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

RETRIEVAL_TOP_K = 15
RERANK_TOP_N = 4

BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6


# ============================================================================
# EMBEDDINGS & VECTOR STORE
# ============================================================================

def create_embedding_model() -> HuggingFaceEmbeddings:
    """Initialize the sentence-transformer embedding model."""
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"[OK] Loaded embedding model: {EMBEDDING_MODEL_NAME}")
    return embedding_model


def load_vector_store(embedding_model: HuggingFaceEmbeddings) -> Chroma:
    """Load the persistent ChromaDB vector store."""
    persist_path = Path(CHROMA_PERSIST_DIR)

    if not persist_path.exists():
        print(f"[!] ChromaDB not found at '{CHROMA_PERSIST_DIR}'. Run ingest_pdfs.py first.")
        raise FileNotFoundError("ChromaDB directory not found.")
        
    print(f"[i] Found existing ChromaDB at '{CHROMA_PERSIST_DIR}' — loading...")
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    collection_count = vector_store._collection.count()
    print(f"[OK] Loaded existing collection with {collection_count} vectors")
    return vector_store


# ============================================================================
# HYBRID RETRIEVAL PIPELINE
# ============================================================================

def get_all_docs_from_chroma(vector_store: Chroma) -> list[Document]:
    """Retrieve all documents from ChromaDB for BM25 initialization."""
    count = vector_store._collection.count()
    docs = []
    batch_size = 10000
    for offset in range(0, count, batch_size):
        batch = vector_store._collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"]
        )
        documents = batch.get("documents", [])
        metadatas = batch.get("metadatas", [])
        for doc, meta in zip(documents, metadatas):
            docs.append(Document(page_content=doc, metadata=meta))
    return docs


def create_retrieval_pipeline(vector_store: Chroma) -> ContextualCompressionRetriever:
    """
    Build a three-stage hybrid retrieval pipeline:
    BM25 (keyword) + Vector (semantic) → Ensemble → Cross-Encoder Rerank
    """
    # Sync BM25 corpus with what's actually in ChromaDB
    print("[*] Synchronizing BM25 corpus with ChromaDB...")
    sync_chunks = get_all_docs_from_chroma(vector_store)
    
    # Stage 1a: BM25 keyword retriever
    bm25_retriever = BM25Retriever.from_documents(
        documents=sync_chunks,
        k=RETRIEVAL_TOP_K,
    )

    # Stage 1b: Vector semantic retriever
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_TOP_K},
    )

    # Stage 2: Ensemble (Reciprocal Rank Fusion)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[BM25_WEIGHT, VECTOR_WEIGHT],
    )

    # Stage 3: Cross-Encoder reranker
    cross_encoder = HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )

    compressor = CrossEncoderReranker(
        model=cross_encoder,
        top_n=RERANK_TOP_N,
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )

    print(
        f"[OK] Hybrid retrieval pipeline ready: "
        f"BM25({BM25_WEIGHT}) + Vector({VECTOR_WEIGHT}) -> "
        f"Ensemble -> top-{RERANK_TOP_N} Cross-Encoder rerank"
    )
    return compression_retriever


# ============================================================================
# SELF-QUERY METADATA FILTERING (alternative retriever)
# ============================================================================

LEGAL_METADATA_FIELDS = [
    AttributeInfo(
        name="case_name",
        description="The name of the legal case (e.g., 'Plaintiff v. Defendant')",
        type="string",
    ),
    AttributeInfo(
        name="date",
        description="The date the court decision was issued, in YYYY-MM-DD format",
        type="string",
    ),
    AttributeInfo(
        name="court",
        description="The name of the court that issued the decision",
        type="string",
    ),
    AttributeInfo(
        name="summary",
        description="A brief summary of the case and its legal issues",
        type="string",
    ),
]

DOCUMENT_CONTENT_DESCRIPTION = (
    "Full text of legal court opinions, including holdings, reasoning, "
    "citations to prior case law, and disposition of the case."
)


def create_self_query_retriever(vector_store: Chroma) -> SelfQueryRetriever:
    """
    Create a Self-Query Retriever for automatic metadata filtering.
    Useful when users specify court names, date ranges, or case names.
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=0.0,
    )

    self_query_retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vector_store,
        document_contents=DOCUMENT_CONTENT_DESCRIPTION,
        metadata_field_info=LEGAL_METADATA_FIELDS,
        verbose=True,
    )

    print("[OK] Self-Query retriever ready: auto-filters by case_name, date, court, summary")
    return self_query_retriever


# ============================================================================
# RAG CHAIN (LLM Generation)
# ============================================================================

LEGAL_SYSTEM_PROMPT = """\
You are a senior legal research assistant specializing in Indian law analysis.

════════════════════════════════════════════════════════════════════════
ANALYSIS RULES:
════════════════════════════════════════════════════════════════════════

1. USE PROVIDED CONTEXT WHEN RELEVANT: First, evaluate the retrieved case excerpts 
   provided below. If they are relevant to the user's query, use them to form 
   your legal analysis.
   
2. CITE EXPLICITLY: For every retrieved precedent you reference, you MUST state:
   - The exact case_name as it appears in the metadata
   - The court that decided the case
   - The date of the decision
   Format: "In **[Case Name]** ([Court], [Date]), the court held that..."

3. GENERAL KNOWLEDGE FALLBACK: If the provided context does NOT contain relevant 
   precedents (for example, if the user asks about general Indian laws like 
   cyberbullying, ragging in colleges, consumer rights, or criminal complaints 
   that aren't in the database), you MUST ignore the irrelevant context and 
   provide a comprehensive, accurate answer based on your vast pre-trained 
   knowledge of Indian Law (e.g., IPC, BNS, UGC Regulations, IT Act, etc.).

4. DISTINGUISH WHEN NECESSARY: If a case in the context is only partially
   relevant or could be distinguished on its facts, say so explicitly.

════════════════════════════════════════════════════════════════════════
CONTEXT — Retrieved Case Excerpts:
════════════════════════════════════════════════════════════════════════

{context}

════════════════════════════════════════════════════════════════════════
"""

HUMAN_PROMPT = "Legal research query: {question}"


def format_docs_with_metadata(docs: list[Document]) -> str:
    """Format retrieved documents with metadata headers for the LLM context."""
    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = (
            f"[PRECEDENT {i}]\n"
            f"  Case:  {meta.get('case_name', 'Unknown')}\n"
            f"  Court: {meta.get('court', 'Unknown')}\n"
            f"  Date:  {meta.get('date', 'Unknown')}\n"
            f"  ─────────────────────────────────────────"
        )
        formatted_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n{'='*72}\n\n".join(formatted_parts)


def build_rag_chain(retriever: ContextualCompressionRetriever) -> object:
    """Build the complete RAG chain: retriever → prompt → LLM → output."""
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", LEGAL_SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs_with_metadata),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"[OK] RAG chain built: retriever -> prompt -> {LLM_MODEL_NAME} -> output")
    return rag_chain


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

def run_pipeline(query: str):
    """Execute the full Legal RAG pipeline end-to-end (CLI mode)."""
    print("=" * 72)
    print("  LEGAL CASE PRECEDENT RAG SYSTEM")
    print("=" * 72)

    embedding_model = create_embedding_model()
    vector_store = load_vector_store(embedding_model)
    retriever = create_retrieval_pipeline(vector_store)
    rag_chain = build_rag_chain(retriever)

    print(f'\n[?] Query: "{query}"')
    response = rag_chain.invoke(query)

    print("\n--- RESPONSE ---\n")
    print(response)
    print("\n" + "=" * 72)
    return response


if __name__ == "__main__":
    load_dotenv()

    if not os.environ.get("GOOGLE_API_KEY"):
        print(
            "[!] WARNING: GOOGLE_API_KEY not found in environment.\n"
            "    Create a .env file with: GOOGLE_API_KEY=...\n"
        )

    demo_query = "What is the precedent for breach of contract via email delay?"

    try:
        run_pipeline(query=demo_query)
    except Exception as e:
        print(f"\n[ERR] Pipeline error: {e}")
        sys.exit(1)
