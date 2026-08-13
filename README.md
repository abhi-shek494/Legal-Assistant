# ⚖️ Legal RAG Assistant — Indian Supreme Court Precedents

An intelligent, grounded AI legal research assistant designed to search and analyze Indian Supreme Court case laws (1950–2024). Built with **LangChain**, **ChromaDB**, **Sentence-Transformers (BGE)**, and **Google Gemini**, with a real-time interactive UI powered by **Chainlit**.

---

## ✨ Features

- **Hybrid Search**: Combines BM25 keyword matching (for exact legal citations/statutes) with dense vector search (for semantic understanding).
- **Cross-Encoder Reranking**: Re-scores search candidates using `BAAI/bge-reranker-base` so only the top, most relevant precedents are passed to the LLM.
- **Verifiable Citations**: Every answer includes clickable source links in the UI so you can inspect court names, decision dates, and exact case excerpts.
- **Fallback Reasoning**: Answers from verified database precedents when available, or falls back gracefully to general Indian law knowledge (IPC/BNS, IT Act, etc.).

---

## 📊 Dataset

The vector database was generated from the public Kaggle dataset:
👉 **[Legal Dataset - SC Judgments India (1950–2024)](https://www.kaggle.com/datasets/adarshsingh0903/legal-dataset-sc-judgments-india-19502024)**

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/yadavkrish18/Legal-Assistant.git
cd Legal-Assistant
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
# Create environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Add Your Gemini API Key
Create a `.env` file in the root folder:
```env
GOOGLE_API_KEY=your_actual_google_gemini_api_key
```

### 4. Download the Pre-Embedded Vector Database
Embedding 122k+ case chunks takes 1–2 hours. To skip this:
1. Go to the [**Releases**](https://github.com/yadavkrish18/Legal-Assistant/releases) tab of this repository.
2. Download `chroma_legal_db.zip`.
3. Extract `chroma_legal_db.zip` into the root directory of this project so that `./chroma_legal_db/` exists.

*(Optional: If you want to embed raw PDFs yourself, place your PDFs inside `./archive/supreme_court_judgments/` and run `python ingest_pdfs.py`)*

### 5. Launch the App
```bash
python -m chainlit run app.py -w
```
Open **http://localhost:8000** in your browser!

---

## 📁 Repository Structure & What to Upload

If you are pushing this project to GitHub, **only upload the code and configuration files**. Do not upload database binaries or secret keys directly to Git.

### ✅ Upload to GitHub:
- [`app.py`](app.py) — Chainlit frontend UI & event handlers.
- [`legal_rag_system.py`](legal_rag_system.py) — Core RAG logic, hybrid retriever, reranker, and LCEL chain.
- [`ingest_pdfs.py`](ingest_pdfs.py) — PDF text cleaner, metadata extractor, and Chroma indexer.
- [`requirements.txt`](requirements.txt) — Project dependencies.
- [`chainlit.md`](chainlit.md) — Chainlit welcome panel.
- [`.gitignore`](.gitignore) — Excludes secrets and binary database folders.
- [`README.md`](README.md) — Documentation.

### ❌ Do NOT Upload (Ignored by `.gitignore`):
- `.env` — **Contains your private API keys.**
- `chroma_legal_db/` — Vector database directory (Upload to GitHub Releases instead).
- `__pycache__/` — Python bytecode cache.
- `archive/` — Raw PDF storage.

---

## 📜 License
MIT License. Free for educational and research use.
