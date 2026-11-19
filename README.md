# Legal RAG Assistant

An interactive AI-powered legal research assistant for exploring Indian Supreme Court judgments. The project combines keyword search, semantic retrieval, cross-encoder reranking, and grounded generation to provide answers with traceable source excerpts.

**Tech stack:** Python, LangChain, ChromaDB, BGE embeddings, BM25, Hugging Face cross-encoders, Google Gemini, and Chainlit.

---

## Key Features

- **Hybrid retrieval:** Combines BM25 keyword matching with dense vector similarity.
- **Cross-encoder reranking:** Uses `BAAI/bge-reranker-base` to prioritize relevant precedents.
- **Grounded responses:** Generates answers from retrieved case content using Google Gemini.
- **Source transparency:** Displays case metadata and clickable source excerpts beside each response.
- **PDF ingestion:** Cleans and chunks Indian Kanoon PDFs, then creates a persistent ChromaDB collection.
- **Interactive UI:** Provides real-time retrieval and generation steps through Chainlit.

---

## Architecture

```text
PDF judgments
  -> text cleaning and chunking
  -> BGE embeddings
  -> ChromaDB vector store

User query
  -> BM25 + vector retrieval
  -> ensemble ranking
  -> cross-encoder reranking
  -> Gemini grounded response
  -> cited UI sources
```

---

## Dataset

The vector database can be generated from the public [Legal Dataset - SC Judgments India (1950-2024)](https://www.kaggle.com/datasets/adarshsingh0903/legal-dataset-sc-judgments-india-19502024).

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/abhi-shek494/Legal-Assistant.git
cd Legal-Assistant
```

### 2. Create an environment and install dependencies

```bash
python -m venv venv
```

Activate the environment:

```powershell
# Windows PowerShell
venv\Scripts\Activate.ps1
```

```bash
# Linux or macOS
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure Google Gemini

Create a `.env` file in the project directory:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

Keep API keys out of source control.

### 4. Prepare the vector database

For prebuilt data, download `chroma_legal_db.zip` from the repository [Releases](https://github.com/abhi-shek494/Legal-Assistant/releases), extract it, and confirm that `./chroma_legal_db/` exists.

To build it from PDFs:

1. Place judgment PDFs under `./archive/supreme_court_judgments/`.
2. Run `python ingest_pdfs.py`.

### 5. Run the application

```bash
python -m chainlit run app.py -w
```

Open `http://localhost:8000`.

---

## Project Structure

```text
Legal-Assistant/
├── app.py                # Chainlit interface and chat event handlers
├── legal_rag_system.py   # Retrieval, reranking, and RAG chain
├── ingest_pdfs.py        # PDF parsing and vector database ingestion
├── requirements.txt      # Python dependencies
├── chainlit.md           # Chainlit welcome screen
├── .chainlit/config.toml # Chainlit configuration
└── README.md
```

---

## Contributing

This project was developed collaboratively. Contributions are welcome through issues and pull requests.

---

## Disclaimer

This application is an educational legal research aid and not a substitute for advice from a qualified legal professional. AI-generated answers should always be verified against the cited judgments and current law.

---

## License

MIT License.
