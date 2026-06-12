# MediBot — RAG Medical Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) chatbot that answers medical questions by grounding every response in the *Gale Encyclopedia of Medicine (2nd Edition)*. Answers never go beyond the source material; every response includes expandable page-level citations.

**Live demo:** [Streamlit Community Cloud](https://rag-medical-chatbot.streamlit.app) *(deploy link — update after deployment)*

---

## Features

| Feature | Implementation |
|---|---|
| Conversational memory | Last 3 turns injected into every prompt |
| Hybrid retrieval | BM25 sparse + FAISS dense, score-fused (0.4 / 0.6) |
| Cross-encoder reranking | `ms-marco-MiniLM-L-6-v2` reranks 8 candidates to top 3 |
| Streaming responses | Token-by-token via `st.write_stream` |
| Page-level citations | Expandable source cards with page numbers |
| Feedback logging | Thumbs up/down per answer written to `feedback.csv` |
| Auto vectorstore build | Rebuilds FAISS index from `data/` if missing |
| Docker support | Single `docker-compose up` deployment |
| Offline evaluation | 50 ground-truth QA pairs with ROUGE + context recall metrics |

---

## Architecture

```
User Query
    |
    v
[Streamlit UI — app.py]
    |
    |-- Hybrid Retrieval (BM25 + FAISS EnsembleRetriever, k=8)
    |       |
    |       v
    |-- Cross-Encoder Reranking (top 3 of 8)
    |
    |-- Context + Chat History --> System Prompt
    |
    v
[ChatHuggingFace — Qwen2.5-7B-Instruct (Streaming)]
    |
    v
Answer + Source Citations
```

**Ingestion pipeline** (`ingest.py`, run once):
```
data/*.pdf --> PyPDFLoader --> RecursiveCharacterTextSplitter (500/50)
           --> all-MiniLM-L6-v2 Embeddings --> FAISS.save_local()
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Qwen/Qwen2.5-7B-Instruct (HuggingFace Inference Endpoint) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| Vector Store | FAISS (persistent, disk-backed) |
| Sparse Retrieval | BM25 (rank-bm25) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Orchestration | LangChain 0.3.x |
| PDF Ingestion | LangChain PyPDFLoader |
| Containerisation | Docker + Docker Compose |
| Eval Metrics | ROUGE-1/2/L, Context Recall, Latency P50/P95 |

---

## Project Structure

```
.
├── app.py                  Main Streamlit application
├── ingest.py               One-time ingestion pipeline
├── config.py               All tunable parameters
├── eval.py                 50-question offline evaluation
├── requirements.txt        Python dependencies
├── Dockerfile
├── docker-compose.yml
├── data/
│   └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf
└── vectorstore/
    └── db_faiss/           Pre-built FAISS index (committed to repo)
```

---

## Setup

### Local (pip)

```bash
git clone https://github.com/shreyaschhabra/RAG-medical-chatbot.git
cd RAG-medical-chatbot
pip install -r requirements.txt
echo "HF_TOKEN=your_token_here" > .env
streamlit run app.py
```

### Local (Docker)

```bash
echo "HF_TOKEN=your_token_here" > .env
docker-compose up --build
```

App is at `http://localhost:8501`.

### Rebuild the vectorstore

Only needed if you add new PDFs to `data/`:

```bash
python ingest.py
```

---

## Streamlit Community Cloud Deployment

1. Fork or push this repo to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set **Main file path** to `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   HF_TOKEN = "your_huggingface_token"
   ```
5. Click **Deploy**.

The vectorstore is committed to the repo so cold-start time is fast (~30 s for model loading only).

---

## Configuration

All parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `REPO_ID` | `Qwen/Qwen2.5-7B-Instruct` | HuggingFace model |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `RERANK_MODEL` | `ms-marco-MiniLM-L-6-v2` | Cross-encoder for reranking |
| `FETCH_K` | `8` | Candidate docs fetched before reranking |
| `TOP_K` | `3` | Final docs after reranking |
| `CHUNK_SIZE` | `500` | Characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `TEMPERATURE` | `0.4` | LLM sampling temperature |
| `MAX_TOKENS` | `512` | Max new tokens per response |

---

## Evaluation

Run the offline evaluation against 50 ground-truth medical QA pairs:

```bash
python eval.py
```

Results are printed to console and saved to `eval_results.json`.

### Metrics Computed

| Metric | Description |
|---|---|
| ROUGE-1 F1 | Unigram overlap between answer and ground truth |
| ROUGE-2 F1 | Bigram overlap |
| ROUGE-L F1 | Longest common subsequence F1 |
| Context Recall | Word-level overlap between retrieved chunks and ground truth |
| Avg Latency (s) | Mean end-to-end response time |
| P50 / P95 Latency | Median and 95th-percentile response time |

### Results

| Metric | Score | Notes |
|---|---|---|
| Context Recall | **0.5749** | ROUGE-L recall of retrieved context vs ground truth |
| Hit Rate @5 | **1.00** | Every query retrieved at least one relevant chunk |
| MRR @5 | **1.00** | Top-ranked chunk was always relevant |
| Avg Retrieval Latency | 0.452 s | Hybrid BM25 + FAISS + cross-encoder, FETCH_K=15 |
| P50 Latency | 0.335 s | Median query |
| P95 Latency | 1.239 s | 95th-percentile query |

> Hit Rate and MRR use ROUGE-L recall ≥ 0.12 as the relevance signal (stemmed). ROUGE F1 scores are low by design: retrieving 5 × 1000-char encyclopedia chunks against a 150-char ground-truth summary gives high recall but low precision, pulling F1 down — the LLM synthesises the answer from the verbose context. Context Recall of 0.57 means retrieved passages contain 57% of expected answer content on average.

---

## Knowledge Base

**Source:** Gale Encyclopedia of Medicine, 2nd Edition  
**Coverage:** Diseases, conditions, symptoms, diagnostics, treatments, and procedures  
**Index stats:** ~12 MB PDF → ~3,000 chunks (500 chars, 50 overlap) → 384-dim FAISS flat index

---

## Limitations

- No conversation history is visible to the LLM beyond the last 3 turns.
- Responses are grounded in a 2nd edition encyclopedia; current clinical guidelines may differ.
- HuggingFace Inference API latency varies with server load.
- Scanned/image-based PDFs are not supported by the ingestion pipeline.
- `feedback.csv` is ephemeral on Streamlit Cloud (resets on each deploy).

---

## Disclaimer

MediBot is a research and educational tool. It is not a substitute for professional medical advice, diagnosis, or treatment.

---
