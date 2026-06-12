"""RAG pipeline — resource loading, retrieval, reranking, and generation.

Import this module from app.py. All ML/LLM logic lives here; no UI code.
"""
import csv
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from sentence_transformers import CrossEncoder

from config import (
    DB_PATH, EMBED_MODEL, FETCH_K, MAX_TOKENS,
    RERANK_MODEL, REPO_ID, TEMPERATURE, TOP_K,
)
from ingest import build_vectorstore

load_dotenv()

_SYSTEM = (
    "You are MediBot, a medical Q&A assistant. "
    "Answer ONLY using the context below. "
    "If the answer is not in the context, say 'I don't have that information in my knowledge base.'"
    "\n\nContext:\n{context}"
)


def _hf_token() -> str:
    try:
        return st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
    except Exception:
        return os.environ.get("HF_TOKEN", "")


@st.cache_resource(show_spinner="Loading knowledge base and models — first run takes ~60 s…")
def load_resources():
    """Load (or build) FAISS, BM25, cross-encoder, and LLM. Cached for the session lifetime."""
    if not os.path.exists(f"{DB_PATH}/index.faiss"):
        build_vectorstore()

    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db  = FAISS.load_local(DB_PATH, emb, allow_dangerous_deserialization=True)

    all_docs  = list(db.docstore._dict.values())
    bm25      = BM25Retriever.from_documents(all_docs)
    bm25.k    = FETCH_K
    faiss_ret = db.as_retriever(search_kwargs={"k": FETCH_K})
    retriever = EnsembleRetriever(retrievers=[bm25, faiss_ret], weights=[0.4, 0.6])

    reranker = CrossEncoder(RERANK_MODEL)

    endpoint = HuggingFaceEndpoint(
        repo_id=REPO_ID,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_TOKENS,
        huggingfacehub_api_token=_hf_token(),
        streaming=True,
        task="text-generation",
    )
    llm = ChatHuggingFace(llm=endpoint)

    return retriever, reranker, llm


def retrieve_docs(query: str, retriever, reranker) -> list:
    """Hybrid-retrieve FETCH_K candidates, then rerank and return top TOP_K."""
    docs   = retriever.invoke(query)
    pairs  = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in ranked[:TOP_K]]


def stream_response(query: str, history: list, docs: list, llm):
    """Yield response tokens one by one, with the last 3 conversation turns as context."""
    context  = "\n\n---\n\n".join(d.page_content for d in docs)
    hist_str = "\n".join(f"Human: {q}\nAssistant: {a}" for q, a in history[-3:])
    system   = _SYSTEM.format(context=context)
    if hist_str:
        system += f"\n\nConversation so far:\n{hist_str}"

    messages = [SystemMessage(content=system), HumanMessage(content=query)]
    try:
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content
    except Exception:
        yield llm.invoke(messages).content


def log_feedback(question: str, answer: str, rating: str) -> None:
    """Append one feedback row to feedback.csv."""
    with open("feedback.csv", "a", newline="") as f:
        csv.writer(f).writerow([datetime.now().isoformat(), question, answer[:300], rating])
