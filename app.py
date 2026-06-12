"""MediBot — Streamlit UI.

This file contains only presentation logic.
All ML / RAG work is delegated to pipeline.py.
"""
import csv
import os

import streamlit as st

from config import REPO_ID
from pipeline import load_resources, log_feedback, retrieve_docs, stream_response

# ── Page config & CSS ─────────────────────────────────────────────────────────
st.set_page_config(page_title="MediBot", page_icon="🩺", layout="wide")
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background:#0d1117; color:#c9d1d9; }
[data-testid="stSidebar"] { background:#161b22; border-right:1px solid #21262d; }
[data-testid="stChatMessage"] {
    background:#161b22; border:1px solid #21262d; border-radius:10px;
    margin:8px 0; padding:14px 18px; overflow:visible !important; height:auto !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background:#1a2233; border-color:#2a3a55;
}
[data-testid="stChatMessage"] p  { margin:0.4em 0; line-height:1.65; }
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 { color:#79c0ff; }
[data-testid="stChatMessage"] code {
    background:#0d1117; border:1px solid #30363d;
    border-radius:4px; padding:1px 6px; color:#d2a8ff;
}
.source-card {
    background:#1e293b; border-left:3px solid #3b82f6; border-radius:6px;
    padding:10px 14px; margin-bottom:8px; font-size:0.82rem; color:#cbd5e1;
}
.source-meta { font-size:0.72rem; color:#64748b; margin-top:5px; }
.block-container { padding-top:1.5rem; }
[data-testid="stChatInput"] {
    background:#161b22 !important; border:1px solid #30363d !important; border-radius:10px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_sources(sources: list) -> None:
    for src in sources:
        meta  = src["metadata"]
        page  = meta.get("page_label", meta.get("page", "?"))
        fname = os.path.basename(meta.get("source", "Encyclopedia"))
        st.markdown(
            f'<div class="source-card">{src["content"]}'
            f'<div class="source-meta">📖 {fname} — Page {page}</div></div>',
            unsafe_allow_html=True,
        )


def render_feedback_buttons(i: int, msg: dict) -> None:
    prev_content = st.session_state.messages[i - 1]["content"] if i > 0 else ""
    c1, c2, _ = st.columns([1, 1, 20])
    if c1.button("👍", key=f"up_{i}"):
        st.session_state.messages[i]["feedback"] = "positive"
        log_feedback(prev_content, msg["content"], "positive")
        st.rerun()
    if c2.button("👎", key=f"dn_{i}"):
        st.session_state.messages[i]["feedback"] = "negative"
        log_feedback(prev_content, msg["content"], "negative")
        st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/caduceus.png", width=60)
    st.title("MediBot")
    st.caption("RAG-powered medical Q&A")
    st.divider()
    st.markdown("**Knowledge Base**  \nGale Encyclopedia of Medicine (2nd ed.)")
    st.markdown(f"**Model**  \n`{REPO_ID}`")
    st.markdown("**Retrieval**  \nHybrid BM25 + FAISS + Cross-Encoder Reranking")
    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()

    if os.path.exists("feedback.csv"):
        with open("feedback.csv") as f:
            rows = list(csv.reader(f))
        pos = sum(1 for r in rows if r and r[-1] == "positive")
        neg = sum(1 for r in rows if r and r[-1] == "negative")
        if rows:
            st.caption(f"Feedback — 👍 {pos}  👎 {neg}")

    st.divider()
    st.caption("Not a substitute for professional medical advice.")


# ── Session state defaults ────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []     # list of (question, answer) tuples


# ── Load ML resources (cached) ────────────────────────────────────────────────

retriever, reranker, llm = load_resources()


# ── Main panel ────────────────────────────────────────────────────────────────

st.title("🩺 MediBot")
st.caption("Ask anything from the Gale Encyclopedia of Medicine. Every answer includes page-level citations.")

# Render existing conversation
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🩺" if msg["role"] == "assistant" else "🙋"):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if msg.get("sources"):
                with st.expander(f"📄 {len(msg['sources'])} source(s)"):
                    render_sources(msg["sources"])

            if msg.get("feedback") is None:
                render_feedback_buttons(i, msg)
            else:
                st.caption(f"Feedback: {msg['feedback']}")

# Handle new user message
if user_input := st.chat_input("Ask a medical question…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🙋"):
        st.markdown(user_input)

    docs = retrieve_docs(user_input, retriever, reranker)

    with st.chat_message("assistant", avatar="🩺"):
        response = st.write_stream(stream_response(user_input, st.session_state.history, docs, llm))
        sources  = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
        with st.expander(f"📄 {len(sources)} source(s)"):
            render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
        "feedback": None,
    })
    st.session_state.history.append((user_input, response))
