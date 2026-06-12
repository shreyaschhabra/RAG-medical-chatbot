import os
import csv
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder

from config import REPO_ID, EMBED_MODEL, RERANK_MODEL, DB_PATH, FETCH_K, TOP_K, TEMPERATURE, MAX_TOKENS
from ingest import build_vectorstore

load_dotenv()

def _token():
    try:
        return st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
    except Exception:
        return os.environ.get("HF_TOKEN", "")

SYSTEM = (
    "You are MediBot, a medical Q&A assistant. Answer ONLY from the context below. "
    "If the answer is not in the context, say 'I don't have that information in my knowledge base.'\n\n"
    "Context:\n{context}"
)

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
[data-testid="stChatMessage"] p { margin:0.4em 0; line-height:1.65; }
[data-testid="stChatMessage"] h2,[data-testid="stChatMessage"] h3 { color:#79c0ff; }
[data-testid="stChatMessage"] code { background:#0d1117; border:1px solid #30363d; border-radius:4px; padding:1px 6px; color:#d2a8ff; }
.source-card { background:#1e293b; border-left:3px solid #3b82f6; border-radius:6px; padding:10px 14px; margin-bottom:8px; font-size:0.82rem; color:#cbd5e1; }
.source-meta { font-size:0.72rem; color:#64748b; margin-top:5px; }
.block-container { padding-top:1.5rem; }
[data-testid="stChatInput"] { background:#161b22 !important; border:1px solid #30363d !important; border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading knowledge base and models — first run takes ~60 s…")
def load_resources():
    if not os.path.exists(f"{DB_PATH}/index.faiss"):
        build_vectorstore()

    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
    db  = FAISS.load_local(DB_PATH, emb, allow_dangerous_deserialization=True)

    all_docs   = list(db.docstore._dict.values())
    bm25       = BM25Retriever.from_documents(all_docs); bm25.k = FETCH_K
    faiss_ret  = db.as_retriever(search_kwargs={"k": FETCH_K})
    retriever  = EnsembleRetriever(retrievers=[bm25, faiss_ret], weights=[0.4, 0.6])

    reranker = CrossEncoder(RERANK_MODEL)

    endpoint = HuggingFaceEndpoint(
        repo_id=REPO_ID, temperature=TEMPERATURE, max_new_tokens=MAX_TOKENS,
        huggingfacehub_api_token=_token(), streaming=True, task="text-generation",
    )
    llm = ChatHuggingFace(llm=endpoint)
    return retriever, reranker, llm


def retrieve(query, retriever, reranker):
    docs   = retriever.invoke(query)
    pairs  = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    return [d for _, d in sorted(zip(scores, docs), reverse=True)][:TOP_K]


def stream_response(query, history, docs, llm):
    context  = "\n\n---\n\n".join(d.page_content for d in docs)
    hist_str = "\n".join(f"Human: {q}\nAssistant: {a}" for q, a in history[-3:])
    system   = SYSTEM.format(context=context)
    if hist_str:
        system += f"\n\nConversation so far:\n{hist_str}"
    msgs = [SystemMessage(content=system), HumanMessage(content=query)]
    try:
        for chunk in llm.stream(msgs):
            if chunk.content:
                yield chunk.content
    except Exception:
        yield llm.invoke(msgs).content


def log_feedback(question, answer, rating):
    with open("feedback.csv", "a", newline="") as f:
        csv.writer(f).writerow([datetime.now().isoformat(), question, answer[:300], rating])


def render_sources(sources):
    for src in sources:
        meta  = src["metadata"]
        page  = meta.get("page_label", meta.get("page", "?"))
        fname = os.path.basename(meta.get("source", "Encyclopedia"))
        st.markdown(
            f'<div class="source-card">{src["content"]}'
            f'<div class="source-meta">📖 {fname} — Page {page}</div></div>',
            unsafe_allow_html=True,
        )


# ── Sidebar ──────────────────────────────────────────────────────────────────
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


# ── Main ─────────────────────────────────────────────────────────────────────
st.title("🩺 MediBot")
st.caption("Ask anything from the Gale Encyclopedia of Medicine. Sources are shown with every answer.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

retriever, reranker, llm = load_resources()

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🩺" if msg["role"] == "assistant" else "🙋"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("sources"):
                with st.expander(f"📄 {len(msg['sources'])} source(s)"):
                    render_sources(msg["sources"])
            if msg.get("feedback") is None:
                c1, c2, _ = st.columns([1, 1, 20])
                if c1.button("👍", key=f"up_{i}"):
                    st.session_state.messages[i]["feedback"] = "positive"
                    prev = st.session_state.messages[i - 1]["content"] if i > 0 else ""
                    log_feedback(prev, msg["content"], "positive")
                    st.rerun()
                if c2.button("👎", key=f"dn_{i}"):
                    st.session_state.messages[i]["feedback"] = "negative"
                    prev = st.session_state.messages[i - 1]["content"] if i > 0 else ""
                    log_feedback(prev, msg["content"], "negative")
                    st.rerun()
            else:
                st.caption(f"Feedback: {msg['feedback']}")

if user_input := st.chat_input("Ask a medical question…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🙋"):
        st.markdown(user_input)

    docs = retrieve(user_input, retriever, reranker)

    with st.chat_message("assistant", avatar="🩺"):
        response = st.write_stream(stream_response(user_input, st.session_state.history, docs, llm))
        sources  = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
        with st.expander(f"📄 {len(sources)} source(s)"):
            render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant", "content": response,
        "sources": sources, "feedback": None,
    })
    st.session_state.history.append((user_input, response))
