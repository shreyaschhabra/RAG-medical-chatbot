"""Run once to (re)build the FAISS vectorstore from PDFs in data/."""
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import DATA_PATH, DB_PATH, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def build_vectorstore():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)
    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = FAISS.from_documents(chunks, emb)
    db.save_local(DB_PATH)
    print(f"Built vectorstore: {len(chunks)} chunks -> {DB_PATH}")
    return db


if __name__ == "__main__":
    build_vectorstore()
