"""
ingest.py
---------
Ingère les DOCUMENTS (PDF, Word, TXT) dans ChromaDB pour le RAG.
⚠️  NE PAS ingérer les fichiers Excel → ils sont déjà dans SQLite.

Usage : python ingest.py
"""

import os
import uuid
import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_PATH, DOCUMENTS_PATH, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.OllamaEmbeddingFunction(
        model_name=EMBED_MODEL,
        url="http://localhost:11434/api/embeddings",
    )
    return client.get_or_create_collection("documents", embedding_function=embed_fn)


def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return [c for c in chunks if len(c.strip()) > 30]


def read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except ImportError:
            print("  ⚠️  pdfplumber non installé : pip install pdfplumber")
            return ""

    if ext in (".docx", ".doc"):
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            print("  ⚠️  python-docx non installé : pip install python-docx")
            return ""

    # Ignorer les Excel (déjà en SQLite)
    if ext in (".xlsx", ".xls", ".csv"):
        print(f"  ⏭️  Ignoré (données SQL) : {path}")
        return ""

    return ""


def ingest():
    if not os.path.exists(DOCUMENTS_PATH):
        os.makedirs(DOCUMENTS_PATH)
        print(f"📁 Dossier créé : {DOCUMENTS_PATH} — placez vos PDF/Word/TXT dedans.")
        return

    collection = get_collection()
    files = [f for f in os.listdir(DOCUMENTS_PATH) if not f.startswith(".")]

    if not files:
        print(f"📁 Aucun fichier dans {DOCUMENTS_PATH}")
        return

    for filename in files:
        path = os.path.join(DOCUMENTS_PATH, filename)
        print(f"📄 Ingestion : {filename}")
        text = read_file(path)
        if not text.strip():
            continue
        chunks = chunk_text(text)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metas = [{"source": filename, "chunk": i} for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids, metadatas=metas)
        print(f"   ✅ {len(chunks)} chunks indexés")

    print("\n✅ Ingestion terminée.")


if __name__ == "__main__":
    ingest()
