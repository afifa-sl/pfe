import os

# --- Chemins ---
DB_PATH = "organisation.db"          # Base SQLite (tes fichiers Excel convertis)
CHROMA_PATH = "chroma_db"           # Base vectorielle pour les documents RAG
DOCUMENTS_PATH = "documents/"       # Dossier pour tes PDF/Word/txt

# --- Modèle LLM ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")          # ou llama3, etc.
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# --- Paramètres RAG ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
