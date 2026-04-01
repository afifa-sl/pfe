"""
query.py
--------
Point d'entrée des questions.
"""

import ollama
import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_PATH, LLM_MODEL, EMBED_MODEL, TOP_K
from sql_agent import is_sql_question, query_sql, DB_SCHEMA


# ── Initialisation ChromaDB ─────────────────────────────────────────
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embed_fn = embedding_functions.OllamaEmbeddingFunction(
        model_name=EMBED_MODEL,
        url="http://localhost:11434/api/embeddings",
    )
    return client.get_or_create_collection("documents", embedding_function=embed_fn)


# ── RAG ─────────────────────────────────────────────────────────────
def rag_query(question: str) -> str:
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=TOP_K)

    if not results["documents"] or not results["documents"][0]:
        return "Aucun document pertinent trouvé."

    context = "\n\n".join(results["documents"][0])

    prompt = f"""
Tu es un assistant expert.

Réponds uniquement avec les informations du contexte.

CONTEXTE :
{context}

QUESTION : {question}
RÉPONSE :
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


# ── SQL via LLM ─────────────────────────────────────────────────────
def format_natural(rows):
    if not rows:
        return "Aucun résultat."

    r = rows[0]
    return f"{r.get('NOM','')} {r.get('PRENOM','')} est {r.get('FONCTION','')} du {r.get('CHANTIER','')}."


def llm_sql_query(question: str, schema: str) -> str:
    import sqlite3
    import re
    from config import DB_PATH

    prompt = f"""
Tu es un expert SQL SQLite.

OBJECTIF :
Générer une requête SQL correcte.

RÈGLES :
- Utiliser les colonnes : NOM, PRENOM, FONCTION, CHANTIER
- Utiliser DISTINCT si liste
- Pas d'explication
- Seulement SQL

SCHÉMA :
{schema}

QUESTION :
{question}

SQL :
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0,
            "num_predict": 150
        }
    )

    sql = response["message"]["content"].strip()

    # Nettoyage
    sql = re.sub(r"```sql|```", "", sql).strip()
    sql = sql.split("\n")[0]

    print(f"🧠 SQL généré : {sql}")

    # 🔥 Correction intelligente (générale)
    if "from departement" in sql.lower() and "nom" not in sql.lower():
        sql = "SELECT NOM, PRENOM, FONCTION, CHANTIER FROM departement"

    if "from service" in sql.lower() and "nom" not in sql.lower():
        sql = "SELECT NOM, PRENOM, FONCTION, CHANTIER FROM service"

    if "from direction" in sql.lower() and "nom" not in sql.lower():
        sql = "SELECT NOM, PRENOM, FONCTION, CHANTIER FROM direction"

    print(f"🛠️ SQL corrigé : {sql}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]

    except sqlite3.Error as e:
        conn.close()
        return f"❌ Erreur SQL : {e}"

    finally:
        conn.close()

    # ✅ Réponse intelligente
    if any(x in question.lower() for x in ["qui", "chef", "responsable"]):
        return format_natural(rows)

    if not rows:
        return "Aucun résultat trouvé."

    # Liste classique
    lines = []
    for r in rows[:20]:
        line = " | ".join(f"{k}: {v}" for k, v in r.items() if v not in (None, "", "nan"))
        lines.append("  " + line)

    if len(rows) > 20:
        lines.append(f"  ... ({len(rows) - 20} lignes supplémentaires)")

    return "\n".join(lines)

# ── Routeur ─────────────────────────────────────────────────────────
def answer(question: str) -> str:
    print(f"\n🔍 Question : {question}")

    if is_sql_question(question):
        print("📊 → Route : SQL")
        result = query_sql(question, llm_generate_sql=llm_sql_query)
        if result:
            return result

    print("📄 → Route : RAG")
    return rag_query(question)


# ── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("✅ Système prêt. Tapez 'exit' pour quitter.\n")

    while True:
        q = input("Votre question : ").strip()

        if q.lower() in ("exit", "quit", "q"):
            break

        if q:
            print("\n" + answer(q) + "\n")