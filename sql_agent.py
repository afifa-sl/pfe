"""
sql_agent.py
------------
Répond aux questions sur l'organigramme via SQL (direction, département, service, poste).
"""

import sqlite3
import re
from config import DB_PATH


# ── Schéma lisible pour le LLM ──────────────────────────────────────────────
DB_SCHEMA = """
Tables disponibles dans la base SQLite :

1. direction(ID, AFFECT_PAR, SHORT_LIBELLE_DIRECTION, AFFECTATION, CHANTIER, MATRICULE, NOM, PRENOM, OBSERVATION, FONCTION)

2. departement(ID, AFFECT_PAR, SHORT_LIBELLE_DIRECTION, AFFECTATION, CHANTIER, MATRICULE, NOM, PRENOM, OBSERVATION, FONCTION)

3. service(ID, AFFECT_PAR, SHORT_LIBELLE_DIRECTION, AFFECTATION, CHANTIER, MATRICULE, NOM, PRENOM, OBSERVATION, FONCTION)

4. poste(ID, LIBELLE_POSTE_BASE, LIBELLE_POSTE, CD_ACTIVITE, CD_FILIERE, CD_SFILIERE,
         NUM_EMPLOI, LIBELLE_ACTIVITE, LIBELLE_FILIERE, LIBELLE_SOUS_FILIERE, CATEGORIE)
"""


# ── Détection question SQL ──────────────────────────────────────────────
SQL_KEYWORDS = [
    "direction", "département", "departement", "service",
    "poste", "emploi", "responsable", "chef", "directeur",
    "matricule", "drh", "dfc", "dtc", "dsi", "qhse",
    "combien", "liste", "qui est", "organigramme"
]


def is_sql_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in SQL_KEYWORDS)


# ── Exécution SQL ──────────────────────────────────────────────
def _run_sql(sql: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        rows = [dict(r) for r in conn.execute(sql).fetchall()]
    except sqlite3.Error as e:
        rows = [{"erreur": str(e)}]
    finally:
        conn.close()

    return rows


# ── Format résultat ──────────────────────────────────────────────
def _format_rows(rows: list[dict], max_rows: int = 20) -> str:
    if not rows:
        return "Aucun résultat trouvé."

    if "erreur" in rows[0]:
        return f"Erreur SQL : {rows[0]['erreur']}"

    lines = []
    for r in rows[:max_rows]:
        line = " | ".join(f"{k}: {v}" for k, v in r.items() if v not in (None, "", "nan"))
        lines.append("  " + line)

    if len(rows) > max_rows:
        lines.append(f"  ... ({len(rows) - max_rows} lignes supplémentaires)")

    return "\n".join(lines)


# ── Logique principale ──────────────────────────────────────────────
def query_sql(question: str, llm_generate_sql=None) -> str:
    q = question.lower()

    # 🎯 CORRECTION INTELLIGENTE (RAPIDE ET FIABLE)
    if "departement" in q and ("liste" in q or "lister" in q):
        rows = _run_sql("SELECT DISTINCT CHANTIER FROM departement")
        return f"**Liste des départements :**\n{_format_rows(rows)}"

    if "service" in q and ("liste" in q or "lister" in q):
        rows = _run_sql("SELECT DISTINCT CHANTIER FROM service")
        return f"**Liste des services :**\n{_format_rows(rows)}"

    if "direction" in q and ("liste" in q or "lister" in q):
        rows = _run_sql("SELECT DISTINCT SHORT_LIBELLE_DIRECTION FROM direction")
        return f"**Liste des directions :**\n{_format_rows(rows)}"

    # 🎯 Recherche par nom
    name_match = re.search(r"(?:qui est|trouver|chercher)\s+([a-zA-ZÀ-ÿ\s]+)", question, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip().upper()

        rows = _run_sql(f"""
            SELECT 'direction' as niveau, NOM, PRENOM, FONCTION, CHANTIER FROM direction WHERE NOM LIKE '%{name}%'
            UNION ALL
            SELECT 'departement', NOM, PRENOM, FONCTION, CHANTIER FROM departement WHERE NOM LIKE '%{name}%'
            UNION ALL
            SELECT 'service', NOM, PRENOM, FONCTION, CHANTIER FROM service WHERE NOM LIKE '%{name}%'
        """)

        return f"**Résultat pour {name} :**\n{_format_rows(rows)}"

    # 🧠 LLM (fallback)
    if llm_generate_sql:
        result = llm_generate_sql(question, DB_SCHEMA)
        if result:
            return result

    return "❌ Je n'ai pas compris la question."