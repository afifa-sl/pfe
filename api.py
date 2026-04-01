from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from query import answer
from sql_agent import is_sql_question, DB_SCHEMA

app = FastAPI(title="Assistant Organisationnel", version="2.0")

# ✅ Autoriser le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    source: str  # sql | rag


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/query", response_model=QuestionResponse)
def query_endpoint(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question vide")

    try:
        result = answer(req.question)
        source = "sql" if is_sql_question(req.question) else "rag"

        return QuestionResponse(
            question=req.question,
            answer=result,
            source=source
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    import sqlite3
    from config import DB_PATH

    conn = sqlite3.connect(DB_PATH)
    data = {}

    for table in ["direction", "departement", "service", "poste"]:
        data[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    conn.close()
    return data


@app.get("/schema")
def get_schema():
    return {"schema": DB_SCHEMA}