"""관계형 저장소 — PostgreSQL(기본, Docker) 또는 SQLite.

2-테이블 구조: documents, entities. 함수 API는 백엔드와 무관하게 동일.
백엔드 선택: CFG.relational_backend (postgres | sqlite)
"""
from __future__ import annotations
from datetime import datetime, timezone

from config import CFG

_BACKEND = CFG.relational_backend
_PG = _BACKEND == "postgres"

if _PG:
    import psycopg2
    import psycopg2.extras
    PH = "%s"
else:
    import sqlite3
    PH = "?"

_SCHEMA_PG = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      SERIAL PRIMARY KEY,
    file_name   TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    raw_text    TEXT,
    created_at  TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS entities (
    entity_id   SERIAL PRIMARY KEY,
    doc_id      INTEGER NOT NULL REFERENCES documents(doc_id),
    entity_text TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    context     TEXT,
    UNIQUE(doc_id, entity_text, entity_type)
);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_text ON entities(entity_text);
"""

_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name   TEXT NOT NULL, file_type TEXT NOT NULL, raw_text TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(doc_id),
    entity_text TEXT NOT NULL, entity_type TEXT NOT NULL, context TEXT,
    UNIQUE(doc_id, entity_text, entity_type)
);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_text ON entities(entity_text);
"""

_FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create", "replace", "attach", "pragma", "grant")


def connect(db_path: str | None = None):
    if _PG:
        from urllib.parse import urlparse, unquote
        u = urlparse(CFG.database_url)
        return psycopg2.connect(
            host=u.hostname or "localhost", port=u.port or 5432,
            user=unquote(u.username or ""), password=unquote(u.password or ""),
            dbname=(u.path or "/").lstrip("/"), client_encoding="UTF8")
    conn = sqlite3.connect(db_path or CFG.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _cursor(conn):
    if _PG:
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    return conn.cursor()


def init_schema(conn) -> None:
    cur = conn.cursor()
    if _PG:
        cur.execute(_SCHEMA_PG)
    else:
        cur.executescript(_SCHEMA_SQLITE)
    conn.commit()


def insert_document(conn, file_name: str, file_type: str, raw_text: str) -> int:
    cur = _cursor(conn)
    now = datetime.now(timezone.utc).isoformat()
    sql = f"INSERT INTO documents(file_name, file_type, raw_text, created_at) VALUES ({PH},{PH},{PH},{PH})"
    if _PG:
        cur.execute(sql + " RETURNING doc_id", (file_name, file_type, raw_text, now))
        doc_id = cur.fetchone()["doc_id"]
    else:
        cur.execute(sql, (file_name, file_type, raw_text, now))
        doc_id = cur.lastrowid
    conn.commit()
    return int(doc_id)


def insert_entities(conn, doc_id: int, entities: list[dict]) -> list[int]:
    """중복(UNIQUE)은 무시. 새로 적재된 id 반환."""
    ids: list[int] = []
    cur = _cursor(conn)
    for e in entities:
        text = (e.get("entity_text") or "").strip()
        etype = (e.get("entity_type") or "").strip()
        if not text or not etype:
            continue
        ctx = (e.get("context") or "").strip()
        if _PG:
            cur.execute(
                f"INSERT INTO entities(doc_id, entity_text, entity_type, context) VALUES ({PH},{PH},{PH},{PH}) "
                "ON CONFLICT (doc_id, entity_text, entity_type) DO NOTHING RETURNING entity_id",
                (doc_id, text, etype, ctx))
            row = cur.fetchone()
            if row:
                ids.append(int(row["entity_id"]))
        else:
            cur.execute(
                f"INSERT OR IGNORE INTO entities(doc_id, entity_text, entity_type, context) VALUES ({PH},{PH},{PH},{PH})",
                (doc_id, text, etype, ctx))
            if cur.lastrowid:
                ids.append(int(cur.lastrowid))
    conn.commit()
    return ids


def fetch_entity(conn, entity_id: int) -> dict | None:
    cur = _cursor(conn)
    cur.execute(f"SELECT entity_id, doc_id, entity_text, entity_type, context FROM entities WHERE entity_id={PH}",
                (entity_id,))
    r = cur.fetchone()
    return dict(r) if r else None


def all_raw_texts(conn) -> list[str]:
    """기존 문서 본문 목록(중복 지문 계산용)."""
    cur = _cursor(conn)
    cur.execute("SELECT raw_text FROM documents")
    return [(r["raw_text"] or "") for r in cur.fetchall()]


def all_entities(conn) -> list[dict]:
    cur = _cursor(conn)
    cur.execute("SELECT entity_id, doc_id, entity_text, entity_type, context FROM entities")
    return [dict(r) for r in cur.fetchall()]


def execute_readonly(conn, sql: str) -> tuple[list[str], list[dict]]:
    """SELECT/WITH 만 허용. (columns, rows) 반환."""
    s = sql.strip().rstrip(";")
    low = s.lower()
    if not (low.startswith("select") or low.startswith("with")):
        raise ValueError("읽기 전용(SELECT/WITH)만 허용됩니다.")
    if any(f in low.split() for f in _FORBIDDEN):
        raise ValueError("금지된 키워드가 포함되어 있습니다.")
    cur = _cursor(conn)
    cur.execute(s)
    rows = [dict(r) for r in cur.fetchall()]
    cols = list(rows[0].keys()) if rows else [d[0] for d in (cur.description or [])]
    return cols, rows


def schema_text() -> str:
    dialect = "PostgreSQL" if _PG else "SQLite"
    return (
        "테이블 documents(doc_id, file_name, file_type, raw_text, created_at)\n"
        "테이블 entities(entity_id, doc_id, entity_text, entity_type, context)\n"
        "entities.doc_id -> documents.doc_id (외래키)\n"
        f"entity_type 값: {', '.join(CFG.ner_types)}\n"
        f"방언: {dialect}. 날짜 정렬은 documents.created_at 또는 entity_text(날짜) 사용. 읽기 전용 SELECT만 생성."
    )
