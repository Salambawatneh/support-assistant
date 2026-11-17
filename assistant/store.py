"""SQL storage for question history and usage statistics (SQLite, standard library).

Every question, the detected language/category and whether it was answered is
saved so the support team can see what people ask and where the knowledge base
has gaps (unanswered questions). All queries are parameterized.
"""
import sqlite3
import threading

MAX_QUESTION_CHARS = 500

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at     TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    question     TEXT    NOT NULL,
    language     TEXT,
    category     TEXT,
    answered     INTEGER NOT NULL,
    generated_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_queries_category ON queries(category);
"""


class HistoryStore:
    def __init__(self, path: str = ":memory:"):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    def log(self, question: str, result: dict) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO queries (question, language, category, answered, generated_by) "
                "VALUES (?, ?, ?, ?, ?)",
                (question[:MAX_QUESTION_CHARS], result.get("language"),
                 result.get("category", {}).get("category"),
                 1 if result.get("answered") else 0, result.get("generated_by")))

    def recent(self, limit: int = 20) -> list[dict]:
        limit = max(1, min(int(limit), 100))
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, asked_at, question, language, category, answered, generated_by "
                "FROM queries ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        with self._lock:
            by_category = self._conn.execute(
                "SELECT COALESCE(category, 'unknown') AS category, COUNT(*) AS total, "
                "SUM(answered) AS answered FROM queries GROUP BY category "
                "ORDER BY total DESC").fetchall()
            totals = self._conn.execute(
                "SELECT COUNT(*) AS total, COALESCE(SUM(answered), 0) AS answered "
                "FROM queries").fetchone()
        return {"total": totals["total"], "answered": totals["answered"],
                "unanswered": totals["total"] - totals["answered"],
                "by_category": [dict(r) for r in by_category]}
