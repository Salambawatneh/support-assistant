"""REST API for the support assistant (standard library only).

GET  /health
GET  /history?limit=20   recent questions (from the SQL database)
GET  /stats              totals and per-category counts
POST /ask       {"question": "..."}  -> answer + category + sources
POST /classify  {"text": "..."}      -> category + confidence
"""
import json
import os
from urllib.parse import parse_qs, urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .llm import make_generator
from .rag import SupportAssistant
from .store import HistoryStore

KB_DIR = Path(os.environ.get("KB_DIR", Path(__file__).resolve().parent.parent / "knowledge_base"))
MAX_BODY = 16 * 1024


def make_handler(assistant: SupportAssistant, store: HistoryStore = None):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                raise ValueError("Body missing or too large")
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Body must be a JSON object")
            return data

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/health":
                self._send(200, {"status": "ok", "chunks": len(assistant.chunks),
                                 "categories": assistant.categories})
            elif url.path == "/history" and store:
                try:
                    limit = int(parse_qs(url.query).get("limit", ["20"])[0])
                except ValueError:
                    return self._send(400, {"error": "'limit' must be a number"})
                self._send(200, {"queries": store.recent(limit)})
            elif url.path == "/stats" and store:
                self._send(200, store.stats())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                data = self._read_json()
                if self.path == "/ask":
                    q = str(data.get("question", "")).strip()
                    if not q:
                        raise ValueError("'question' is required")
                    result = assistant.ask(q)
                    if store:
                        store.log(q, result)
                    self._send(200, result)
                elif self.path == "/classify":
                    t = str(data.get("text", "")).strip()
                    if not t:
                        raise ValueError("'text' is required")
                    self._send(200, assistant.classify(t))
                else:
                    self._send(404, {"error": "not found"})
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": str(exc)})

        def log_message(self, fmt, *args):  # concise logs
            print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    return Handler


def main():
    generator = None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        generator = make_generator(api_key, os.environ.get("LLM_MODEL", "claude-sonnet-5-5"))
        print("LLM answers enabled", flush=True)
    assistant = SupportAssistant(KB_DIR, generator=generator)
    store = HistoryStore(os.environ.get("HISTORY_DB", "history.db"))
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(assistant, store))
    print(f"Support assistant listening on :{port} ({len(assistant.chunks)} chunks)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
