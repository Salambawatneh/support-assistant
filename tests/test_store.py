import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from assistant.rag import SupportAssistant
from assistant.server import make_handler
from assistant.store import HistoryStore

KB = Path(__file__).resolve().parent.parent / "knowledge_base"


class StoreTests(unittest.TestCase):
    def test_log_recent_and_stats(self):
        store, bot = HistoryStore(), SupportAssistant(KB)
        for q in ["I forgot my password", "printer paper jam", "best pizza recipe"]:
            store.log(q, bot.ask(q))
        self.assertEqual([r["question"] for r in store.recent(2)],
                         ["best pizza recipe", "printer paper jam"])
        s = store.stats()
        self.assertEqual((s["total"], s["answered"], s["unanswered"]), (3, 2, 1))
        self.assertIn("access", [c["category"] for c in s["by_category"]])

    def test_sql_injection_is_stored_as_plain_text(self):
        store = HistoryStore()
        evil = "x'); DROP TABLE queries;--"
        store.log(evil, {"answered": False})
        self.assertEqual(store.recent(1)[0]["question"], evil)
        self.assertEqual(store.stats()["total"], 1)

    def test_long_question_is_truncated(self):
        store = HistoryStore()
        store.log("a" * 5000, {"answered": False})
        self.assertEqual(len(store.recent(1)[0]["question"]), 500)


class ApiHistoryTests(unittest.TestCase):
    def test_history_endpoints(self):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(SupportAssistant(KB), HistoryStore()))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            req = urllib.request.Request(
                base + "/ask", data=json.dumps({"question": "VPN not connecting"}).encode(),
                method="POST", headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req).read()
            hist = json.load(urllib.request.urlopen(base + "/history?limit=5"))
            self.assertEqual(hist["queries"][0]["category"], "network")
            self.assertEqual(json.load(urllib.request.urlopen(base + "/stats"))["total"], 1)
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
