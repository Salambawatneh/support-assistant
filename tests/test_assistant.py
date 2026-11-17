import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from assistant.rag import SupportAssistant, detect_language, normalize_arabic, tokenize
from assistant.server import make_handler

KB = Path(__file__).resolve().parent.parent / "knowledge_base"


class RagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = SupportAssistant(KB)

    def test_arabic_normalization(self):
        self.assertEqual(normalize_arabic("إِعَادَة"), "اعادة")
        self.assertEqual(normalize_arabic("مـــرور"), "مرور")

    def test_language_detection(self):
        self.assertEqual(detect_language("نسيت كلمة المرور"), "ar")
        self.assertEqual(detect_language("I forgot my password"), "en")

    def test_tokenize_strips_stopwords_and_affixes(self):
        self.assertIn("طابع", tokenize("الطابعات"))
        self.assertNotIn("the", tokenize("the printer"))

    def test_english_questions(self):
        cases = {
            "I forgot my password and my account is locked": "access",
            "VPN keeps disconnecting": "network",
            "printer paper jam": "hardware",
            "I received a phishing email": "security",
        }
        for question, category in cases.items():
            with self.subTest(question):
                r = self.bot.ask(question)
                self.assertTrue(r["answered"])
                self.assertEqual(r["category"]["category"], category)

    def test_arabic_questions(self):
        cases = {
            "نسيت كلمة المرور": "access",
            "الطابعة لا تعمل وفيها ورق عالق": "hardware",
            "كيف اتصل بالشبكة الخاصة الافتراضية": "network",
            "وصلتني رسالة تصيد مشبوهة": "security",
        }
        for question, category in cases.items():
            with self.subTest(question):
                r = self.bot.ask(question)
                self.assertTrue(r["answered"])
                self.assertEqual(r["language"], "ar")
                self.assertEqual(r["category"]["category"], category)

    def test_declines_unrelated_question(self):
        r = self.bot.ask("What is the capital of Japan?")
        self.assertFalse(r["answered"])
        self.assertEqual(r["sources"], [])

    def test_custom_generator_receives_context(self):
        seen = {}
        bot = SupportAssistant(KB, generator=lambda q, ctx: seen.update(n=len(ctx)) or "ok")
        self.assertEqual(bot.ask("reset my password")["answer"], "ok")
        self.assertGreater(seen["n"], 0)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(SupportAssistant(KB)))
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.load(resp)
        except urllib.error.HTTPError as err:
            return err.code, json.load(err)

    def test_health(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health") as resp:
            self.assertEqual(json.load(resp)["status"], "ok")

    def test_ask(self):
        status, body = self._post("/ask", {"question": "كيف أعيد تعيين كلمة المرور"})
        self.assertEqual(status, 200)
        self.assertTrue(body["answered"])

    def test_classify(self):
        status, body = self._post("/classify", {"text": "VPN not connecting"})
        self.assertEqual((status, body["category"]), (200, "network"))

    def test_validation(self):
        self.assertEqual(self._post("/ask", {"question": ""})[0], 400)
        self.assertEqual(self._post("/nope", {"x": 1})[0], 404)


if __name__ == "__main__":
    unittest.main()
