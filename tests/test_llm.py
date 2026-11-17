import io
import json
import unittest
from pathlib import Path

from assistant.llm import make_generator
from assistant.rag import SupportAssistant

KB = Path(__file__).resolve().parent.parent / "knowledge_base"


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class LlmTests(unittest.TestCase):
    def test_request_is_grounded_and_response_parsed(self):
        sent = {}

        def fake_open(req, timeout):
            sent["headers"] = dict(req.header_items())
            sent["body"] = json.loads(req.data)
            return FakeResponse(json.dumps(
                {"content": [{"type": "text", "text": "Use the portal [1]."}]}).encode())

        bot = SupportAssistant(KB, generator=make_generator("k", "m", opener=fake_open))
        result = bot.ask("How do I reset my password?")
        self.assertEqual(result["answer"], "Use the portal [1].")
        self.assertEqual(result["generated_by"], "llm")
        prompt = sent["body"]["messages"][0]["content"]
        self.assertIn("self-service portal", prompt)        # retrieved context
        self.assertIn("ONLY", sent["body"]["system"])        # grounding rule
        self.assertEqual(sent["headers"]["X-api-key"], "k")

    def test_falls_back_when_llm_fails(self):
        def broken(req, timeout):
            raise OSError("network down")

        bot = SupportAssistant(KB, generator=make_generator("k", "m", opener=broken))
        result = bot.ask("How do I reset my password?")
        self.assertEqual(result["generated_by"], "extractive")
        self.assertTrue(result["answered"])

    def test_default_is_extractive(self):
        result = SupportAssistant(KB).ask("printer paper jam")
        self.assertEqual(result["generated_by"], "extractive")


if __name__ == "__main__":
    unittest.main()
