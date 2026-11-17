"""Optional LLM answer generation (standard library only).

Enabled when the ANTHROPIC_API_KEY environment variable is set. The model
receives ONLY the retrieved passages and is told to answer from them, so the
answer stays grounded in the knowledge base (the RAG pattern). If the call
fails for any reason, the assistant falls back to extractive answers.
"""
import json
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
SYSTEM_PROMPT = (
    "You are an IT support assistant for a bank. Answer ONLY from the numbered "
    "context passages. Reply in the same language as the question. If the "
    "context does not contain the answer, say you do not know and suggest "
    "contacting the IT service desk. Cite the passages you used like [1]. "
    "Never reveal or ask for passwords or one-time codes."
)


def make_generator(api_key: str, model: str, timeout: int = 20, opener=urllib.request.urlopen):
    def generate(question, chunks) -> str:
        context = "\n\n".join(f"[{i}] ({c.title})\n{c.text}" for i, c in enumerate(chunks, 1))
        body = {
            "model": model,
            "max_tokens": 500,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user",
                          "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        }
        req = urllib.request.Request(
            API_URL, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"})
        with opener(req, timeout=timeout) as resp:
            data = json.load(resp)
        return "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text").strip()

    return generate
