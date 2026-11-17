# Arabic/English IT Support Assistant

A small retrieval-augmented (RAG) service that answers IT support questions in Arabic or English from a knowledge base, classifies the request, and cites its sources. Python standard library only, exposed through a REST API, containerized with Docker.

## How it works

1. **Ingest:** Markdown articles in `knowledge_base/` are split into chunks. Each article starts with `category: <name>` and a `# Title`.
2. **Arabic NLP:** text is normalized (diacritics, tatweel, alef/yeh variants, Arabic digits) and lightly stemmed (common prefixes and suffixes).
3. **Retrieve:** TF-IDF vectors with cosine scoring return the top matching chunks.
4. **Classify:** the request category is scored from the retrieved chunks.
5. **Log:** every question, its language, category and whether it was answered is saved in a SQL database (SQLite, table `queries`) so the team can see what people ask and where the knowledge base has gaps. Questions are truncated to 500 characters and all queries are parameterized.
6. **Answer:** the best passages are returned with sources. If the best score is below a threshold, the assistant declines and points to the service desk instead of guessing.
7. **Optional LLM:** if `ANTHROPIC_API_KEY` is set, the retrieved passages are sent to an LLM that must answer only from them and cite them (`generated_by: "llm"` in the response). Without a key, or if the call fails, answers are extractive quotes of the best passages (`generated_by: "extractive"`). The model can be changed with `LLM_MODEL`.

## Run

```bash
python -m assistant.server            # http://localhost:8000
python -m unittest discover -s tests -t .
```

With LLM answers (Linux/macOS; on Windows PowerShell use `$env:ANTHROPIC_API_KEY="..."`):

```bash
ANTHROPIC_API_KEY=your-key python -m assistant.server
```

Docker:

```bash
docker build -t support-assistant .
docker run -p 8000:8000 support-assistant
```

## API

```bash
curl localhost:8000/health

curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "نسيت كلمة المرور"}'

curl 'localhost:8000/history?limit=5'   # recent questions from the SQL database
curl localhost:8000/stats                # totals and per-category counts

curl -X POST localhost:8000/classify -H 'Content-Type: application/json' \
  -d '{"text": "VPN keeps disconnecting"}'
```

## Limits

- TF-IDF is lexical, not semantic. Replacing `TfidfIndex` with embeddings and a vector database is the natural next step.
- The sample knowledge base is generic and contains no real bank data.
- History is stored unencrypted in a local SQLite file (`history.db`, or `/data/history.db` in Docker). For real bank use, avoid logging personal data, add authentication, and use a managed database such as PostgreSQL.
- The LLM call is covered by unit tests with a mocked HTTP layer; test it with your own key before relying on it.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs the tests, builds the Docker image and smoke-tests the container on every push.
