FROM python:3.12-slim

WORKDIR /app
RUN useradd --create-home appuser && mkdir /data && chown appuser /data
COPY assistant ./assistant
COPY knowledge_base ./knowledge_base

USER appuser
ENV PORT=8000 PYTHONUNBUFFERED=1 HISTORY_DB=/data/history.db
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health')"
CMD ["python", "-m", "assistant.server"]
