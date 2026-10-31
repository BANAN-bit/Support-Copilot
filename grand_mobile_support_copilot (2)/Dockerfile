FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PORT=8000 \
    RULES_PATH=data/rules.json

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 copilot

COPY app/ ./app/
COPY prompts/ ./prompts/
COPY scripts/ ./scripts/
COPY eval/ ./eval/
COPY tests/ ./tests/
COPY Makefile ./
COPY pytest.ini ./

# Каталог для admin_feedback.jsonl и артефактов eval; данные организаторов
# в образ не копируются (см. §10.2) и монтируются в /app/data только для чтения.
RUN mkdir -p /app/data && chown -R copilot:copilot /app/data

USER copilot

EXPOSE 8000

# Liveness-проба (не зависит от смонтированных данных). Готовность проверяйте
# через GET /readyz: в degraded-режиме без rules.json она вернёт 503.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import json,urllib.request; json.load(urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2))"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
