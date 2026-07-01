FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv

WORKDIR /build

RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements-serve.txt .
RUN pip install --upgrade pip && pip install -r requirements-serve.txt

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken_cache

WORKDIR /app

RUN groupadd --system appuser \
    && useradd --system --gid appuser --home-dir /app --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv

RUN mkdir -p "$TIKTOKEN_CACHE_DIR" \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && cache_file="$(find "$TIKTOKEN_CACHE_DIR" -type f | head -n 1)" \
    && cp "$cache_file" "$TIKTOKEN_CACHE_DIR/cl100k_base.tiktoken"

COPY api_client.py config.py logger.py rag_cli.py ./
COPY access/ access/
COPY cache/ cache/
COPY deploy/ deploy/
COPY document_loader/ document_loader/
COPY embeddings/ embeddings/
COPY hybrid/ hybrid/
COPY lexical/ lexical/
COPY observability/ observability/
COPY parent_store/ parent_store/
COPY query_rewrite/ query_rewrite/
COPY rag/ rag/
COPY rerank/ rerank/
COPY service/ service/
COPY text_cleaner/ text_cleaner/
COPY tokenization/ tokenization/
COPY vector_store/ vector_store/
COPY eval/fixtures/knowledge_base/ eval/fixtures/knowledge_base/
COPY eval/fixtures/query_rewrites.jsonl eval/fixtures/query_rewrites.jsonl

RUN chmod +x /app/deploy/docker-entrypoint.sh \
    && mkdir -p /app/logs "$TIKTOKEN_CACHE_DIR" \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

ENTRYPOINT ["/app/deploy/docker-entrypoint.sh"]
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
