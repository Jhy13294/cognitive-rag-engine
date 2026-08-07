# Cognitive RAG Engine

[![Offline Test Gate](https://github.com/Jhy13294/cognitive-rag-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/Jhy13294/cognitive-rag-engine/actions/workflows/tests.yml)

## One-liner

An engineering-oriented RAG system for enterprise knowledge bases, where both the retrieval funnel and generation quality are held behind evaluation gates. The focus is on evaluated retrieval, hybrid search, reranking, ACL pre-filtering, caching, observability, and an async FastAPI service — not on being a single-file demo.

## Why this project

Most RAG examples stop at "PDF → embedding → similarity search → answer." That is enough for a notebook but hides every decision that decides whether retrieval is correct, whether the answer is grounded, and whether the system is safe to put behind a service. This project treats those as the actual work: retrieval quality is measured against a frozen baseline on every push, generation quality is scored by an LLM-as-judge track, access control filters before scoring, and the service ships with caching, observability, and health probes. It is production-shaped rather than production-complete — the boundaries are spelled out under [Production caveats](#production-caveats).

## Architecture

```text
Document ingestion (TXT / Markdown / PDF+OCR / Word)
  → cleaning / chunking (parent-child)
  → embedding
  → dense retrieval + BM25 sparse retrieval
  → RRF hybrid fusion
  → rerank
  → parent expansion
  → context packing
  → generation with citations
  → evaluation / metrics / cache / ACL pre-filtering
```

ACL pre-filtering, Redis caching, and observability wrap this funnel rather than sitting inside it: access filters apply before scoring, caches wrap the retrieve/answer calls, and audit/metrics are a fail-open side channel. See [docs/architecture.md](docs/architecture.md) for the data-flow detail.

## Key features

- **Evaluated retrieval** — deterministic golden-set evaluation (hit rate, MRR, recall, negative-query metrics) with a frozen baseline the CI gate asserts to six decimals on every push.
- **Hybrid retrieval** — dense + BM25 with RRF fusion, plus deterministic multi-query rewriting and cross-query fusion.
- **Rerank and parent-child chunking** — pluggable reranking with dense fallback; child chunks for retrieval, parent chunks expanded for generation.
- **Context packing** — whole-block inclusion, deterministic deduplication, contiguous citations, and token-aware budgets on the generation input.
- **Async FastAPI service** — `/ingest`, `/query`, and SSE `/query/stream`, with a threadpool offload for the synchronous funnel and sanitized error mapping.
- **Redis cache and ACL/RBAC pre-filtering** — three cache layers (query embedding / retrieval / answer) with atomic corpus-version invalidation, and fail-closed ACL filtering applied before candidate scoring with the effective ACL folded into cache keys.
- **Observability and health gates** — bounded-label Prometheus metrics, sanitized JSON-line audit, request IDs, and split liveness/readiness probes; a CI suite pins exact test accounting to prevent silent-skip false greens.

## Quick start

Fastest look, no Docker and no API keys — run the offline retrieval evaluation over the golden set:

```bash
pip install -r requirements.txt
python -m eval.run
```

For a full end-to-end walk-through (offline eval **and** a zero-key full-stack HTTP demo with ACL, citations, cache counters, and health probes), see **[examples/minimal_demo.md](examples/minimal_demo.md)**.

## Evaluation

Retrieval and generation quality are evaluated on separate tracks, and the offline replay path is the default reproducible one:

- **Deterministic retrieval evaluation** — `python -m eval.run` scores 20 labeled queries (16 positive and 4 negative) against five synthetic Markdown documents of 765–887 bytes with `HashEmbeddingProvider`. CI re-runs dense, hybrid, parent-child, and rerank profiles and asserts positive-query MRR@3 to six decimals; negative-query behavior is reported separately.
- **Generation quality (Ragas)** — a live LLM-as-judge track scores Faithfulness, Answer Relevance, and Context Precision/Recall with pinned temperatures, repetition, and median-based gating; CI replays a frozen verdict fixture fully offline. Scores are noisy judge estimates on a small hand-labeled set, not a production quality floor.

See [Generation Quality Evaluation](#generation-quality-evaluation) below and [docs/benchmark.md](docs/benchmark.md) for the zero-paid local latency/QPS methodology.

## Production caveats

This is an enterprise-oriented, production-shaped RAG backend, not a turnkey production-ready product. The boundaries are deliberate and stated up front:

- **Built-in authentication is planned but not implemented.** The service does not validate JWTs or sessions.
- **ACL depends on a trusted upstream gateway.** The principal is read from a trusted upstream header; you must deploy behind an authentication gateway that owns that header. Body-provided principals are rejected unless explicitly enabled for local testing.
- **The offline replay/golden baseline is the default reproducible path.** Live LLM-as-judge evaluation and paid retrieval require API keys; the frozen baselines and offline gates are what run keyless in CI.
- **Evaluation scores are judge estimates on a small set.** The golden set is roughly 20 hand-labeled queries; treat scores as directional, not as a production quality guarantee.

## Project Status

Current milestone: evaluated retrieval funnel plus async HTTP service layer, optional Redis caching, ACL/RBAC pre-filtering, fail-open observability, and deployment health gates.

Current highlights:

- Deterministic dense, hybrid, parent-child, and rerank evaluation profiles are enforced in CI to six decimal places.
- TXT, Markdown, PDF/OCR, and Word ingestion feed exact source-slice chunks; fenced Markdown code remains searchable by default.
- Dense and BM25 retrieval, RRF fusion, reranking, multi-query rewriting, parent expansion, and token-aware context packing share one staged pipeline.
- The async FastAPI service provides ingest, query, SSE streaming, Redis caching, ACL/RBAC pre-filtering, observability, and split liveness/readiness probes.
- Memory providers support keyless local verification; Qdrant, MySQL, Redis, OpenAI-compatible generation, OpenAI embeddings, and Cohere reranking are optional integrations.

See the maintained [capability inventory and explicit gaps](docs/capabilities.md). Built-in authentication, JWT validation, and session management remain out of scope.

## Advanced Learning Notes

Start here when you want to understand the engineering choices behind the project:

- **[Technology Selection](docs/tech-selection.md)**: why the project uses the current embedding providers, vector stores, retrieval stack, service protocol, access filters, caching, and observability design.
- **[Architecture](docs/architecture.md)**: system architecture, ingest/query data flow, retrieval funnel, access control, cache layout, observability, and evaluation boundaries.
- **[Frozen Retrieval Baselines](docs/retrieval-baselines.md)**: current CI numbers, the parent-child migration delta, tie-order controls, and the dependency-lock contract.
- **[Crash / Pitfall Log](docs/dev-log-crashing.md)**: hard-earned debugging notes about metric pollution, fake streaming, async traps, cache invalidation, access boundaries, audit leakage, and token accounting.
- **[Local Benchmarking](docs/benchmark.md)**: reproducible local latency/QPS runs against the keyless compose stack; numbers exclude real paid generation.

## Directory Structure

```text
.
├── api_client.py              # LLM API client
├── config.py                  # Environment-based configuration
├── logger.py                  # Logging utilities
├── main.py                    # CLI chat entry point
├── rag_cli.py                 # RAG application CLI
├── service/                   # FastAPI HTTP adapter
├── cache/                     # Redis cache decorators and serialization
├── access/                    # ACL/RBAC filters, resolvers, and MySQL metadata adapter
├── observability/             # Structured audit, metrics registry, request IDs, and alert hooks
├── ci/                        # Push CI gates: full-suite accounting and frozen retrieval baselines
├── .github/workflows/         # CI: offline test gate, Ragas replay gate, gated live evaluation
├── Dockerfile                 # Service image build
├── docker-compose.yml         # Full local stack: Qdrant, Redis, MySQL, app
├── LICENSE                    # MIT License
├── requirements.txt           # Development dependencies
├── requirements-ci.in        # Exact direct dependencies for CI
├── requirements-ci.constraints # Cross-platform transitive compatibility pins
├── requirements-ci.txt       # Universal compiled CI lock
├── requirements-serve.txt     # Pinned runtime dependencies for the service image
├── document_loader/
│   ├── base.py                # Document model and loader interface
│   ├── chunking.py            # Text chunking
│   ├── loader.py              # Unified loading entry point
│   ├── md_loader.py           # Markdown loader
│   ├── pdf_loader.py          # PDF loader with scan detection, OCR fallback, and tables
│   ├── txt_loader.py          # TXT loader
│   └── word_loader.py         # Word loader
├── embeddings/
│   ├── base.py                # Embedding interface and vector utilities
│   ├── openai_provider.py     # OpenAI embedding provider
│   └── hash_provider.py       # Deterministic local provider for tests
├── vector_store/
│   ├── base.py                # Vector store interface and record models
│   ├── factory.py             # Vector store factory
│   ├── memory_store.py        # In-memory vector store
│   └── qdrant_store.py        # Qdrant vector store adapter
├── parent_store/
│   ├── base.py                # Parent chunk key-value store interface
│   └── memory_store.py        # In-memory parent store
├── tokenization/
│   └── counter.py             # TokenCounter, tiktoken counter, and offline fallback
├── eval/
│   ├── golden_set.jsonl       # Retrieval golden set with relevant source lists
│   ├── baseline.py            # Deterministic HashEmbeddingProvider baseline
│   ├── metrics.py             # hit_rate, MRR, recall, negative metrics
│   ├── reporting.py           # JSON and Markdown report rendering
│   ├── run.py                 # python -m eval.run entry point
│   ├── ragas_run.py           # python -m eval.ragas_run replay/live/compare entry point
│   ├── ragas_evaluation.py    # Ragas replay gate, fixture validation, and gating rules
│   ├── ragas_live.py          # Gated live judge recording
│   ├── bge_embedding.py       # Local fastembed BGE embedding for answer relevance
│   ├── gemini_embedding.py    # Optional Gemini embedding for answer relevance
│   ├── fixtures/              # Evaluation knowledge base and frozen fixtures
│   │   ├── ragas_verdicts.jsonl   # Committed Ragas verdict baseline (replay gate input)
│   │   └── query_rewrites.jsonl   # Deterministic query rewrite fixture
│   └── reports/               # Generated evaluation reports
├── bench/
│   ├── run.py                 # python -m bench.run zero-paid local latency/QPS benchmark
│   └── reports/               # Generated benchmark reports (git-ignored)
├── examples/
│   ├── minimal_demo.md        # Offline eval + zero-key full-stack HTTP demo
│   ├── demo_chat_stub.py      # Local deterministic chat stub for the demo
│   └── docker-compose.demo.yml # Zero-key compose override for the demo
├── lexical/
│   ├── tokenizer.py           # Shared normalization and tokenization
│   └── bm25.py                # IDF, BM25, and lexical scoring primitives
├── hybrid/
│   ├── models.py              # Ranked retrieval result model
│   ├── bm25_retriever.py      # Sparse BM25 retriever over VectorRecord corpus
│   └── rrf.py                 # Reciprocal Rank Fusion
├── rerank/
│   ├── base.py                # Reranker interface and result models
│   ├── deterministic.py       # Offline deterministic lexical reranker
│   ├── cohere_provider.py     # Cohere Rerank provider with retries
│   └── factory.py             # Reranker factory
├── query_rewrite/
│   ├── base.py                # Query rewrite interface and config
│   ├── deterministic.py       # Offline fixture-backed query rewriter
│   ├── chat.py                # Chat-backed query rewriter
│   └── factory.py             # Query rewriter factory
├── rag/
│   ├── pipeline.py            # High-level RAG orchestration and compatibility exports
│   ├── retrieval_orchestrator.py # Dense/hybrid/multi-query retrieval and rerank
│   ├── source_finalizer.py    # Parent expansion and sibling-child collapse
│   ├── prompt_builder.py      # System prompt, scope instructions, context assembly
│   ├── question_classifier.py # Deterministic question-shape heuristics
│   ├── models.py              # Shared source/response data models
│   └── context_packing.py     # Post-retrieval context packing
├── text_cleaner/
│   └── cleaner.py             # Text cleaning
├── tests/
│   ├── fixtures/              # Sample TXT and Markdown fixtures
│   ├── test_acl.py
│   ├── test_acl_mysql_integration.py
│   ├── test_api_client.py
│   ├── test_bge_embedding.py
│   ├── test_cache.py
│   ├── test_ci_gates.py
│   ├── test_context_packing.py
│   ├── test_document_ingestion.py
│   ├── test_e2e_integration.py
│   ├── test_embeddings.py
│   ├── test_eval_metrics.py
│   ├── test_gemini_embedding.py
│   ├── test_hybrid.py
│   ├── test_observability.py
│   ├── test_parent_child.py
│   ├── test_qdrant_store_integration.py
│   ├── test_qdrant_store_mock.py
│   ├── test_query_rewrite.py
│   ├── test_rag_cli.py
│   ├── test_rag_pipeline.py
│   ├── test_ragas_eval.py
│   ├── test_rerank.py
│   ├── test_service.py
│   ├── test_token_counter.py
│   └── test_vector_store.py
└── docs/
    ├── capabilities.md        # Maintained implementation inventory and gaps
    ├── capabilities.zh-CN.md  # Chinese capability inventory
    ├── tech-selection.md      # Technology selection notes
    ├── architecture.md        # System architecture and data-flow diagrams
    ├── retrieval-baselines.md # Frozen metrics and reproducibility controls
    ├── dev-log-crashing.md    # Core pitfall and incident debugging log
    └── learning_notes.zh-CN.md # Legacy learning-note index
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

To reproduce the CI dependency set, install `requirements-ci.txt`. Maintainers update direct pins in `requirements-ci.in`, compatibility pins in `requirements-ci.constraints`, and regenerate the universal lock with the command in its header.

Create a `.env` file:

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions

EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_DIMENSION=512

PDF_EXTRACT_TABLES=false
PDF_OCR_ENABLED=false
PDF_OCR_MIN_CHARS=1
PDF_OCR_DPI=200

VECTOR_STORE_PROVIDER=memory
VECTOR_STORE_COLLECTION=enterprise_kb

RERANK_ENABLED=false
RERANK_PROVIDER=deterministic
RERANK_FETCH_K=30
RERANK_TOP_N=5

HYBRID_ENABLED=false
HYBRID_DENSE_WEIGHT=0.2
HYBRID_SPARSE_WEIGHT=1.0
RRF_K=60
BM25_K1=1.5
BM25_B=0.75

PARENT_CHILD_ENABLED=false
PARENT_CHUNK_SIZE=1600
PARENT_CHUNK_OVERLAP=200
CHILD_CHUNK_SIZE=400
CHILD_CHUNK_OVERLAP=80

CONTEXT_PACKING_ENABLED=false
CONTEXT_DEDUP_ENABLED=false
CONTEXT_NEAR_DUP_ENABLED=false
CONTEXT_NEAR_DUP_THRESHOLD=0.9
CONTEXT_MAX_TOKENS=2048
TOKENIZER_ENCODING=cl100k_base

QUERY_REWRITE_ENABLED=false
QUERY_REWRITE_PROVIDER=deterministic
QUERY_REWRITE_FIXTURE_PATH=eval/fixtures/query_rewrites.jsonl
QUERY_REWRITE_NUM_QUERIES=3
QUERY_REWRITE_TEMPERATURE=0.1
QUERY_REWRITE_CACHE_ENABLED=true
QUERY_REWRITE_WEIGHT_ORIGINAL=1.0
QUERY_REWRITE_WEIGHT_VARIANT=0.7

REDIS_URL=redis://localhost:6379/0
CACHE_NAMESPACE=rag-cache
CACHE_ENABLED=false
CACHE_EMBEDDING_ENABLED=true
CACHE_RETRIEVAL_ENABLED=true
CACHE_ANSWER_ENABLED=true
CACHE_EMBEDDING_TTL=604800
CACHE_RETRIEVAL_TTL=900
CACHE_ANSWER_TTL=300
CACHE_TIMEOUT=0.25

ACL_ENABLED=false
ACL_METADATA_KEY=acl
ACL_DEFAULT_DENY=true
ACL_PRINCIPAL_HEADER=X-Principal
ACL_ALLOW_BODY_PRINCIPAL=false
ACL_INGEST_BINDINGS_ENABLED=false
METADATA_DB_URL=mysql://user:password@localhost:3306/rag_metadata

OBSERVABILITY_ENABLED=false
AUDIT_ENABLED=false
METRICS_ENABLED=false
AUDIT_LOG_PATH=logs/audit.jsonl
AUDIT_LOG_QUERY_TEXT=false
AUDIT_PRINCIPAL_MODE=hash
AUDIT_HASH_SALT=replace-with-a-long-random-secret
METRICS_NAMESPACE=rag
METRICS_PATH=/metrics
READINESS_TIMEOUT=2.0
```

## Usage

Run the basic CLI chat flow:

```bash
python main.py
```

Run the RAG CLI with one question:

```bash
python rag_cli.py tests/fixtures --question "What is this project?"
```

Run the RAG CLI in interactive mode:

```bash
python rag_cli.py tests/fixtures
```

Build an index once and query an existing vector-store collection:

```bash
python rag_cli.py ingest knowledge_base \
  --embedding-provider openai \
  --embedding-dimension 512 \
  --vector-store qdrant

python rag_cli.py query "What does the knowledge base say about deployment?" \
  --embedding-provider openai \
  --embedding-dimension 512 \
  --vector-store qdrant
```

The `memory` vector store is process-local and is useful for separation tests only; use Qdrant for cross-process persistence.

**Index migration (August 2026):** exact source spans, searchable Markdown code, and normalized source path separators change chunk content or record IDs together. Existing persistent Qdrant collections must be recreated once and then re-ingested; set `VECTOR_STORE_RECREATE=true` for that ingest only, then return it to `false`. A normal upsert does not remove records created under the old IDs.

Run the HTTP service locally:

```bash
uvicorn service.app:app --host 127.0.0.1 --port 8000
```

The service exposes `POST /ingest`, `POST /query`, `POST /query/stream` for SSE streaming, `GET /cache/stats`, `GET /health`, and `GET /ready`. When metrics are enabled it also exposes `GET /metrics`. Authentication is intentionally not implemented yet, so do not expose it publicly without a trusted gateway.

Redis caching is disabled by default. To enable it, run Redis, set `CACHE_ENABLED=true`, and configure `REDIS_URL`. `/query/stream` replays a cached L3 answer when present and marks the SSE payload with `cached=true`; otherwise it keeps the live provider stream and backfills L3 after completion.

ACL/RBAC filtering is disabled by default. To enable it, set `ACL_ENABLED=true`, configure the metadata database, and place the service behind an authentication gateway that writes the trusted principal header named by `ACL_PRINCIPAL_HEADER` (`X-Principal` by default). Body-provided principals are rejected unless `ACL_ALLOW_BODY_PRINCIPAL=true` is explicitly enabled for local testing. MySQL ACL bindings can be denormalized into vector payloads during ingest with `ACL_INGEST_BINDINGS_ENABLED=true`; Qdrant payloads remain execution snapshots, so binding changes require re-ingest or re-sync before they affect retrieval.

Observability is disabled by default. Set `OBSERVABILITY_ENABLED=true` and enable at least one of `AUDIT_ENABLED` or `METRICS_ENABLED`; audit hashing additionally requires a deployment secret in `AUDIT_HASH_SALT`. Runtime audit/metric failures fail open, while invalid enabled configuration fails during application creation. Query text is hashed unless `AUDIT_LOG_QUERY_TEXT=true` is explicitly selected. Token metrics prefer provider-reported counters; when upstream usage is unavailable they are labeled `estimated`, and provider watermarks prevent concurrent requests or cached raw responses from double-counting cumulative usage. Embedding estimates only cover query text visible at the service boundary, so provider-side rewrite expansion remains an estimate rather than billing truth.

Run the full local stack with Docker Compose:

```bash
cp .env.compose.example .env
# Fill DEEPSEEK_API_KEY, EMBEDDING_API_KEY, and AUDIT_HASH_SALT before starting.
docker compose up --build
docker compose ps
```

The compose stack starts Qdrant, Redis, MySQL, and the FastAPI app with service-name networking (`qdrant`, `redis`, `mysql`). The app container runs an idempotent startup initializer: it creates the MySQL ACL schema, seeds demo principals and ACL bindings, and ingests `eval/fixtures/knowledge_base/*.md` into Qdrant with trusted `--acl` metadata. This is a full-stack deployment smoke path, not a zero-key chatbot: ingestion needs an embedding API key, and `/query` needs a valid DeepSeek key.

After the four healthchecks are green, try:

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/ready

curl -s http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-Principal: alice" \
  -d '{"question":"What finance approval rules are in the knowledge base?","top_k":3}'

curl -s http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-Principal: bob" \
  -d '{"question":"What finance approval rules are in the knowledge base?","top_k":3}'

curl -s http://localhost:8000/metrics
```

`alice` is seeded with finance access; `bob` is seeded with legal access. Re-running `docker compose up` should repeat the initializer without duplicating ACL rows or vector records. Use `/health` as a liveness probe and `/ready` as a readiness probe: Kubernetes should keep `livenessProbe` on `/health` so transient backend outages do not restart the process, and route traffic with `readinessProbe` on `/ready` so Qdrant, Redis, or MySQL outages fail closed with HTTP 503.

Useful RAG CLI options:

```bash
python rag_cli.py knowledge_base \
  --question "What does the knowledge base say about deployment?" \
  --embedding-provider openai \
  --embedding-dimension 512 \
  --vector-store qdrant \
  --top-k 5 \
  --hybrid \
  --hybrid-fetch-k 30 \
  --hybrid-dense-weight 0.2 \
  --hybrid-sparse-weight 1.0 \
  --parent-child \
  --parent-chunk-size 1600 \
  --child-chunk-size 400 \
  --context-packing \
  --context-dedup \
  --context-max-tokens 2048 \
  --multi-query \
  --query-rewrite-provider deterministic \
  --rerank-provider deterministic \
  --rerank-fetch-k 30 \
  --chunk-size 800 \
  --chunk-overlap 120 \
  --principal alice \
  --metadata-filter "{\"file_type\":\"markdown\"}"
```

Load a document through the unified entry point:

```python
from document_loader import load_document

document = load_document("example.md", clean=True)
```

Load and split a document:

```python
from document_loader import load_and_split_document

chunks = load_and_split_document(
    "example.md",
    clean=True,
    chunk_size=800,
    chunk_overlap=120,
)
```

Load all supported files from a directory:

```python
from document_loader import load_documents

documents = load_documents("knowledge_base", recursive=True, clean=True)
```

Embed chunks with the OpenAI embedding provider:

```python
from document_loader import load_and_split_document
from embeddings import OpenAIEmbeddingProvider

chunks = load_and_split_document("example.md", clean=True)
provider = OpenAIEmbeddingProvider(dimensions=512)
embedded_chunks = provider.embed_documents(chunks)
```

Store and search embedded chunks in memory:

```python
from document_loader import load_and_split_document
from embeddings import OpenAIEmbeddingProvider
from vector_store import InMemoryVectorStore

chunks = load_and_split_document("example.md", clean=True)
provider = OpenAIEmbeddingProvider(dimensions=512)
embedded_chunks = provider.embed_documents(chunks)

store = InMemoryVectorStore(dimension=provider.dimension)
store.add_documents(embedded_chunks)

query_embedding = provider.embed_text("What does this document say about RAG?")
results = store.similarity_search(query_embedding, top_k=3)
```

Use Qdrant as the vector store:

```python
from document_loader import load_and_split_document
from embeddings import OpenAIEmbeddingProvider
from vector_store import QdrantVectorStore

chunks = load_and_split_document("example.md", clean=True)
provider = OpenAIEmbeddingProvider(dimensions=512)
embedded_chunks = provider.embed_documents(chunks)

store = QdrantVectorStore(
    collection_name="enterprise_kb",
    dimension=provider.dimension,
    url="http://localhost:6333",
)
store.add_documents(embedded_chunks)

query_embedding = provider.embed_text("What does this document say about RAG?")
results = store.similarity_search(query_embedding, top_k=3, metadata_filter={"file_type": "markdown"})
```

Run the minimal RAG pipeline:

```python
from api_client import APIClient
from config import Config
from document_loader import load_and_split_document
from embeddings import OpenAIEmbeddingProvider
from rag import RAGPipeline
from vector_store import InMemoryVectorStore

chunks = load_and_split_document("example.md", clean=True)
provider = OpenAIEmbeddingProvider(dimensions=512)
embedded_chunks = provider.embed_documents(chunks)

store = InMemoryVectorStore(dimension=provider.dimension)
store.add_documents(embedded_chunks)

client = APIClient(Config.API_KEY, Config.API_URL)
pipeline = RAGPipeline(provider, store, client)

response = pipeline.answer("What does this document say about RAG?")
print(response.answer)
print(response.sources)
```

Run the pipeline with a deterministic offline reranker:

```python
from rerank import DeterministicReranker

reranker = DeterministicReranker(top_n=5, fetch_k=30)
pipeline = RAGPipeline(provider, store, client, top_k=5, reranker=reranker, fetch_k=30)
```

Run the pipeline with hybrid dense + BM25 retrieval:

```python
from hybrid import BM25Retriever, RRFConfig, ReciprocalRankFusion

bm25 = BM25Retriever(vector_records)
rrf = ReciprocalRankFusion(
    RRFConfig(k=60, weights={"dense": 0.2, "sparse": 1.0})
)
pipeline = RAGPipeline(
    provider,
    store,
    client,
    top_k=5,
    fetch_k=30,
    bm25_retriever=bm25,
    rrf=rrf,
)
```

Run with context packing:

```python
pipeline = RAGPipeline(
    provider,
    store,
    client,
    top_k=5,
    context_packing_enabled=True,
    context_dedup_enabled=True,
    context_max_tokens=2048,
    tokenizer_encoding="cl100k_base",
)
```

Run with parent-child context expansion:

```python
from document_loader import load_and_split_documents_hierarchical
from embeddings import OpenAIEmbeddingProvider
from parent_store import InMemoryParentStore
from rag import RAGPipeline
from vector_store import InMemoryVectorStore

split = load_and_split_documents_hierarchical(
    "knowledge_base",
    parent_chunk_size=1600,
    parent_chunk_overlap=200,
    child_chunk_size=400,
    child_chunk_overlap=80,
)

parent_store = InMemoryParentStore()
parent_store.add_parents(split.parents)

provider = OpenAIEmbeddingProvider(dimensions=512)
embedded_children = provider.embed_documents(split.children)

store = InMemoryVectorStore(dimension=provider.dimension)
store.add_documents(embedded_children)

pipeline = RAGPipeline(
    provider,
    store,
    client,
    parent_store=parent_store,
)
```

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

Run the deterministic retrieval baseline:

```bash
python -m eval.run
```

Run the deterministic rerank evaluation:

```bash
python -m eval.run --rerank-provider deterministic --rerank-fetch-k 30 --rerank-top-n 10
```

Run the hybrid comparison with reranker disabled:

```bash
python -m eval.run --compare-hybrid --hybrid-fetch-k 30
```

Run the parent-child child-corpus retrieval evaluation:

```bash
python -m eval.run --parent-child
```

Run deterministic multi-query retrieval evaluation:

```bash
python -m eval.run --multi-query
```

Force evaluation to bypass Redis cache wrappers:

```bash
python -m eval.run --no-cache
```

The evaluator only depends on a `retrieve(question, top_k)` callable. It does not call the chat model, and the default baseline uses `HashEmbeddingProvider` for offline reproducibility.

The frozen fixture contains five synthetic Markdown documents of 765–887 bytes and 20 queries: 16 positive and 4 negative. MRR and recall claims below apply to the positive queries. At top-k 3 the four negative queries still return non-empty results (`negative_false_recall_rate = 1.000000`), so this system does not yet demonstrate abstention.

Current rerank baseline, using `HashEmbeddingProvider` plus `DeterministicReranker`, improves MRR@3 from `0.677083` to `1.000000` and long_tail MRR@3 from `0.566667` to `1.000000` while keeping recall@5 at `1.000000`. The rerank profile is asserted by `ci/retrieval_baseline_gate.py`; each run writes local reports and CI uploads them as workflow artifacts.

Current hybrid retrieval comparison, with reranker disabled and `HashEmbeddingProvider`, reports dense-only MRR@3 `0.677083`, bm25-only MRR@3 `1.000000`, and fused MRR@3 `1.000000`. The fused route also lifts exact_name and long_tail MRR@3 to `1.000000` while preserving deterministic metric output. The hybrid profile is asserted by the same gate, with reports generated per run rather than committed.

Current parent-child retrieval mode uses child chunks for retrieval and parent chunks only after final top-k selection. Retrieval reports default to child chunks with parent expansion disabled, so any metric movement is attributed to chunk-granularity changes rather than parent expansion. Under exact source-slice chunk semantics its frozen MRR@3 is `0.614583`; the former `0.625000` value depended on fallback slices with incorrect source spans and has been retired. Parent expansion quality is a generation-side concern and is deferred to the later Ragas evaluation.

Current context packing is a generation-input assembly step. It does not enter `retrieve()` and must not be reported as MRR/recall improvement. With all context flags off, the legacy character-based `_build_context` path is preserved. When enabled, packing includes whole blocks or skips them, renumbers citations contiguously, folds exact duplicates, optionally folds near duplicates behind a flag, and uses `TokenCounter` for token budgets. tiktoken is optional; when it is unavailable, the system falls back to a deterministic heuristic counter. For Chinese text, true token counting may create more but legal smaller batches; the benefit is correctness and zero over-limit requests, not claiming fewer batches. Generation quality and coherence remain deferred to Ragas-style answer-quality evaluation.

Current multi-query retrieval is a retrieval-side change, so hit_rate/MRR/recall are valid measurement surfaces. The deterministic fixture keeps the original query as `q0`, adds frozen variants for paraphrase and long-tail cases, fuses per-query results with the existing RRF implementation, and then reuses the existing rerank, parent expansion, and context packing stages. In the offline HashEmbeddingProvider baseline, `python -m eval.run --multi-query` improves paraphrase recall@3 from `0.800000` to `1.000000`, keeps long_tail recall@3 at `0.900000`, and improves long_tail MRR@3 from `0.566667` to `0.900000`. The four negative queries have no fixture rewrite, so multi-query is fully bypassed for them and their results are byte-identical to the single-path baseline; multi-query-on-negative risk is not covered offline and is deferred to the abstain/threshold work in the roadmap. These deterministic fixture gains are a controlled demonstration that RRF fusion plumbing delivers the targeted paraphrase/long_tail gains when fed known-good rewrites. Production LLM rewrites may land above or below this, and query drift can fall below single-path. This is not a production floor.

Current Redis caching is an outer wrapper, not a second retrieval pipeline. With `CACHE_ENABLED=false`, providers and pipelines are returned unwrapped. With cache enabled, L1 wraps `EmbeddingProvider.embed_text`, L2 wraps `retrieve`, and L3 wraps `answer`; `rag/pipeline.py` remains unchanged. L1 does not include `corpus_version` because text embeddings are a model+text function. L2/L3 include the Redis-persisted corpus version so `/ingest` makes stale retrieval and answer entries unreachable across process restarts and multiple service replicas. Negative answers are cached like any other exact query result but remain bounded by TTL and corpus version. The default unit tests use a FakeRedis substitute; true `redis.asyncio` coverage is gated behind `REDIS_URL` and must include both live Redis commands and cross-event-loop calls.
The live Redis path has passed the production-shape regression: repeated synchronous and service-thread calls reuse a store-owned Redis loop, hit L1/L2/L3 on the second call, and keep Redis error counts at zero.

Current ACL/RBAC support is fail-closed pre-filtering, not post-filtering. The effective ACL filter is built on the server side from a trusted principal, then ANDed with any client metadata filter so clients can narrow results but cannot widen access. Records are authorized when `record.acl` intersects the resolved allowed ACL subjects; records with missing or empty ACL are restricted for ordinary users. The same filter is applied before dense scoring, BM25 sparse scoring, each multi-query variant, and cache key construction. MySQL is the metadata source for principal membership and optional document/chunk ACL bindings, while Qdrant payloads are execution snapshots used for fast retrieval. Updating bindings in MySQL requires re-ingest or re-sync before the vector-store payload changes. The FastAPI service does not validate JWTs or sessions; production deployments must put it behind a trusted authentication gateway that owns the principal header.

Current observability is an outer service adapter, not part of retrieval or authorization decisions. Structured audit records contain a generated request ID, opaque principal/query identifiers, bounded request metadata, and stable record IDs, but never source content or ACL subjects. Prometheus labels are restricted to route, outcome, retrieval mode, token source, and cache layer; request IDs, principals, query text, and source IDs remain out of metric labels. Empty retrieval, ACL denial, and server failures expose stable alert codes and optional hooks. All observer failures are fail-open, which deliberately differs from ACL's fail-closed security boundary.

## Generation Quality Evaluation

Generation quality uses four Ragas dimensions: Faithfulness, Answer Relevance, Context Precision, and Context Recall. It is deliberately isolated from deterministic retrieval evaluation: `python -m eval.run` remains offline and byte-reproducible, while Ragas writes JSON only under `eval/reports/ragas/`.

The live judge uses the project's DeepSeek OpenAI-compatible API configuration. Answer Relevance embeds text through a selectable provider (`RAGAS_EMBEDDING_PROVIDER`). The `bge` provider runs a local fastembed/ONNX model (`BAAI/bge-small-en-v1.5`, 384-dim, no torch) with no API key, cost, or rate limit, so the live gate can run fully offline. The `gemini` provider calls Gemini `gemini-embedding-001` through the native `batchEmbedContents` API with `RETRIEVAL_QUERY`, bounded batches, dimension validation, and asynchronous retry. Both paths L2-normalize vectors, and changing the provider, model, or dimension invalidates the recorded baseline. This embedding scores only the offline answer-relevance gate; product retrieval uses its own embedding and is unaffected.

The quality gate has two tracks:

- `replay` reads a committed live verdict fixture, performs no network calls, needs no API key, and is suitable for every-push CI.
- `live` calls the existing `RAGPipeline.answer()` path once per golden question, fixes the generated answer and contexts, then repeats the external judge at least twice to measure spread. It requires both `RAGAS_ENABLED=true` and `RUN_RAGAS_EVAL=true`.

Run the offline gate after a live fixture has been recorded:

```bash
python -m eval.ragas_run replay
```

Record or refresh a live baseline explicitly:

```bash
python -m eval.ragas_run live --profile baseline --repetitions 3 --refresh-fixture
```

Record feature profiles separately and compare their means together with judge spread:

```bash
python -m eval.ragas_run live --profile context_packing --repetitions 3 --fixture eval/fixtures/ragas-context-packing.jsonl --refresh-fixture
python -m eval.ragas_run compare --before eval/fixtures/ragas_verdicts.jsonl --after eval/fixtures/ragas-context-packing.jsonl --before-label baseline --after-label context-packing
```

The gate covers only answer-consuming dimensions: Faithfulness, Answer Relevance, and negative abstention/fabrication are gated, while Context Precision/Recall are reported-only and still emitted in full because they do not consume the generated answer. Generation prompts are calibrated per question shape toward an "answer plus minimal supporting phrase" middle ground — a bare answer leaves the judge's reverse-questions under-informed, while over-scaffolding drifts them away from the original question — and together with versioned statement extraction this brings all 16 positive samples to passing Faithfulness and Answer Relevance medians with every negative abstaining. That passing capture has been promoted to the committed verdict fixture (schema `ragas-verdicts-v5`, pinning the generation/extraction prompt versions, judge model, embedding, and gating scope), and `python -m eval.ragas_run replay` verifies the offline gate passes.

Live evaluation is a controlled data-egress path. Positive samples send the question, generated answer, retrieved context text, and hand-written ground truth to the configured judge; with the `gemini` provider, Answer Relevance also sends text to the embedding endpoint, while the local `bge` provider performs no network calls. Fixtures store hashes and verdicts rather than raw evaluation text. Scores on this small hand-labeled set are noisy judge estimates, not deterministic facts or a production quality floor. `temperature=0` does not remove provider or model variance.

## License

This repository is licensed under the [MIT License](LICENSE).

## Development Conventions

- Code identifiers use English.
- Class and function docstrings are written in English.
- Key implementation comments are written in English.
- `README.md` is the primary English project document.
- `README.zh-CN.md` is the Chinese project document.
- Chinese learning notes are kept separately under `docs/`.

## Roadmap

1. Add more ingestion edge-case fixtures.
2. Add score thresholding or abstain logic for negative queries.
3. Add external alert routing for the existing observability hooks.
4. Add production deployment checks around Qdrant version compatibility.
5. Add built-in authentication or gateway integration checks for production deployments.
