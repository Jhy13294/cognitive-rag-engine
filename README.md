# AI QA Assistant

An early-stage Python project for building an enterprise-grade RAG knowledge base.

The current codebase focuses on the foundation:

- DeepSeek-compatible chat API client
- Environment-based configuration
- Rotating file logging
- Document loading for TXT, Markdown, PDF, and Word files
- Conservative text cleaning
- Metadata-preserving text chunking for later vector indexing
- Production embedding and vector-store adapters
- Deterministic retrieval evaluation and optional reranking
- Shared lexical retrieval primitives
- Hybrid dense + BM25 retrieval with RRF fusion
- Parent-child chunking for child retrieval and parent context expansion
- Optional context packing with whole-block inclusion, deterministic deduplication, and token-aware budgets
- Query rewrite / multi-query retrieval with deterministic fixtures and cross-query RRF
- Async FastAPI service layer for ingest/query and SSE streaming
- Optional Redis cache wrappers for query embeddings, retrieval results, and generated answers
- Optional MySQL-backed ACL/RBAC pre-filtering with fail-closed access checks
- Optional structured audit logging and bounded-label Prometheus metrics with fail-open runtime emission

Built-in authentication is planned but not implemented yet.

## Project Status

Current milestone: evaluated retrieval funnel plus async HTTP service layer, optional Redis caching, ACL/RBAC pre-filtering, and fail-open observability.

Implemented:

- Basic LLM API call flow
- Reusable API client with retry handling
- Document model and loader interface
- TXT loader
- Markdown loader with Front Matter support
- PDF loader with optional table extraction
- Word `.docx` loader
- Text cleaner
- Text splitter for RAG chunks
- Unified document loading entry point
- Sample fixtures and ingestion tests
- Embedding provider abstraction
- OpenAI production embedding provider with dimensions support, batching, retries, and usage logging
- Shared token counting abstraction with optional tiktoken support and deterministic offline fallback
- Deterministic local hash embedding provider for tests
- Vector store abstraction
- In-memory vector store for local retrieval tests
- Qdrant production vector store adapter with batching, metadata filters, payload indexes, and retry handling
- Vector-store factory and CLI provider selection
- Minimal RAG pipeline with retrieval, context assembly, chat generation, and sources
- Split ingest/query CLI commands with cross-process persistence through Qdrant
- Deterministic retrieval evaluation with a JSONL golden set, HashEmbeddingProvider baseline, and JSON/Markdown reports
- Reranker abstraction with deterministic offline reranker and Cohere neural reranker provider
- Optional rerank insertion in `RAGPipeline.retrieve`: dense fetch, rerank, top-k selection, and dense fallback on failure
- Shared `lexical/` core for tokenization, IDF, BM25, and deterministic lexical scoring
- BM25 sparse retriever that emits the same `VectorRecord.id` values as dense retrieval
- RRF fusion with configurable dense/sparse weights and stable tie-breaking
- Hybrid comparison evaluation for dense-only, bm25-only, and fused retrieval with reranker disabled
- Parent-child splitter that indexes child chunks and stores parent chunks separately
- In-memory ParentStore for deterministic parent lookup by `parent_id`
- Optional post-top-k parent expansion in `RAGPipeline.retrieve`, with sibling child collapse by parent id
- ContextPacker for build_prompt/answer paths only: whole-block packing, exact deduplication, optional near-duplicate deduplication, contiguous citations, and token budgets
- OpenAI embedding batch construction based on TokenCounter instead of the legacy `len/4` estimate
- QueryRewriter abstraction with deterministic fixture-backed rewrites and a chat-backed production provider
- Multi-query retrieval in `RAGPipeline.retrieve`: original query plus variants, per-variant retrieval, cross-query RRF, then the existing rerank/parent expansion/context packing stages
- Async FastAPI adapter with `/ingest`, `/query`, `/query/stream`, Pydantic models, threadpool offload for the synchronous retrieval funnel, sanitized exception mapping, and cache invalidation after ingest
- Async `httpx.AsyncClient` chat and OpenAI embedding providers with `asyncio.sleep` retry backoff and OpenAI-compatible streaming chat deltas
- Redis-backed cache decorators:
  - L1 caches query embeddings by embedding model, dimension, normalization flag, and normalized question.
  - L2 caches retrieved sources by persistent corpus version, exact query/options key, top-k, and metadata filter.
  - L3 caches full `RAGResponse` answers by the L2 key plus system prompt and chat generation factors.
  - The Redis store owns a dedicated background event loop for the `redis.asyncio` client, so synchronous CLI/evaluation calls and service threadpool calls do not reuse a client across closed event loops.
  - Redis failures fail open as cache misses.
  - `/ingest` clears the in-process pipeline cache and increments a Redis-persisted corpus version.
- ACL/RBAC pre-filtering:
  - MySQL metadata resolver for principal membership and document/chunk ACL bindings.
  - Ingest can denormalize trusted ACL subjects or MySQL `acl_binding` subjects into vector payload metadata.
  - FastAPI query endpoints read principal from a trusted upstream header by default and fail closed when identity or ACL resolution is missing.
  - Dense memory search, Qdrant payload filters, BM25 sparse retrieval, multi-query variants, and L2/L3 cache keys all use the same effective ACL filter.
- Observability:
  - Pure-ASGI request IDs are returned through `X-Request-ID` without buffering SSE responses.
  - One sanitized JSON-line audit record is emitted for successful, empty, and ACL-denied queries; principal and query values are keyed hashes by default, and source content/ACL/API keys are never recorded.
  - `GET /metrics` exposes request latency, reported or estimated token usage, top-k score histograms, empty retrievals, and true L1/L2/L3 cache counters with bounded labels.
  - Audit, metric, and alert-hook failures are contained fail-open and cannot turn a valid query into an HTTP failure.

Not implemented yet:

- Built-in authentication, JWT validation, or session management

## Advanced Learning Notes

Start here when you want to understand the engineering choices behind the project:

- **[Technology Selection](docs/tech-selection.md)**: why the project uses the current embedding providers, vector stores, retrieval stack, service protocol, access filters, caching, and observability design.
- **[Architecture](docs/architecture.md)**: system architecture, ingest/query data flow, retrieval funnel, access control, cache layout, observability, and evaluation boundaries.
- **[Crash / Pitfall Log](docs/dev-log-crashing.md)**: hard-earned debugging notes about metric pollution, fake streaming, async traps, cache invalidation, access boundaries, audit leakage, and token accounting.

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
├── requirements.txt           # Python dependencies
├── document_loader/
│   ├── base.py                # Document model and loader interface
│   ├── chunking.py            # Text chunking
│   ├── loader.py              # Unified loading entry point
│   ├── md_loader.py           # Markdown loader
│   ├── pdf_loader.py          # PDF loader
│   ├── txt_loader.py          # TXT loader
│   └── word_loader.py         # Word loader
├── embeddings/
│   ├── base.py                # Embedding interface and vector utilities
│   ├── openai_provider.py     # OpenAI production embedding provider
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
│   ├── fixtures/query_rewrites.jsonl # Deterministic query rewrite fixture
│   ├── baseline.py            # Deterministic HashEmbeddingProvider baseline
│   ├── metrics.py             # hit_rate, MRR, recall, negative metrics
│   ├── reporting.py           # JSON and Markdown report rendering
│   ├── run.py                 # python -m eval.run entry point
│   ├── fixtures/              # Evaluation knowledge-base fixtures
│   └── reports/               # Generated evaluation reports
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
│   ├── chat.py                # Chat-backed production query rewriter
│   └── factory.py             # Query rewriter factory
├── rag/
│   ├── context_packing.py     # Post-retrieval context packing
│   └── pipeline.py            # Minimal RAG pipeline
├── text_cleaner/
│   └── cleaner.py             # Text cleaning
├── tests/
│   ├── fixtures/              # Sample TXT and Markdown fixtures
│   ├── test_document_ingestion.py
│   ├── test_api_client.py
│   ├── test_embeddings.py
│   ├── test_eval_metrics.py
│   ├── test_hybrid.py
│   ├── test_parent_child.py
│   ├── test_context_packing.py
│   ├── test_token_counter.py
│   ├── test_rerank.py
│   ├── test_acl.py
│   ├── test_acl_mysql_integration.py
│   ├── test_qdrant_store_mock.py
│   ├── test_qdrant_store_integration.py
│   ├── test_vector_store.py
│   ├── test_rag_pipeline.py
│   ├── test_service.py
│   ├── test_observability.py
│   └── test_rag_cli.py
└── docs/
    ├── tech-selection.md      # Technology selection notes
    ├── architecture.md        # System architecture and data-flow diagrams
    ├── dev-log-crashing.md    # Core pitfall and incident debugging log
    └── learning_notes.zh-CN.md # Legacy learning-note index
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions

EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_DIMENSION=512

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

Run the HTTP service locally:

```bash
uvicorn service.app:app --host 127.0.0.1 --port 8000
```

The service exposes `POST /ingest`, `POST /query`, `POST /query/stream` for SSE streaming, `GET /cache/stats`, and `GET /health`. When metrics are enabled it also exposes `GET /metrics`. Authentication is intentionally not implemented yet, so do not expose it publicly without a trusted gateway.

Redis caching is disabled by default. To enable it, run Redis, set `CACHE_ENABLED=true`, and configure `REDIS_URL`. `/query/stream` replays a cached L3 answer when present and marks the SSE payload with `cached=true`; otherwise it keeps the live provider stream and backfills L3 after completion.

ACL/RBAC filtering is disabled by default. To enable it, set `ACL_ENABLED=true`, configure the metadata database, and place the service behind an authentication gateway that writes the trusted principal header named by `ACL_PRINCIPAL_HEADER` (`X-Principal` by default). Body-provided principals are rejected unless `ACL_ALLOW_BODY_PRINCIPAL=true` is explicitly enabled for local testing. MySQL ACL bindings can be denormalized into vector payloads during ingest with `ACL_INGEST_BINDINGS_ENABLED=true`; Qdrant payloads remain execution snapshots, so binding changes require re-ingest or re-sync before they affect retrieval.

Observability is disabled by default. Set `OBSERVABILITY_ENABLED=true` and enable at least one of `AUDIT_ENABLED` or `METRICS_ENABLED`; audit hashing additionally requires a deployment secret in `AUDIT_HASH_SALT`. Runtime audit/metric failures fail open, while invalid enabled configuration fails during application creation. Query text is hashed unless `AUDIT_LOG_QUERY_TEXT=true` is explicitly selected. Token metrics prefer provider-reported counters; when upstream usage is unavailable they are labeled `estimated`, and provider watermarks prevent concurrent requests or cached raw responses from double-counting cumulative usage. Embedding estimates only cover query text visible at the service boundary, so provider-side rewrite expansion remains an estimate rather than billing truth.

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

Embed chunks with the production embedding provider:

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

Current rerank baseline, using `HashEmbeddingProvider` plus `DeterministicReranker`, improves MRR@3 from `0.677083` to `1.000000` and long_tail MRR@3 from `0.566667` to `1.000000` while keeping recall@5 at `1.000000`. The latest rerank report is under `eval/reports/`.

Current hybrid retrieval comparison, with reranker disabled and `HashEmbeddingProvider`, reports dense-only MRR@3 `0.677083`, bm25-only MRR@3 `1.000000`, and fused MRR@3 `1.000000`. The fused route also lifts exact_name and long_tail MRR@3 to `1.000000` while preserving deterministic byte-stable reports. The latest hybrid reports are under `eval/reports/`.

Current parent-child retrieval mode uses child chunks for retrieval and parent chunks only after final top-k selection. Retrieval reports default to child chunks with parent expansion disabled, so any metric movement is attributed to chunk-granularity changes rather than parent expansion. Parent expansion quality is a generation-side concern and is deferred to the later Ragas evaluation.

Current context packing is a generation-input assembly step. It does not enter `retrieve()` and must not be reported as MRR/recall improvement. With all context flags off, the legacy character-based `_build_context` path is preserved. When enabled, packing includes whole blocks or skips them, renumbers citations contiguously, folds exact duplicates, optionally folds near duplicates behind a flag, and uses `TokenCounter` for token budgets. tiktoken is optional; when it is unavailable, the system falls back to a deterministic heuristic counter. For Chinese text, true token counting may create more but legal smaller batches; the benefit is correctness and zero over-limit requests, not claiming fewer batches. Generation quality and coherence remain deferred to Ragas-style answer-quality evaluation.

Current multi-query retrieval is a retrieval-side change, so hit_rate/MRR/recall are valid measurement surfaces. The deterministic fixture keeps the original query as `q0`, adds frozen variants for paraphrase and long-tail cases, fuses per-query results with the existing RRF implementation, and then reuses the existing rerank, parent expansion, and context packing stages. In the offline HashEmbeddingProvider baseline, `python -m eval.run --multi-query` improves paraphrase recall@3 from `0.800000` to `1.000000`, keeps long_tail recall@3 at `0.900000`, and improves long_tail MRR@3 from `0.566667` to `0.900000`. The four negative queries have no fixture rewrite, so multi-query is fully bypassed for them and their results are byte-identical to the single-path baseline; multi-query-on-negative risk is not covered offline and is deferred to the abstain/threshold work in the roadmap. These deterministic fixture gains are a controlled demonstration that RRF fusion plumbing delivers the targeted paraphrase/long_tail gains when fed known-good rewrites. Production LLM rewrites may land above or below this, and query drift can fall below single-path. This is not a production floor.

Current Redis caching is an outer wrapper, not a second retrieval pipeline. With `CACHE_ENABLED=false`, providers and pipelines are returned unwrapped. With cache enabled, L1 wraps `EmbeddingProvider.embed_text`, L2 wraps `retrieve`, and L3 wraps `answer`; `rag/pipeline.py` remains unchanged. L1 does not include `corpus_version` because text embeddings are a model+text function. L2/L3 include the Redis-persisted corpus version so `/ingest` makes stale retrieval and answer entries unreachable across process restarts and multiple service replicas. Negative answers are cached like any other exact query result but remain bounded by TTL and corpus version. The default unit tests use a FakeRedis substitute; true `redis.asyncio` coverage is gated behind `REDIS_URL` and must include both live Redis commands and cross-event-loop calls.
The live Redis path has passed the production-shape regression: repeated synchronous and service-thread calls reuse a store-owned Redis loop, hit L1/L2/L3 on the second call, and keep Redis error counts at zero.

Current ACL/RBAC support is fail-closed pre-filtering, not post-filtering. The effective ACL filter is built on the server side from a trusted principal, then ANDed with any client metadata filter so clients can narrow results but cannot widen access. Records are authorized when `record.acl` intersects the resolved allowed ACL subjects; records with missing or empty ACL are restricted for ordinary users. The same filter is applied before dense scoring, BM25 sparse scoring, each multi-query variant, and cache key construction. MySQL is the metadata source for principal membership and optional document/chunk ACL bindings, while Qdrant payloads are execution snapshots used for fast retrieval. Updating bindings in MySQL requires re-ingest or re-sync before the vector-store payload changes. The FastAPI service does not validate JWTs or sessions; production deployments must put it behind a trusted authentication gateway that owns the principal header.

Current observability is an outer service adapter, not part of retrieval or authorization decisions. Structured audit records contain a generated request ID, opaque principal/query identifiers, bounded request metadata, and stable record IDs, but never source content or ACL subjects. Prometheus labels are restricted to route, outcome, retrieval mode, token source, and cache layer; request IDs, principals, query text, and source IDs remain out of metric labels. Empty retrieval, ACL denial, and server failures expose stable alert codes and optional hooks. All observer failures are fail-open, which deliberately differs from ACL's fail-closed security boundary.

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
3. Add deployment health gates and external alert routing for the existing observability hooks.
4. Add production deployment checks around Qdrant version compatibility and health probes.
5. Add built-in authentication or gateway integration checks for production deployments.
