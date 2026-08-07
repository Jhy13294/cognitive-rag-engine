# Capability Inventory

This inventory carries the detailed implementation status that was formerly embedded in the README. It describes shipped repository behavior, not a production-readiness certification.

## Ingestion and representation

- A shared `Document` model and loader interface cover TXT, Markdown, PDF, and Word `.docx` inputs through one loading entry point.
- TXT loading rejects obvious binary payloads and supports common fallback encodings.
- Markdown loading parses Front Matter, headings, links, tables, and fenced code. Code bodies remain searchable by default; explicit extract and drop compatibility modes remain available.
- PDF loading detects scanned pages and supports gated OCR and table extraction. Word loading preserves paragraphs and multiple tables.
- The conservative text cleaner and flat or parent-child splitters preserve metadata. Every emitted chunk is an exact contiguous slice of the processed document and records its half-open `start_char`/`end_char` span.
- Parent-child ingestion indexes child chunks and stores parents separately behind deterministic `parent_id` values.

## Retrieval and generation funnel

- Embedding providers share one abstraction. The repository includes a deterministic local hash provider and an OpenAI provider with dimensions, batching, retry, token counting, and usage logging.
- Vector stores share one abstraction. The in-memory store supports deterministic local evaluation; the Qdrant adapter supports batching, metadata filters, payload indexes, retries, and cross-process persistence.
- Dense search and BM25 sparse search emit compatible record identifiers. Reciprocal Rank Fusion combines them with configurable weights and deterministic tie-breaking.
- Reranking is pluggable: deterministic lexical reranking supports offline tests, Cohere is available as an external neural provider, and failures fall back to dense order.
- Multi-query retrieval supports the original query plus deterministic fixtures or chat-backed rewrites, followed by cross-query RRF and the existing rerank/parent-expansion stages.
- Parent expansion happens after final child selection and collapses sibling child hits by parent id.
- `ContextPacker` performs whole-block packing, exact and optional near-duplicate folding, contiguous citation numbering, and token-aware budgets on generation input.
- The RAG pipeline assembles retrieved context, invokes an OpenAI-compatible chat client, and returns the answer with source records.

## Service, cache, and access control

- The CLI separates ingest and query commands; Qdrant is the persistent cross-process path.
- The async FastAPI adapter exposes `/ingest`, `/query`, and SSE `/query/stream`, offloads the synchronous retrieval funnel, sanitizes errors, and invalidates caches after ingest.
- Chat and OpenAI embedding clients use `httpx.AsyncClient`, asynchronous retry backoff, and OpenAI-compatible streaming deltas.
- Redis-backed L1/L2/L3 decorators cache query embeddings, retrieval results, and full answers. Retrieval and answer keys include a persistent corpus version; Redis failures fail open as misses.
- The Redis store owns a background event loop so synchronous CLI calls and service-thread calls do not reuse a client across closed loops.
- ACL/RBAC is applied before candidate scoring. A MySQL resolver supplies principal membership and optional document/chunk bindings, while vector payloads carry the retrieval-time ACL snapshot.
- Query identity comes from a trusted upstream header by default. Missing identity or ACL resolution fails closed, and client metadata filters can narrow but cannot widen the server-built ACL filter.
- Dense search, Qdrant filtering, BM25, multi-query variants, and cache keys use the same effective ACL filter.

## Evaluation, observability, and deployment

- Deterministic JSONL evaluation reports hit rate, MRR, recall, capability slices, and negative-query behavior. CI gates exact unittest accounting and four retrieval profiles.
- A separate Ragas track evaluates generation quality. Live judging is explicitly gated; CI replays a frozen verdict fixture without keys.
- Pure-ASGI request IDs propagate through thread offload and return in `X-Request-ID` without buffering SSE.
- Sanitized JSON-line audit records hash principal and query identifiers and exclude source content, ACL subjects, and secrets. Audit, metrics, and alert-hook failures are fail-open.
- `/metrics` exposes bounded-label latency, token usage, retrieval score, empty-result, and cache counters.
- `/health` is a shallow liveness probe. `/ready` checks only enabled required backends and fails closed with a sanitized dependency name.
- Docker Compose provides Qdrant, Redis, MySQL, and the app with health checks and idempotent demo initialization. Runtime secrets remain environment-injected.

## Explicit gaps

Built-in authentication, JWT validation, and session management are not implemented. Production deployments must place the service behind a trusted authentication gateway. The frozen corpus is deliberately small, negative-query abstention is not implemented, and live provider quality and capacity remain deployment-specific concerns.
