# Minimal Demo

Two ways to see the engine run end to end. Both are zero paid keys, both are
reproducible, and both are honest about what they prove.

- **Path A — offline retrieval eval** (~30 seconds, no Docker): runs the
  deterministic retrieval funnel over the golden set and prints the metrics.
- **Path B — full-stack HTTP demo** (~3 minutes, Docker): boots the FastAPI
  service with Qdrant, Redis, and MySQL, then walks `/health`, `/ready`,
  `/query` under two principals, and `/metrics`.

Both paths use the offline hash embedding provider, so retrieval ranking and
the demo answer text are **not semantically meaningful**. What these demos
prove is that the wiring works — the retrieval funnel, ACL pre-filtering,
citations, cache counters, fail-closed access, and health probes — not answer
quality. For semantic retrieval and real generated answers, set a real
embedding key and DeepSeek key and follow the paths in the main
[README](../README.md).

## Path A — offline retrieval eval

No Docker, no API keys, no external services.

```bash
pip install -r requirements.txt
python -m eval.run
```

This embeds `eval/fixtures/knowledge_base/*.md` with the deterministic
`HashEmbeddingProvider`, retrieves against the JSONL golden set, and prints a
Markdown report. The overall block shows exactly the frozen baseline the CI
retrieval gate asserts on every push:

```text
## Overall Metrics
| k | hit_rate | MRR      | recall   | negative_empty_rate | negative_false_recall_rate |
| 3 | 0.937500 | 0.677083 | 0.906250 | 0.000000            | 1.000000                   |
| 5 | 1.000000 | 0.692708 | 1.000000 | 0.000000            | 1.000000                   |
```

The evaluator never calls the chat model — it only needs a
`retrieve(question, top_k)` callable — so this run is fully offline and
byte-reproducible. `MRR@3 = 0.677083` for the dense baseline is the same number
the frozen retrieval gate checks to six decimals. Try the retrieval variants:

```bash
python -m eval.run --compare-hybrid --hybrid-fetch-k 30   # dense vs bm25 vs fused
python -m eval.run --rerank-provider deterministic         # dense + rerank
python -m eval.run --multi-query                           # query rewrite fusion
```

## Path B — full-stack HTTP demo

Needs Docker Desktop. Still zero paid keys: a local deterministic chat stub
stands in for DeepSeek, and the hash embedding provider stands in for a paid
embedding model.

### 1. Start the local chat stub

In one terminal, start the deterministic OpenAI-compatible stub and leave it
running:

```bash
python examples/demo_chat_stub.py
# Deterministic chat stub listening on http://0.0.0.0:18080 (Ctrl+C to stop)
```

The app container reaches it at `host.docker.internal:18080`. If the stub is
not running, `/query` fails with `upstream_network_error` rather than falling
back to a paid provider — that is the intended zero-paid guarantee, not a bug.

### 2. Bring up the stack

In a second terminal, the base compose file requires `AUDIT_HASH_SALT`, so
export a throwaway value before starting (the demo override also sets one for
the app container, but compose interpolates the base file first):

```bash
# bash
export AUDIT_HASH_SALT=demo-local-audit-salt
docker compose -f docker-compose.yml -f examples/docker-compose.demo.yml up -d --build
```

```powershell
# PowerShell
$env:AUDIT_HASH_SALT = "demo-local-audit-salt"
docker compose -f docker-compose.yml -f examples/docker-compose.demo.yml up -d --build
```

The demo override ([`examples/docker-compose.demo.yml`](docker-compose.demo.yml))
swaps the paid providers for the hash embedding and the local stub, and unmaps
the Qdrant/Redis/MySQL host ports so the demo cannot collide with instances you
may already run on `6333`, `6379`, or `3306`. The app still listens on
`localhost:8000`.

Wait until all four containers report healthy:

```bash
docker compose -f docker-compose.yml -f examples/docker-compose.demo.yml ps
```

The app's startup initializer seeds demo principals (`alice` with finance
access, `bob` with legal access) and ingests the knowledge base with trusted
ACL metadata.

### 3. Walk the endpoints

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/ready

# alice has finance access
curl -s http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-Principal: alice" \
  -d '{"question":"What are the finance approval rules?","top_k":3}'

# bob has legal access — same question, ACL pre-filter returns different sources
curl -s http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-Principal: bob" \
  -d '{"question":"What are the finance approval rules?","top_k":3}'

# no principal — fails closed with HTTP 403
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"anything","top_k":3}'

curl -s http://localhost:8000/metrics | grep rag_cache_operations_total
```

What to look for:

- **Citations**: each answer carries bracketed `[1]` citations and a `sources`
  list with the originating document path.
- **ACL pre-filtering**: `alice` and `bob` get different `sources` for the same
  question because the effective ACL filter is applied before scoring, not
  after. `alice` sees `role:finance` documents; `bob` sees `role:legal` ones.
- **Fail-closed access**: a request with no principal returns `403` with
  `acl_forbidden`, never a partial answer.
- **Cache counters**: `/metrics` exposes real `rag_cache_operations_total`
  values with bounded labels; the second identical query embedding is an L1
  hit.
- **Readiness**: `/ready` names each required backend and its latency, and
  reports the missing generation key as a non-blocking dependency so the
  keyless stack still boots.

### 4. Tear down

```bash
docker compose -f docker-compose.yml -f examples/docker-compose.demo.yml down -v
```

Then stop the chat stub with Ctrl+C in the first terminal.
