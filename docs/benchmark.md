# Local Benchmarking

I use this benchmark to measure the local service stack without spending API quota. It is a closed-loop, single-machine benchmark for smoke-level latency and cache behavior, not a load-testing platform and not a production capacity claim.

## Scope

The runner lives in `bench/` and starts with:

```powershell
.\.venv\Scripts\python.exe -m bench.run --chat-stub-host 0.0.0.0
```

It sends HTTP traffic to `http://127.0.0.1:8000` by default and rejects non-localhost targets unless `--allow-non-localhost` is passed. Raw JSON and Markdown reports are written under `bench/reports/`, which is intentionally ignored by Git.

The benchmark covers four views:

- Cold unique questions: each measured request gets a nonce so L1/L2/L3 are cold. This measures the retrieval, ACL, context packing, and deterministic generation path. Because the deterministic query-rewrite fixture is keyed by exact question text, nonce questions exercise the rewrite guard path but do not claim fixture-backed rewrite variants.
- Hot fixed question: the same question is primed into L3, then measured. The report cross-checks Prometheus cache counters and expects the L3 hit delta to equal the measured request count with zero cache errors.
- Streaming TTFT: `/query/stream` is measured for a live miss and for L3 replay. The report records first SSE event latency, first token latency, total latency, and cached/replay flags.
- Closed-loop concurrency: concurrency levels `1`, `8`, and `32` report QPS, P50, P95, and error rate. P99 is omitted unless there are at least 1000 measured samples.

## Zero-Paid Setup

The app still expects an OpenAI-compatible chat endpoint for generation, so I point it at the benchmark's local deterministic chat stub. This proves the query path is live without calling DeepSeek.

Create a temporary override outside Git-tracked files, for example `bench/reports/docker-compose.bench.override.yml`. The `!reset` entries unpublish backend host ports, so the benchmark stack does not collide with any Qdrant, Redis, or MySQL instance already listening on this machine; the benchmark only talks to the app port:

```yaml
services:
  qdrant:
    ports: !reset []
  redis:
    ports: !reset []
  mysql:
    ports: !reset []
  app:
    environment:
      DEEPSEEK_API_KEY: bench-local
      DEEPSEEK_API_URL: http://host.docker.internal:18080/v1/chat/completions
      EMBEDDING_PROVIDER: hash
      EMBEDDING_DIMENSION: "128"
      VECTOR_STORE_COLLECTION: bench_kb
      CACHE_NAMESPACE: bench-cache
      AUDIT_HASH_SALT: bench-local-audit-salt
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

Then start the stack. The base compose file refuses to interpolate without `AUDIT_HASH_SALT`, so I export a throwaway value for the benchmark session (the override pins the container value anyway):

```powershell
$env:AUDIT_HASH_SALT = "bench-local-audit-salt"
docker compose -f docker-compose.yml -f bench/reports/docker-compose.bench.override.yml up -d --build
curl http://127.0.0.1:8000/ready
```

Run the benchmark while the stub is listening:

```powershell
.\.venv\Scripts\python.exe -m bench.run --chat-stub-host 0.0.0.0 --warmup 20 --samples 200 --concurrency 1 8 32
```

The runner performs a preflight query and aborts unless the local chat stub receives the request. That is the zero-paid guard: a successful run must show `paid_network_calls_expected: 0` and a positive `stub_request_delta`.

## Reading The Numbers

This is a closed-loop benchmark: each worker waits for a response before sending the next request. I do not treat it as open-loop capacity. Hot L3 replay numbers are useful for cache overhead and endpoint serialization, but they are not full-system RAG throughput. Cold unique-request numbers are the better local view of retrieval and generation plumbing, still with deterministic local generation rather than a real LLM.

QPS is wall-clock based, which makes it fragile: a single stalled request (I have observed one 30-second timeout from local Docker networking jitter) can halve a scenario's reported QPS while P50/P95 barely move. When comparing runs I read percentiles first and treat QPS as secondary; the raw JSON keeps per-scenario max latency and status codes so outliers stay visible.

The runner discards at least 20 warmup requests per scenario and records at least 200 measured samples. Performance assertions do not run in CI because timer-based gates would be noisy and machine-specific.

## Environment Notes

Each report captures OS, Python version, CPU count, hostname, target URL, warmup count, sample count, and readiness payload. For Windows Docker Desktop, I treat the virtualization layer as part of the local result and note it when comparing runs.

## Latest Local Runs

I keep raw benchmark reports out of Git because they are machine-specific. The two runs below were captured on the same machine with the keyless compose stack, hash embedding at 128 dimensions, and the deterministic local chat stub. Environment summary: Windows 11, Python 3.12.7, AMD64, 32 logical CPUs, 32 GB RAM, Docker compose target with Windows virtualization overhead included in the measured path, observability metrics enabled.

| Run | Samples | Cold 1-worker QPS | Cold P50/P95 | Hot 1-worker QPS | Hot P50/P95 | Hot L3 hit delta | Stream live first token | Stream replay first token |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |
| `bench-20260703T034716Z` | 200 | 24.22 | 40.84 / 43.75 ms | 57.05 | 17.44 / 18.84 ms | 200 / 200, errors 0 | 38.21 ms | 17.19 ms |
| `bench-20260703T035119Z` | 200 | 24.40 | 40.47 / 43.68 ms | 59.04 | 16.81 / 18.01 ms | 200 / 200, errors 0 | 39.19 ms | 16.51 ms |

Closed-loop concurrency stayed in the same order of magnitude across both runs:

| Run | Cold 8 QPS / P95 | Hot 8 QPS / P95 | Cold 32 QPS / P95 | Hot 32 QPS / P95 |
| --- | --- | --- | --- | --- |
| `bench-20260703T034716Z` | 40.86 / 239.51 ms | 201.60 / 46.78 ms | 38.56 / 1363.53 ms | 188.74 / 244.41 ms |
| `bench-20260703T035119Z` | 41.67 / 240.22 ms | 203.37 / 47.14 ms | 38.88 / 1348.47 ms | 186.57 / 240.58 ms |

Both reports recorded `paid_network_calls_expected: 0` and `stub_request_delta.requests: 1` during preflight, proving the measured service path reached the local deterministic chat stub rather than a paid provider.
