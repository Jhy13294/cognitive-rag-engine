from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import socket
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_OUTPUT_DIR = "bench/reports"
DEFAULT_HOT_QUESTION = "Which service routes telemetry from connected devices into regional ingestion clusters?"
DEFAULT_COLD_QUESTION = "Which service routes telemetry from connected devices into regional ingestion clusters?"
DEFAULT_STREAM_QUESTION = "Which service routes telemetry from connected devices into regional ingestion clusters?"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", ""}
MIN_WARMUP = 20
MIN_SAMPLES = 200


@dataclass
class RequestSample:
    """One measured HTTP request."""

    latency_ms: float
    status_code: int
    error: Optional[str] = None


@dataclass
class ScenarioResult:
    """Aggregated latency and throughput result for one scenario."""

    name: str
    concurrency: int
    requests: int
    errors: int
    elapsed_seconds: float
    qps: float
    latency_ms: Dict[str, Optional[float]]
    status_codes: Dict[str, int]
    notes: List[str] = field(default_factory=list)
    cache_delta: Optional[Dict[str, Any]] = None


@dataclass
class StreamResult:
    """Server-sent-event timing result."""

    name: str
    status_code: int
    first_event_ms: Optional[float]
    first_token_ms: Optional[float]
    total_ms: float
    event_names: List[str]
    cached: bool
    stream_replay: bool
    error: Optional[str] = None


class DeterministicChatStub:
    """Small OpenAI-compatible chat endpoint for zero-paid local benchmarks."""

    def __init__(self, host: str, port: int):
        """Create a stub server that can be started from the benchmark process."""
        self.host = host
        self.port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.request_count = 0
        self.stream_count = 0

    def start(self) -> None:
        """Start the HTTP server in a background thread."""
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
                outer._handle_post(self)

            def log_message(self, _format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the server if it was started."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def snapshot(self) -> Dict[str, int]:
        """Return request counters for zero-paid verification."""
        with self._lock:
            return {"requests": self.request_count, "streams": self.stream_count}

    def _handle_post(self, handler: BaseHTTPRequestHandler) -> None:
        """Handle one OpenAI-compatible chat completion request."""
        if handler.path.rstrip("/") not in {"/v1/chat/completions", "/chat/completions"}:
            handler.send_error(404, "Not found")
            return
        length = int(handler.headers.get("Content-Length", "0") or 0)
        raw_body = handler.rfile.read(length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            handler.send_error(400, "Invalid JSON")
            return

        messages = payload.get("messages") or []
        prompt = "\n".join(str(message.get("content", "")) for message in messages)
        answer = deterministic_answer(prompt)
        with self._lock:
            self.request_count += 1
            if payload.get("stream"):
                self.stream_count += 1

        if payload.get("stream"):
            self._write_stream(handler, answer)
            return
        self._write_json(handler, chat_response(answer, prompt))

    @staticmethod
    def _write_json(handler: BaseHTTPRequestHandler, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    @staticmethod
    def _write_stream(handler: BaseHTTPRequestHandler, answer: str) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        for token in split_answer(answer):
            event = {
                "choices": [
                    {
                        "delta": {"content": token},
                        "index": 0,
                        "finish_reason": None,
                    }
                ]
            }
            handler.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
            handler.wfile.flush()
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()


def deterministic_answer(prompt: str) -> str:
    """Return a deterministic answer based on retrieved context text."""
    if "No relevant context was retrieved." in prompt:
        return "The answer is not available in the knowledge base."
    if "Nimbus Gateway" in prompt:
        return "Nimbus Gateway routes IoT device telemetry into regional ingestion clusters [1]."
    if "Apollo Dashboard" in prompt:
        return "Apollo Dashboard helps account teams inspect renewal risk [1]."
    if "ORION-17" in prompt:
        return "The ORION-17 onboarding checklist is completed within the first three business days [1]."
    if "EXP-204" in prompt:
        return "EXP-204 allows distance-based mileage reimbursement without fuel receipts [1]."
    return "The answer is supported by the retrieved local benchmark context [1]."


def split_answer(answer: str) -> Iterable[str]:
    """Split a deterministic answer into small stream chunks."""
    words = answer.split(" ")
    if len(words) <= 2:
        return [answer]
    chunks = []
    for index in range(0, len(words), 3):
        chunks.append(" ".join(words[index : index + 3]) + (" " if index + 3 < len(words) else ""))
    return chunks


def chat_response(answer: str, prompt: str) -> Dict[str, Any]:
    """Build an OpenAI-compatible non-streaming response."""
    prompt_tokens = max(1, len(prompt.split()))
    completion_tokens = max(1, len(answer.split()))
    return {
        "id": f"bench-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "model": "deterministic-bench-chat",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a zero-paid local benchmark against the RAG HTTP service. "
            "The target defaults to localhost and the built-in chat stub must be hit."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Service base URL. Defaults to localhost.")
    parser.add_argument(
        "--allow-non-localhost",
        action="store_true",
        help="Allow a non-localhost benchmark target. This is off by default.",
    )
    parser.add_argument("--principal", default="product", help="Trusted X-Principal value for ACL-filtered queries.")
    parser.add_argument("--embedding-dimension", type=int, default=128, help="Hash embedding dimension used at ingest.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k sources requested from the service.")
    parser.add_argument("--warmup", type=int, default=MIN_WARMUP, help="Warmup requests to discard per scenario.")
    parser.add_argument("--samples", type=int, default=MIN_SAMPLES, help="Measured requests per scenario.")
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 8, 32],
        help="Closed-loop concurrency levels for gradient scenarios.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for JSON and Markdown reports.")
    parser.add_argument("--label", default="", help="Optional run label stored in the report.")
    parser.add_argument("--cold-question", default=DEFAULT_COLD_QUESTION, help="Base question for unique cold requests.")
    parser.add_argument("--hot-question", default=DEFAULT_HOT_QUESTION, help="Fixed question for hot L3-cache requests.")
    parser.add_argument("--stream-question", default=DEFAULT_STREAM_QUESTION, help="Question for streaming timing.")
    parser.add_argument(
        "--chat-stub",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start a local deterministic OpenAI-compatible chat stub.",
    )
    parser.add_argument("--chat-stub-host", default="127.0.0.1", help="Host for the local chat stub.")
    parser.add_argument("--chat-stub-port", type=int, default=18080, help="Port for the local chat stub.")
    parser.add_argument(
        "--require-stub-hit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort unless the preflight query hits the deterministic chat stub.",
    )
    return parser


async def run_benchmark(args: argparse.Namespace, stub: Optional[DeterministicChatStub]) -> Dict[str, Any]:
    """Run all benchmark scenarios and return a serializable report."""
    ensure_allowed_base_url(args.base_url, args.allow_non_localhost)
    ensure_sampling_contract(args.warmup, args.samples, args.concurrency)

    started_at = datetime.now(timezone.utc).isoformat()
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        timeout=args.timeout,
        trust_env=False,
    ) as client:
        health = await get_json(client, "/health")
        ready = await get_json(client, "/ready")
        await assert_metrics_available(client)

        zero_paid = await verify_zero_paid_path(client, args, stub)
        scenarios: List[ScenarioResult] = []
        streams: List[StreamResult] = []

        scenarios.append(
            await run_query_scenario(
                client,
                args,
                name="cold_unique_full_path",
                concurrency=1,
                question_factory=lambda i: f"{args.cold_question} Bench nonce {uuid.uuid4().hex}-{i}",
                multi_query=True,
                cache_check=False,
            )
        )
        scenarios.append(
            await run_hot_scenario(client, args, concurrency=1)
        )

        streams.append(await run_live_stream(client, args))
        streams.append(await run_replay_stream(client, args))

        for concurrency in args.concurrency:
            scenarios.append(
                await run_query_scenario(
                    client,
                    args,
                    name=f"cold_unique_concurrency_{concurrency}",
                    concurrency=concurrency,
                    question_factory=lambda i, level=concurrency: (
                        f"{args.cold_question} Closed-loop cold level {level} nonce {uuid.uuid4().hex}-{i}"
                    ),
                    multi_query=True,
                    cache_check=False,
                )
            )
            scenarios.append(await run_hot_scenario(client, args, concurrency=concurrency))

        finished_at = datetime.now(timezone.utc).isoformat()
        report = {
            "metadata": {
                "started_at": started_at,
                "finished_at": finished_at,
                "label": args.label,
                "base_url": args.base_url,
                "closed_loop": True,
                "warmup_discarded_per_scenario": args.warmup,
                "samples_per_scenario": args.samples,
                "p99_reported_only_when_samples_at_least": 1000,
                "environment": environment_snapshot(),
                "health": health,
                "ready": ready,
                "zero_paid_evidence": zero_paid,
            },
            "query_scenarios": [scenario_to_dict(item) for item in scenarios],
            "streaming": [stream_to_dict(item) for item in streams],
        }
        return report


async def verify_zero_paid_path(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    stub: Optional[DeterministicChatStub],
) -> Dict[str, Any]:
    """Prove that a live query reaches the local deterministic chat stub."""
    before = stub.snapshot() if stub is not None else {"requests": 0, "streams": 0}
    preflight_question = f"{args.hot_question} zero-paid preflight nonce {uuid.uuid4().hex}"
    response = await client.post(
        "/query",
        json=query_payload(preflight_question, args, multi_query=True),
        headers=principal_headers(args.principal),
    )
    body = response.text
    if response.status_code != 200:
        raise RuntimeError(f"Zero-paid preflight query failed with HTTP {response.status_code}: {body[:300]}")
    after = stub.snapshot() if stub is not None else {"requests": 0, "streams": 0}
    delta = {
        "requests": after["requests"] - before["requests"],
        "streams": after["streams"] - before["streams"],
    }
    if args.require_stub_hit and delta["requests"] <= 0:
        raise RuntimeError(
            "The preflight query did not hit the local deterministic chat stub. "
            "Point DEEPSEEK_API_URL at the stub before running this benchmark."
        )
    return {
        "local_chat_stub_enabled": stub is not None,
        "stub_request_delta": delta,
        "api_key_required": "a dummy local key is enough when the service points to the stub",
        "paid_network_calls_expected": 0,
    }


async def run_hot_scenario(client: httpx.AsyncClient, args: argparse.Namespace, concurrency: int) -> ScenarioResult:
    """Run a fixed-question hot L3-cache scenario and cross-check Prometheus counters."""
    question = args.hot_question
    await client.post("/query", json=query_payload(question, args, multi_query=True), headers=principal_headers(args.principal))
    await run_warmup(client, args, concurrency, lambda _i: question, multi_query=True)
    metrics_before = await metrics_snapshot(client)
    cache_before = await cache_stats(client)
    result = await run_query_scenario(
        client,
        args,
        name=f"hot_l3_replay_concurrency_{concurrency}",
        concurrency=concurrency,
        question_factory=lambda _i: question,
        multi_query=True,
        cache_check=False,
        warmup_already_done=True,
    )
    metrics_after = await metrics_snapshot(client)
    cache_after = await cache_stats(client)
    l3_hit_delta = counter_delta(metrics_before, metrics_after, "rag_cache_operations_total", {"cache_layer": "L3", "outcome": "hit"})
    cache_error_delta = numeric_delta(cache_before, cache_after, "errors")
    result.cache_delta = {
        "prometheus_l3_hit_delta": l3_hit_delta,
        "expected_sample_hits": args.samples,
        "cache_stats_error_delta": cache_error_delta,
    }
    if int(l3_hit_delta) != args.samples:
        result.notes.append("L3 hit delta did not equal the measured request count.")
    if cache_error_delta != 0:
        result.notes.append("Cache stats reported errors during the hot scenario.")
    return result


async def run_query_scenario(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    *,
    name: str,
    concurrency: int,
    question_factory,
    multi_query: bool,
    cache_check: bool,
    warmup_already_done: bool = False,
) -> ScenarioResult:
    """Run one closed-loop query scenario."""
    if not warmup_already_done:
        await run_warmup(client, args, concurrency, question_factory, multi_query=multi_query)
    counter = AtomicCounter()
    samples: List[RequestSample] = []
    started = time.perf_counter()

    async def worker() -> None:
        while True:
            index = await counter.next()
            if index >= args.samples:
                return
            samples.append(await post_query(client, args, question_factory(index), multi_query=multi_query))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    elapsed = time.perf_counter() - started
    return summarize_scenario(name, concurrency, samples, elapsed, notes=[] if not cache_check else ["cache checked separately"])


async def run_warmup(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    concurrency: int,
    question_factory,
    *,
    multi_query: bool,
) -> None:
    """Run and discard warmup requests."""
    counter = AtomicCounter()

    async def worker() -> None:
        while True:
            index = await counter.next()
            if index >= args.warmup:
                return
            await post_query(client, args, question_factory(index), multi_query=multi_query)

    await asyncio.gather(*(worker() for _ in range(max(1, min(concurrency, args.warmup)))))


async def post_query(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    question: str,
    *,
    multi_query: bool,
) -> RequestSample:
    """Post one query and measure latency."""
    started = time.perf_counter()
    try:
        response = await client.post(
            "/query",
            json=query_payload(question, args, multi_query=multi_query),
            headers=principal_headers(args.principal),
        )
        error = None if response.status_code == 200 else response.text[:300]
        return RequestSample(
            latency_ms=(time.perf_counter() - started) * 1000,
            status_code=response.status_code,
            error=error,
        )
    except Exception as error:
        return RequestSample(
            latency_ms=(time.perf_counter() - started) * 1000,
            status_code=0,
            error=type(error).__name__,
        )


async def run_live_stream(client: httpx.AsyncClient, args: argparse.Namespace) -> StreamResult:
    """Measure streaming first-event and first-token latency on a live miss."""
    question = f"{args.stream_question} live stream nonce {uuid.uuid4().hex}"
    return await post_stream(client, args, question, name="stream_live_miss")


async def run_replay_stream(client: httpx.AsyncClient, args: argparse.Namespace) -> StreamResult:
    """Measure streaming replay latency after priming L3 with a full answer."""
    question = f"{args.stream_question} replay stream fixed"
    prime = await client.post(
        "/query",
        json=query_payload(question, args, multi_query=True),
        headers=principal_headers(args.principal),
    )
    if prime.status_code != 200:
        return StreamResult(
            name="stream_l3_replay",
            status_code=prime.status_code,
            first_event_ms=None,
            first_token_ms=None,
            total_ms=0.0,
            event_names=[],
            cached=False,
            stream_replay=False,
            error=prime.text[:300],
        )
    return await post_stream(client, args, question, name="stream_l3_replay")


async def post_stream(client: httpx.AsyncClient, args: argparse.Namespace, question: str, *, name: str) -> StreamResult:
    """Post one streaming query and measure SSE timing."""
    started = time.perf_counter()
    first_event: Optional[float] = None
    first_token: Optional[float] = None
    event_names: List[str] = []
    cached = False
    stream_replay = False
    error: Optional[str] = None
    status_code = 0

    try:
        async with client.stream(
            "POST",
            "/query/stream",
            json=query_payload(question, args, multi_query=True),
            headers=principal_headers(args.principal),
        ) as response:
            status_code = response.status_code
            buffer: List[str] = []
            async for line in response.aiter_lines():
                if line == "":
                    event = parse_sse_block(buffer)
                    buffer = []
                    if event is None:
                        continue
                    event_name, payload = event
                    if first_event is None:
                        first_event = (time.perf_counter() - started) * 1000
                    event_names.append(event_name)
                    cached = cached or bool(payload.get("cached"))
                    stream_replay = stream_replay or bool(payload.get("stream_replay"))
                    if event_name == "token" and first_token is None:
                        first_token = (time.perf_counter() - started) * 1000
                    if event_name == "error":
                        error = json.dumps(payload, ensure_ascii=False)
                else:
                    buffer.append(line)
    except Exception as caught:
        error = type(caught).__name__
    total_ms = (time.perf_counter() - started) * 1000
    return StreamResult(
        name=name,
        status_code=status_code,
        first_event_ms=round_or_none(first_event),
        first_token_ms=round_or_none(first_token),
        total_ms=round(total_ms, 3),
        event_names=event_names,
        cached=cached,
        stream_replay=stream_replay,
        error=error,
    )


def query_payload(question: str, args: argparse.Namespace, *, multi_query: bool) -> Dict[str, Any]:
    """Build a query request that exercises the local benchmark funnel."""
    return {
        "question": question,
        "top_k": args.top_k,
        "embedding_provider": "hash",
        "embedding_dimension": args.embedding_dimension,
        "vector_store": "qdrant",
        "rerank_provider": "none",
        "hybrid": True,
        "hybrid_fetch_k": max(args.top_k * 4, 20),
        "multi_query": multi_query,
        "query_rewrite_provider": "deterministic",
        "query_rewrite_num_queries": 3,
        "context_packing": True,
        "context_dedup": True,
        "context_max_tokens": 2048,
        "max_context_chars": 4000,
    }


def principal_headers(principal: str) -> Dict[str, str]:
    """Return the trusted-principal header used by the service."""
    return {"X-Principal": principal}


async def get_json(client: httpx.AsyncClient, path: str) -> Dict[str, Any]:
    """Fetch JSON from a service endpoint."""
    response = await client.get(path)
    if response.status_code >= 400:
        raise RuntimeError(f"GET {path} failed with HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


async def assert_metrics_available(client: httpx.AsyncClient) -> None:
    """Fail early when observability metrics are not enabled."""
    response = await client.get("/metrics")
    if response.status_code != 200:
        raise RuntimeError("GET /metrics must be enabled for cache cross-checks.")


async def metrics_snapshot(client: httpx.AsyncClient) -> Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float]:
    """Read and parse the Prometheus text endpoint."""
    response = await client.get("/metrics")
    response.raise_for_status()
    return parse_prometheus(response.text)


async def cache_stats(client: httpx.AsyncClient) -> Dict[str, Any]:
    """Read the service cache counters."""
    response = await client.get("/cache/stats")
    response.raise_for_status()
    return response.json()


class AtomicCounter:
    """Tiny asyncio-safe counter for closed-loop workers."""

    def __init__(self) -> None:
        self.value = 0
        self._lock = asyncio.Lock()

    async def next(self) -> int:
        """Return and increment the current counter value."""
        async with self._lock:
            value = self.value
            self.value += 1
            return value


def summarize_scenario(
    name: str,
    concurrency: int,
    samples: Sequence[RequestSample],
    elapsed_seconds: float,
    *,
    notes: List[str],
) -> ScenarioResult:
    """Aggregate request samples into a scenario result."""
    latencies = [sample.latency_ms for sample in samples]
    status_codes: Dict[str, int] = {}
    for sample in samples:
        key = str(sample.status_code)
        status_codes[key] = status_codes.get(key, 0) + 1
    errors = sum(1 for sample in samples if sample.status_code != 200)
    return ScenarioResult(
        name=name,
        concurrency=concurrency,
        requests=len(samples),
        errors=errors,
        elapsed_seconds=round(elapsed_seconds, 6),
        qps=round(len(samples) / elapsed_seconds, 6) if elapsed_seconds > 0 else 0.0,
        latency_ms=summarize_latencies(latencies),
        status_codes=dict(sorted(status_codes.items())),
        notes=notes,
    )


def summarize_latencies(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Return latency summary with P99 omitted for small samples."""
    if not values:
        return {"min": None, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "min": round(min(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "p50": round(percentile(values, 50), 3),
        "p95": round(percentile(values, 95), 3),
        "p99": round(percentile(values, 99), 3) if len(values) >= 1000 else None,
        "max": round(max(values), 3),
    }


def percentile(values: Sequence[float], p: float) -> float:
    """Return a nearest-rank percentile for benchmark reporting."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if p < 0 or p > 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    if p == 0:
        return ordered[0]
    index = math.ceil((p / 100.0) * len(ordered)) - 1
    return ordered[min(max(index, 0), len(ordered) - 1)]


def parse_prometheus(text: str) -> Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float]:
    """Parse simple Prometheus counter/gauge lines into a keyed mapping."""
    parsed: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        metric_part, value_part = line.rsplit(" ", 1)
        name, labels = parse_metric_identity(metric_part)
        try:
            value = float(value_part)
        except ValueError:
            continue
        parsed[(name, tuple(sorted(labels.items())))] = value
    return parsed


def parse_metric_identity(metric_part: str) -> Tuple[str, Dict[str, str]]:
    """Parse a Prometheus metric name and labels."""
    if "{" not in metric_part:
        return metric_part, {}
    name, label_text = metric_part.split("{", 1)
    label_text = label_text.rstrip("}")
    labels: Dict[str, str] = {}
    if not label_text:
        return name, labels
    for item in label_text.split(","):
        key, value = item.split("=", 1)
        labels[key] = value.strip('"')
    return name, labels


def counter_delta(
    before: Mapping[Tuple[str, Tuple[Tuple[str, str], ...]], float],
    after: Mapping[Tuple[str, Tuple[Tuple[str, str], ...]], float],
    name: str,
    labels: Mapping[str, str],
) -> float:
    """Return the delta for a specific Prometheus series."""
    label_tuple = tuple(sorted(labels.items()))
    return after.get((name, label_tuple), 0.0) - before.get((name, label_tuple), 0.0)


def numeric_delta(before: Mapping[str, Any], after: Mapping[str, Any], key: str) -> float:
    """Return a numeric delta from two simple JSON mappings."""
    return float(after.get(key, 0) or 0) - float(before.get(key, 0) or 0)


def parse_sse_block(lines: Sequence[str]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Parse one SSE block into event name and JSON payload."""
    if not lines:
        return None
    event_name: Optional[str] = None
    data_lines: List[str] = []
    for line in lines:
        if line.startswith("event: "):
            event_name = line[len("event: ") :]
        elif line.startswith("data: "):
            data_lines.append(line[len("data: ") :])
    if event_name is None or not data_lines:
        return None
    return event_name, json.loads("\n".join(data_lines))


def ensure_allowed_base_url(base_url: str, allow_non_localhost: bool) -> None:
    """Keep the benchmark local by default."""
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if host in LOCAL_HOSTS:
        return
    if allow_non_localhost:
        return
    raise ValueError("Benchmark targets must be localhost unless --allow-non-localhost is set.")


def ensure_sampling_contract(warmup: int, samples: int, concurrency: Sequence[int]) -> None:
    """Enforce the published sampling floor."""
    if warmup < MIN_WARMUP:
        raise ValueError(f"warmup must be at least {MIN_WARMUP}")
    if samples < MIN_SAMPLES:
        raise ValueError(f"samples must be at least {MIN_SAMPLES}")
    if not concurrency or any(value <= 0 for value in concurrency):
        raise ValueError("concurrency values must be positive")


def environment_snapshot() -> Dict[str, Any]:
    """Capture local machine facts without adding runtime dependencies."""
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
        "docker_shape": "local compose target; Windows Docker Desktop/WSL overhead should be noted by the runner",
    }


def scenario_to_dict(result: ScenarioResult) -> Dict[str, Any]:
    """Serialize a scenario result."""
    payload = {
        "name": result.name,
        "concurrency": result.concurrency,
        "requests": result.requests,
        "errors": result.errors,
        "error_rate": round(result.errors / result.requests, 6) if result.requests else 0.0,
        "elapsed_seconds": result.elapsed_seconds,
        "qps": result.qps,
        "latency_ms": result.latency_ms,
        "status_codes": result.status_codes,
        "notes": result.notes,
    }
    if result.cache_delta is not None:
        payload["cache_delta"] = result.cache_delta
    return payload


def stream_to_dict(result: StreamResult) -> Dict[str, Any]:
    """Serialize a stream result."""
    return {
        "name": result.name,
        "status_code": result.status_code,
        "first_event_ms": result.first_event_ms,
        "first_token_ms": result.first_token_ms,
        "total_ms": result.total_ms,
        "event_names": result.event_names,
        "cached": result.cached,
        "stream_replay": result.stream_replay,
        "error": result.error,
    }


def write_reports(report: Mapping[str, Any], output_dir: str) -> Tuple[Path, Path]:
    """Write raw JSON and a readable Markdown summary."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = path / f"bench-{timestamp}.json"
    markdown_path = path / f"bench-{timestamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(format_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def format_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown report."""
    metadata = report["metadata"]
    lines = [
        "# Local Benchmark Report",
        "",
        f"- Started: `{metadata['started_at']}`",
        f"- Finished: `{metadata['finished_at']}`",
        f"- Base URL: `{metadata['base_url']}`",
        f"- Warmup discarded: `{metadata['warmup_discarded_per_scenario']}`",
        f"- Samples per scenario: `{metadata['samples_per_scenario']}`",
        f"- Zero-paid evidence: `{json.dumps(metadata['zero_paid_evidence'], sort_keys=True)}`",
        "",
        "## Query Scenarios",
        "",
        "| Scenario | Concurrency | Requests | Errors | QPS | P50 ms | P95 ms | P99 ms | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["query_scenarios"]:
        latency = item["latency_ms"]
        lines.append(
            "| {name} | {concurrency} | {requests} | {errors} | {qps:.3f} | {p50} | {p95} | {p99} | {notes} |".format(
                name=item["name"],
                concurrency=item["concurrency"],
                requests=item["requests"],
                errors=item["errors"],
                qps=float(item["qps"]),
                p50=latency["p50"],
                p95=latency["p95"],
                p99=latency["p99"] if latency["p99"] is not None else "not reported",
                notes="; ".join(item.get("notes") or []),
            )
        )
    lines.extend(
        [
            "",
            "## Streaming",
            "",
            "| Scenario | HTTP | First event ms | First token ms | Total ms | Cached | Replay | Events |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for item in report["streaming"]:
        lines.append(
            "| {name} | {status} | {first_event} | {first_token} | {total} | {cached} | {replay} | {events} |".format(
                name=item["name"],
                status=item["status_code"],
                first_event=item["first_event_ms"],
                first_token=item["first_token_ms"],
                total=item["total_ms"],
                cached=item["cached"],
                replay=item["stream_replay"],
                events=", ".join(item["event_names"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def round_or_none(value: Optional[float]) -> Optional[float]:
    """Round a float when present."""
    return None if value is None else round(value, 3)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    stub: Optional[DeterministicChatStub] = None
    if args.chat_stub:
        stub = DeterministicChatStub(args.chat_stub_host, args.chat_stub_port)
        stub.start()
    try:
        report = asyncio.run(run_benchmark(args, stub))
        json_path, markdown_path = write_reports(report, args.output_dir)
        print(f"Wrote {json_path}")
        print(f"Wrote {markdown_path}")
        return 0
    finally:
        if stub is not None:
            stub.stop()


if __name__ == "__main__":
    raise SystemExit(main())
