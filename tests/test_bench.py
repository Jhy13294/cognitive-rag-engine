import unittest

from bench.run import (
    counter_delta,
    ensure_allowed_base_url,
    parse_prometheus,
    percentile,
    summarize_latencies,
)


class BenchUtilityTests(unittest.TestCase):
    """Pure unit tests for local benchmark helpers."""

    def test_percentile_uses_nearest_rank(self) -> None:
        values = [1, 2, 3, 4, 5]

        self.assertEqual(percentile(values, 50), 3)
        self.assertEqual(percentile(values, 95), 5)

    def test_small_sample_summary_omits_p99(self) -> None:
        summary = summarize_latencies([10.0, 20.0, 30.0])

        self.assertEqual(summary["p50"], 20.0)
        self.assertIsNone(summary["p99"])

    def test_prometheus_parser_and_counter_delta(self) -> None:
        before = parse_prometheus('rag_cache_operations_total{cache_layer="L3",outcome="hit"} 7\n')
        after = parse_prometheus('rag_cache_operations_total{cache_layer="L3",outcome="hit"} 12\n')

        self.assertEqual(
            counter_delta(
                before,
                after,
                "rag_cache_operations_total",
                {"cache_layer": "L3", "outcome": "hit"},
            ),
            5,
        )

    def test_base_url_defaults_to_loopback_only(self) -> None:
        ensure_allowed_base_url("http://localhost:8000", allow_non_localhost=False)

        with self.assertRaises(ValueError):
            ensure_allowed_base_url("https://example.com", allow_non_localhost=False)


if __name__ == "__main__":
    unittest.main()
