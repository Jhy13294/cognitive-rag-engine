import unittest

from ci.retrieval_baseline_gate import assert_profile_mrr
from ci.unittest_gate import accounting_mismatches


class CIGateTests(unittest.TestCase):
    def test_retrieval_baseline_gate_rejects_exact_mrr_drift(self):
        report = {"metrics": {"3": {"mrr": 0.677083}}}

        self.assertEqual(
            assert_profile_mrr("dense", report, "0.677083")["mrr_at_3"],
            "0.677083",
        )
        with self.assertRaisesRegex(AssertionError, "MRR@3 drifted"):
            assert_profile_mrr("dense", report, "0.677084")

    def test_unittest_gate_rejects_skip_bucket_drift(self):
        summary = {
            "ran": 1,
            "failures": 0,
            "errors": 0,
            "skipped": 1,
            "skip_buckets": {"one reason": 1},
            "expected": {
                "ran": 1,
                "failures": 0,
                "errors": 0,
                "skipped": 1,
                "skip_buckets": {"different reason": 1},
            },
        }

        self.assertTrue(
            any("skip buckets drifted" in item for item in accounting_mismatches(summary))
        )


if __name__ == "__main__":
    unittest.main()
