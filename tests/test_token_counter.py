import unittest
from unittest.mock import patch

from tokenization import (
    HeuristicTokenCounter,
    TiktokenCounter,
    create_token_counter,
    legacy_len_div_four_count,
    token_count_error_percent,
    validate_tokenizer_encoding,
)


class TokenCounterTests(unittest.TestCase):
    def test_heuristic_counter_is_deterministic_and_cjk_aware(self):
        counter = HeuristicTokenCounter()
        text = "企业级知识库 supports RAG"

        self.assertEqual(counter.count(text), counter.count(text))
        self.assertGreater(counter.count(text), legacy_len_div_four_count(text))

    def test_create_token_counter_falls_back_when_tiktoken_is_missing(self):
        with patch("tokenization.counter.TiktokenCounter", side_effect=RuntimeError("missing")):
            counter = create_token_counter("cl100k_base")

        self.assertIsInstance(counter, HeuristicTokenCounter)

    def test_validate_tokenizer_encoding_rejects_unknown_encoding(self):
        with self.assertRaises(ValueError):
            validate_tokenizer_encoding("unknown_encoding")

    def test_tiktoken_counter_has_zero_error_against_itself_when_available(self):
        try:
            counter = TiktokenCounter("cl100k_base")
        except RuntimeError:
            self.skipTest("tiktoken is not installed")

        samples = [
            "企业级知识库支持检索增强生成。",
            "Enterprise knowledge bases support retrieval augmented generation.",
        ]
        for text in samples:
            true_count = counter.count(text)
            legacy_error = token_count_error_percent(legacy_len_div_four_count(text), true_count)
            fixed_error = token_count_error_percent(counter.count(text), true_count)

            self.assertGreaterEqual(legacy_error, 0.0)
            self.assertEqual(fixed_error, 0.0)


if __name__ == "__main__":
    unittest.main()
