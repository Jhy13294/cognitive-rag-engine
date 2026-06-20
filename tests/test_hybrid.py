import unittest

from hybrid import BM25Retriever, RRFConfig, RankedRecord, ReciprocalRankFusion
from lexical import bm25_score, build_bm25_index, tokenize
from vector_store import VectorRecord


class LexicalCoreTests(unittest.TestCase):
    def test_tokenize_normalizes_tokens_deterministically(self):
        tokens = tokenize("Traveling employees submitted fuel receipts.")

        self.assertEqual(tokens, ["travel", "employ", "submitt", "fuel", "receipt"])

    def test_bm25_score_prefers_matching_document(self):
        index = build_bm25_index(
            {
                "a": tokenize("mileage reimbursement fuel receipts"),
                "b": tokenize("hardware token recovery code"),
            }
        )

        matching = bm25_score(tokenize("fuel reimbursement"), index, "a")
        non_matching = bm25_score(tokenize("fuel reimbursement"), index, "b")

        self.assertGreater(matching, non_matching)


class HybridRetrievalTests(unittest.TestCase):
    def test_bm25_retriever_emits_vector_record_id(self):
        records = [
            VectorRecord(
                id="record-finance",
                content="The EXP-204 mileage reimbursement policy requires fuel receipts.",
                embedding=[1.0, 0.0],
                metadata={"source": "finance.md", "chunk_index": 0},
            ),
            VectorRecord(
                id="record-security",
                content="Hardware token recovery uses the SEC-44 process.",
                embedding=[0.0, 1.0],
                metadata={"source": "security.md", "chunk_index": 0},
            ),
        ]
        retriever = BM25Retriever(records)

        results = retriever.retrieve("Which reimbursement policy needs fuel receipts?", top_k=2)

        self.assertEqual(results[0].id, "record-finance")
        self.assertEqual(results[0].metadata["id"], "record-finance")

    def test_bm25_retriever_applies_acl_filter_before_top_k(self):
        records = [
            VectorRecord(
                id="unauthorized-best",
                content="mileage reimbursement reimbursement reimbursement fuel receipts",
                embedding=[1.0, 0.0],
                metadata={"source": "secret.md", "acl": ["role:secret"]},
            ),
            VectorRecord(
                id="authorized-weaker",
                content="mileage reimbursement policy",
                embedding=[0.0, 1.0],
                metadata={"source": "finance.md", "acl": ["role:finance"]},
            ),
        ]
        retriever = BM25Retriever(records)

        results = retriever.retrieve(
            "mileage reimbursement fuel",
            top_k=1,
            metadata_filter={"acl": ["role:finance"]},
        )

        self.assertEqual([result.id for result in results], ["authorized-weaker"])

    def test_rrf_merges_dense_and_sparse_by_record_id(self):
        dense = [
            RankedRecord(id="shared", score=0.9, content="dense shared", metadata={"source": "a.md"}),
            RankedRecord(id="dense-only", score=0.8, content="dense only", metadata={"source": "b.md"}),
        ]
        sparse = [
            RankedRecord(id="sparse-only", score=3.0, content="sparse only", metadata={"source": "c.md"}),
            RankedRecord(id="shared", score=2.0, content="sparse shared", metadata={"source": "a.md"}),
        ]
        fusion = ReciprocalRankFusion(RRFConfig(k=60, weights={"dense": 1.0, "sparse": 1.0}))

        results = fusion.fuse({"dense": dense, "sparse": sparse}, top_k=3)

        shared = next(result for result in results if result.id == "shared")
        self.assertEqual(shared.metadata["dense_rank"], 1)
        self.assertEqual(shared.metadata["sparse_rank"], 2)
        self.assertGreater(shared.score, results[-1].score)

    def test_rrf_tie_breaks_by_record_id(self):
        fusion = ReciprocalRankFusion(RRFConfig(k=60, weights={"dense": 1.0, "sparse": 1.0}))

        results = fusion.fuse(
            {
                "dense": [RankedRecord(id="b", score=0.1, content="b", metadata={})],
                "sparse": [RankedRecord(id="a", score=0.1, content="a", metadata={})],
            },
            top_k=2,
        )

        self.assertEqual([result.id for result in results], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
