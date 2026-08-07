# Frozen Retrieval Baselines

The CI contract runs `python -m ci.retrieval_baseline_gate` without network access or cache state. It evaluates 20 labeled queries—16 positive and 4 negative—against five synthetic Markdown documents of 765–887 bytes. The golden-set SHA-256 is `0f341e368e2d8f0e65603133233a88da5d55a6a3debb090ea7a3eb35fb3b7a78`.

## Current MRR@3 contract

| Profile | Previous gate | Exact-source-slice gate | Result |
| --- | ---: | ---: | --- |
| dense | `0.677083` | `0.677083` | unchanged |
| hybrid | `1.000000` | `1.000000` | unchanged |
| parent_child | `0.625000` | `0.614583` | intentionally re-pinned after the span migration |
| rerank | not gated | `1.000000` | added to the gate |

These MRR values are computed over positive queries. At top-k 3, all four negative queries still return non-empty results, so `negative_false_recall_rate` is `1.000000`. A perfect rerank or hybrid MRR on this fixture is a mechanism check on a small corpus, not evidence of large-corpus quality or abstention.

## Why parent_child changed

The old splitter stripped pieces, rejoined them with new separators, and then tried to recover source offsets with `find()`. When lookup failed, the hierarchical path silently replaced the intended chunk with a same-length slice starting at a fallback offset. The resulting text was not the intended parent or child chunk even though its final metadata could appear self-consistent.

The migrated splitter carries half-open offsets throughout splitting and merging. Every chunk is now an exact contiguous source slice. With those semantics, q013 and q015 move outside the first three parent-child results while q014 moves to rank 1, changing MRR@3 from `0.625000` to `0.614583`. Dense and hybrid stay unchanged, which bounds the measured effect to the hierarchical chunk corpus.

## Reproducibility controls

Supported files are enumerated by normalized relative path, and record ids normalize source path separators. Equal vector scores use an explicit key of normalized source, chunk index, source span, and record id, so insertion order cannot select a different tied result. The regression suite builds the same tied corpus in forward and reverse order and requires identical rankings.

`requirements-ci.in` records the known-green direct dependency versions. `requirements-ci.constraints` carries the platform-neutral transitive versions from the Python 3.12.7 environment that passed the complete gate; Linux/Windows-only packages still receive universal environment markers during resolution. Every package shared by that environment and the compiled lock matches exactly. `requirements-ci.txt` is the fully resolved Python 3.12 universal lock used by GitHub Actions and can be regenerated with:

```bash
uv pip compile --universal --python-version 3.12 -c requirements-ci.constraints requirements-ci.in -o requirements-ci.txt
```

Evaluation JSON and Markdown reports are generated under the chosen `--report-dir` and remain untracked. CI uploads `ci-artifacts/` for each workflow run; `ci/retrieval_baseline_gate.py` is the authoritative numeric assertion.
