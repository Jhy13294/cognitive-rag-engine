"""Run the deterministic local chat stub used by the zero-key demo.

This starts the same OpenAI-compatible stub that the local benchmark uses
(`bench.run.DeterministicChatStub`), so the compose stack can answer
`/query` and `/query/stream` requests without any paid API key. Keep it
running while you send demo queries; stop it with Ctrl+C.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.run import DeterministicChatStub  # noqa: E402


def main() -> None:
    stub = DeterministicChatStub("0.0.0.0", 18080)
    stub.start()
    print("Deterministic chat stub listening on http://0.0.0.0:18080 (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stub.stop()
        print(f"Stub stopped | served: {stub.snapshot()}")


if __name__ == "__main__":
    main()
