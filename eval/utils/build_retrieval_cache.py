# eval/deepeval_suite/build_retrieval_cache.py
import json
from pathlib import Path

from backend.retrieval.hybrid import retrieve

GOLDEN_PATH = Path("eval/deepeval_suite/goldens/retriever_golden_dataset.json")
CACHE_PATH = Path("eval/deepeval_suite/goldens/retrieval_cache.json")

goldens = json.loads(GOLDEN_PATH.read_text())
cache = {}
for g in goldens:
    retrieved = retrieve(g["question"], chat_id=None)
    cache[g["question"]] = [doc["text"] for doc in retrieved]

CACHE_PATH.write_text(json.dumps(cache, indent=2))
print(f"Cached retrieval for {len(cache)} questions.")