"""
Builds eval/rag_eval/golden_dataset.csv from GOLDEN.

Runs hybrid retrieval ONCE per question and freezes the retrieved chunks
into the CSV (context_json), so every model under test answers from the
exact same context - and main.py needs neither Qdrant nor the embedder.

Run (from repo root, Qdrant up):  python -m eval.rag_eval.make_golden_dataset
Output: eval/rag_eval/golden_dataset.csv
"""
import csv, json
from pathlib import Path

from backend.retrieval.hybrid import retrieve
from eval.rag_eval.golden_dataset_generator import GOLDEN, KEYS

TOP_K = 5
OUT = Path(__file__).parent / "golden_dataset.csv"

rows_out = []
hits = answerable = 0
for g in GOLDEN:
    chunks = retrieve(g["q"], top_k=TOP_K)
    ctx = [{k: c.get(k) for k in ("chunk_id", "text", "paper_title", "authors", "year", "page")}
           for c in chunks]
    hit = (not g["not_found"]) and any(c["chunk_id"].startswith(f"{g['source']}_c") for c in ctx)
    if not g["not_found"]:
        answerable += 1
        hits += hit
    print(f"[{g['diff']:8}] #{g['id']:2}  hit={str(hit):5}  top={[c['chunk_id'] for c in ctx]}")
    rows_out.append({
        "id": g["id"],
        "difficulty": g["diff"],
        "question": g["q"],
        "expected_answer": g["answer"],
        "source": g["source"],
        "keys_json": json.dumps(KEYS.get(g["id"], [])),
        "not_found": str(g["not_found"]).upper(),
        "context_json": json.dumps(ctx, ensure_ascii=False),
        "retrieval_hit": str(hit).upper(),
    })

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
    w.writeheader(); w.writerows(rows_out)

print(f"\nRetrieval hit-rate@{TOP_K}: {hits}/{answerable} = {100*hits/answerable:.1f}%")
print(f"Wrote {OUT} with {len(rows_out)} rows")
