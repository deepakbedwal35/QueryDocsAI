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

import fitz

from backend.ingestion.pipeline import process_pdf
from backend.ingestion.qdrant_utils import delete_points_by_doc, upsert_points
from backend.retrieval.hybrid import retrieve
from eval.rag_eval.golden_dataset_generator import (GOLDEN, KEYS, UPLOAD_CHAT_ID,
                                                     UPLOAD_DOC_ID, UPLOAD_FIXTURE)

TOP_K = 5
OUT = Path(__file__).parent / "golden_dataset.csv"

# (Re)upload the fixture resume into the eval chat, same path as a user upload.
pdf = fitz.open()
pdf.new_page().insert_textbox(fitz.Rect(50, 50, 550, 800), UPLOAD_FIXTURE, fontsize=10)
delete_points_by_doc(UPLOAD_DOC_ID, UPLOAD_CHAT_ID)
points, _ = process_pdf(pdf.tobytes(), UPLOAD_DOC_ID, UPLOAD_CHAT_ID, "eval_resume.pdf")
upsert_points(points)

rows_out = []
hits = answerable = 0
for g in GOLDEN:
    chunks = retrieve(g["q"], top_k=TOP_K, chat_id=UPLOAD_CHAT_ID if g.get("chat") else None)
    ctx = [{k: c.get(k) for k in ("chunk_id", "text", "paper_title", "authors", "year", "page")}
           for c in chunks]
    hit = (not g["not_found"]) and any(c["chunk_id"].startswith(f"{s}_c")
                                       for c in ctx for s in g["source"].split("|"))
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
