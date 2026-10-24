"""
main.py - controls the entire RAG eval flow.

For each model:
  1. load the frozen golden dataset (question + retrieved context)
  2. generate an answer from that context (backend generate_answer)
  3. evaluate (refusal / citations / grounding / correctness, evaluator.py)
  4. log the score

Files it depends on:
  - golden_dataset.csv   (built by make_golden_dataset.py)
  - models.py            (MODELS under test)
  - evaluator.py         (the scoring logic)

Run (from repo root):
  python -m eval.rag_eval.main
  python -m eval.rag_eval.main --models "GPT-OSS 120B" --limit 5 --threshold 0.6
  python -m eval.rag_eval.main --rescore        # re-grade saved answers after editing evaluator/KEYS
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

from backend.config import GROQ_API_KEY
from backend.generation.groq_client import generate_answer
from eval.rag_eval.evaluator import evaluate_one
from eval.rag_eval.models import MAX_TOKENS, MODELS

HERE = Path(__file__).parent
GOLDEN_PATH = HERE / "golden_dataset.csv"
RESULTS_PATH = HERE / "eval_results.csv"


def generate_with_retry(question, chunks, slug, attempts=5):
    """Groq free tier has low tokens-per-minute limits: on 429, wait as told and retry."""
    for i in range(attempts):
        try:
            return generate_answer(question, chunks, model=slug, temperature=0,
                                   max_tokens=MAX_TOKENS.get(slug, 2000))
        except Exception as e:
            if "rate_limit_exceeded" not in str(e) or i == attempts - 1:
                raise
            m = re.search(r"try again in ([\d.]+)s", str(e))
            wait = float(m.group(1)) + 1 if m else 2 ** (i + 2)
            print(f"      rate limited, retrying in {wait:.1f}s")
            time.sleep(wait)


def summarise(df, golden, models):
    print(f"\n{'='*60}\nFINAL SCOREBOARD\n{'='*60}")
    by_diff = df.pivot_table(index="model", columns="difficulty", values="correct", aggfunc="mean")
    summary = df.groupby("model").agg(accuracy=("correct", "mean"),
                                      cite_prec=("citation_precision", "mean"),
                                      answer_recall=("answer_recall", "mean"))
    print((summary.join(by_diff) * 100).round(1).reindex([n for n, _ in models]).to_string())
    print("\nFailure reasons:")
    print(pd.crosstab(df["model"], df["reason"]).reindex([n for n, _ in models]).to_string())
    hit = (golden["retrieval_hit"] == "TRUE").sum() / (golden["not_found"] == "FALSE").sum()
    print(f"\nRetrieval hit-rate@k (frozen context): {100*hit:.1f}%")
    return summary


def rescore(models):
    """Re-grade answers already in eval_results.csv with the current evaluator - no API calls."""
    golden = pd.read_csv(GOLDEN_PATH, dtype=str, keep_default_na=False)
    by_id = golden.set_index("id")
    df = pd.read_csv(RESULTS_PATH, dtype=str, keep_default_na=False)
    df = df[df["model"].isin([n for n, _ in models])]
    rows = []
    for _, r in df.iterrows():
        g = by_id.loc[r["id"]]
        verdict = (dict(correct=False, reason="gen_error", citation_precision=None, answer_recall=None)
                   if r["reason"] == "gen_error"
                   else evaluate_one(g, r["answer"], json.loads(g["context_json"])))
        rows.append(dict(model=r["model"], id=r["id"], difficulty=r["difficulty"], **verdict,
                         retrieval_hit=g["retrieval_hit"], answer=r["answer"]))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_PATH, index=False)
    return summarise(out, golden[golden["id"].isin(out["id"])], models)


def run_eval(models, limit=None):
    golden = pd.read_csv(GOLDEN_PATH, dtype=str, keep_default_na=False)
    if limit:
        golden = golden.head(limit)
    total = len(golden)

    all_rows = []
    for name, slug in models:
        print(f"\n{'='*60}\nMODEL: {name}  ({slug})\n{'='*60}")
        correct = 0

        for _, g in golden.iterrows():
            chunks = json.loads(g["context_json"])

            # 1-2. generate
            try:
                answer = generate_with_retry(g["question"], chunks, slug)
            except Exception as e:
                print(f"  #{g['id']:2} [{g['difficulty']:8}] GEN-ERROR: {e}")
                all_rows.append(dict(model=name, id=g["id"], difficulty=g["difficulty"],
                                     correct=False, reason="gen_error", citation_precision=None,
                                     answer_recall=None, retrieval_hit=g["retrieval_hit"], answer=""))
                continue

            # 3. evaluate
            verdict = evaluate_one(g, answer, chunks)
            correct += verdict["correct"]

            mark = "OK " if verdict["correct"] else "XX "
            print(f"  #{g['id']:2} [{g['difficulty']:8}] {mark} {verdict['reason']}")
            all_rows.append(dict(model=name, id=g["id"], difficulty=g["difficulty"], **verdict,
                                 retrieval_hit=g["retrieval_hit"], answer=answer))

        print(f"\n  SCORE: {correct}/{total} = {100*correct/total:.1f}%")

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_PATH, index=False)

    # 4. log final scores
    summary = summarise(df, golden, models)
    print(f"Detailed results saved -> {RESULTS_PATH}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="display names from models.py (default: all)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threshold", type=float, help="exit 1 if any model's accuracy is below this")
    ap.add_argument("--rescore", action="store_true", help="re-grade saved eval_results.csv, no API calls")
    args = ap.parse_args()
    models = [m for m in MODELS if not args.models or m[0] in args.models]

    if args.rescore:
        rescore(models)
        sys.exit()

    if not GROQ_API_KEY:
        sys.exit("Missing GROQ_API_KEY - add it to your .env file")
    summary = run_eval(models, args.limit)
    if args.threshold is not None and (summary["accuracy"] < args.threshold).any():
        sys.exit(f"FAIL: accuracy below threshold {args.threshold}")
