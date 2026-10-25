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
  python -m eval.rag_eval.main --judge          # + LLM-as-judge faithfulness / answer relevancy (JUDGE_MODEL)
  python -m eval.rag_eval.main --rescore --judge # judge saved answers without regenerating them
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from groq import APIConnectionError

from backend.config import GROQ_API_KEY
from backend.generation.groq_client import generate_answer
from eval.rag_eval.evaluator import evaluate_one
from eval.rag_eval.judge import judge
from eval.rag_eval.models import MAX_TOKENS, MODELS

HERE = Path(__file__).parent
GOLDEN_PATH = HERE / "golden_dataset.csv"
RESULTS_PATH = HERE / "eval_results.csv"
JUDGE_COLS = ("faithfulness", "answer_relevancy")


def generate_with_retry(question, chunks, slug, attempts=5):
    return with_retry(lambda: generate_answer(question, chunks, model=slug, temperature=0,
                                              max_tokens=MAX_TOKENS.get(slug, 2000)), attempts)


class NetworkDown(RuntimeError):
    """Groq unreachable for several minutes: abort the run rather than record
    every remaining question as a wrong answer and overwrite eval_results.csv."""


NETWORK_WAITS = (5, 15, 30, 60, 120, 120)  # ~6 min of backoff before giving up


def with_retry(call, attempts=5):
    """Groq free tier has low tokens-per-minute limits: on 429, wait as told and retry.
    On connection errors/timeouts (network drop), back off and retry, then raise NetworkDown."""
    rate_tries = net_tries = 0
    while True:
        try:
            return call()
        except APIConnectionError as e:  # includes APITimeoutError
            if net_tries == len(NETWORK_WAITS):
                raise NetworkDown(f"Groq unreachable for ~{sum(NETWORK_WAITS) // 60} min: {e}") from e
            wait = NETWORK_WAITS[net_tries]
            net_tries += 1
            print(f"      connection error, retrying in {wait}s")
            time.sleep(wait)
        except Exception as e:
            rate_tries += 1
            if "rate_limit_exceeded" not in str(e) or rate_tries == attempts:
                raise
            m = re.search(r"try again in ([\d.]+)s", str(e))
            wait = float(m.group(1)) + 1 if m else 2 ** (rate_tries + 1)
            print(f"      rate limited, retrying in {wait:.1f}s")
            time.sleep(wait)


def retrieval_metrics(golden):
    """Precision@k / recall@k / MRR of the frozen context against each answerable row's source.

    With one gold source per question, recall@k is 1 if any source chunk was retrieved
    (hit-rate) and precision@k is the share of the k chunks that come from the source.
    """
    precision, recall, rr = [], [], []
    for _, g in golden[golden["not_found"] == "FALSE"].iterrows():
        ids = [c["chunk_id"] for c in json.loads(g["context_json"])]
        rel = [any(cid.startswith(f"{s}_c") for s in g["source"].split("|")) for cid in ids]
        precision.append(sum(rel) / len(ids) if ids else 0.0)
        recall.append(float(any(rel)))
        rr.append(1 / (rel.index(True) + 1) if any(rel) else 0.0)
    n = len(recall) or 1
    return dict(precision=sum(precision) / n, recall=sum(recall) / n, mrr=sum(rr) / n)


def add_judge_scores(df, golden):
    """LLM-as-judge faithfulness + answer relevancy for every answer with content.

    A false_refusal that also carries content (e.g. "I cannot find... but X attended Y")
    is judged too: that content is exactly where hallucinations hide."""
    by_id = golden.set_index("id")
    for i, r in df.iterrows():
        bare_refusal = r["reason"] == "false_refusal" and len(str(r["answer"]).strip()) < 80
        if bare_refusal or r["reason"] in ("gen_error", "empty_answer", "refusal_ok"):
            continue
        g = by_id.loc[r["id"]]
        try:
            scores = with_retry(lambda: judge(g["question"], r["answer"], json.loads(g["context_json"])))
        except NetworkDown:
            raise
        except Exception as e:
            print(f"  judge failed on {r['model']} #{r['id']}: {e}")
            continue
        for k, v in scores.items():
            df.loc[i, k] = v
    return df


def summarise(df, golden, models):
    print(f"\n{'='*60}\nFINAL SCOREBOARD\n{'='*60}")
    df = df.copy()
    for col in ("correct", "citation_precision", "answer_recall") + JUDGE_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col].replace({"True": 1, "False": 0, "": None}))
    by_diff = df.pivot_table(index="model", columns="difficulty", values="correct", aggfunc="mean")
    aggs = dict(accuracy=("correct", "mean"), cite_prec=("citation_precision", "mean"),
                answer_recall=("answer_recall", "mean"))
    aggs.update({c: (c, "mean") for c in JUDGE_COLS if c in df})
    summary = df.groupby("model").agg(**aggs)
    print((summary.join(by_diff) * 100).round(1).reindex([n for n, _ in models]).to_string())
    print("\nFailure reasons:")
    print(pd.crosstab(df["model"], df["reason"]).reindex([n for n, _ in models]).to_string())
    m = retrieval_metrics(golden)
    print(f"\nRetrieval (frozen context): precision@k {100*m['precision']:.1f}%  "
          f"recall@k {100*m['recall']:.1f}%  MRR {m['mrr']:.3f}")
    return summary


def rescore(models, use_judge=False):
    """Re-grade answers already in eval_results.csv with the current evaluator.
    No API calls unless use_judge (then only the judge runs, answers are not regenerated)."""
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
                         retrieval_hit=g["retrieval_hit"], answer=r["answer"],
                         **{c: r[c] for c in JUDGE_COLS if c in r}))
    out = pd.DataFrame(rows)
    if use_judge:
        out = add_judge_scores(out, golden)
    out.to_csv(RESULTS_PATH, index=False)
    return summarise(out, golden[golden["id"].isin(out["id"])], models)


def run_eval(models, limit=None, use_judge=False):
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
            except NetworkDown:
                raise
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
    if use_judge:
        df = add_judge_scores(df, golden)
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
    ap.add_argument("--judge", action="store_true", help="add LLM-as-judge faithfulness + answer relevancy")
    args = ap.parse_args()
    models = [m for m in MODELS if not args.models or m[0] in args.models]

    if not GROQ_API_KEY and (args.judge or not args.rescore):
        sys.exit("Missing GROQ_API_KEY - add it to your .env file")
    try:
        if args.rescore:
            rescore(models, args.judge)
            sys.exit()
        summary = run_eval(models, args.limit, args.judge)
    except NetworkDown as e:
        sys.exit(f"ABORTED: {e}\n{RESULTS_PATH.name} was left unchanged - re-run when the network is back.")
    if args.threshold is not None and (summary["accuracy"] < args.threshold).any():
        sys.exit(f"FAIL: accuracy below threshold {args.threshold}")
