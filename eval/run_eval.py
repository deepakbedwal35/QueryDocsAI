"""
eval/run_eval.py

Runs the eval suite against the live pipeline (retrieve -> generate
-> verify) and reports:

  - hit-rate@k       : did retrieval surface at least one of the
                        expected chunk_ids, for questions where we
                        know the answer should be in the corpus?
  - faithfulness      : of the citations the model produced, what
                        fraction were valid (i.e. not hallucinated)?
  - not-found accuracy: for questions with no answer in the corpus,
                        did the pipeline correctly say so (and vice
                        versa for questions that should be answered)?

Usage (run from the project root, so imports resolve):
    python -m eval.run_eval
    python -m eval.run_eval --top-k 3
    python -m eval.run_eval --limit 5      # quick smoke test, cheap on API calls

Requires GROQ_API_KEY to be set (loaded via backend.config), since
this exercises the real LLM, not a mock.

Exit code is 1 if aggregate metrics fall below the thresholds below,
so this can gate CI (see .github/workflows/eval.yml).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import backend.config  # noqa: F401  (loads .env -> GROQ_API_KEY)
from backend.generation.citation_verifier import verify_citations
from backend.generation.groq_client import generate_answer
from real_retrieval import retrieve

HERE = Path(__file__).parent
QUESTIONS_PATH = HERE / "test_questions.json"
REPORT_PATH = HERE / "results" / "eval_report.json"

_NOT_FOUND_PHRASE = "cannot find the answer"

# Minimum acceptable aggregate scores for eval.yml to pass. Tune these
# as the real (non-mock) retrieval pipeline comes online — mock
# retrieval's naive keyword matching should comfortably clear these.
HIT_RATE_THRESHOLD = 0.7
FAITHFULNESS_THRESHOLD = 0.9
NOT_FOUND_ACCURACY_THRESHOLD = 0.8


def load_questions() -> list[dict]:
    questions = json.loads(QUESTIONS_PATH.read_text())
    for i, q in enumerate(questions, start=1):
        q.setdefault("id", f"q{i:03d}")
    return questions


def hit_rate_at_k(
    expected_paper_titles: list[str], retrieved_chunks: list[dict]
) -> bool | None:
    """
    True/False if this question has known expected papers; None (not
    applicable) if it's a not-found question with nothing to check
    retrieval against.

    Checked at the paper level, not exact chunk_id, since chunk
    boundaries are somewhat arbitrary -- what matters is whether
    retrieval surfaced content from the RIGHT PAPER, not the exact
    same slice of text a human happened to pick.
    """
    if not expected_paper_titles:
        return None
    retrieved_titles = {c["paper_title"] for c in retrieved_chunks}
    return any(title in retrieved_titles for title in expected_paper_titles)


def run_one(question_item: dict, top_k: int) -> dict:
    question = question_item["question"]
    expected_paper_titles = question_item.get("expected_paper_titles", [])
    expect_not_found = question_item.get("expect_not_found", False)

    chunks = retrieve(question, top_k=top_k)
    hit = hit_rate_at_k(expected_paper_titles, chunks)

    raw_answer = generate_answer(question, chunks)
    result = verify_citations(raw_answer, chunks)

    valid = result["valid_citations"]
    hallucinated = result["hallucinated_citations"]
    total_citations = len(valid) + len(hallucinated)
    faithfulness = (len(valid) / total_citations) if total_citations > 0 else None

    answer_found = _NOT_FOUND_PHRASE not in result["clean_answer"].lower()
    not_found_correct = (
        (answer_found is False) if expect_not_found else (answer_found is True)
    )

    return {
        "id": question_item["id"],
        "question": question,
        "expected_paper_titles": expected_paper_titles,
        "retrieved_paper_titles": list({c["paper_title"] for c in chunks}),
        "hit_at_k": hit,
        "answer": result["clean_answer"],
        "valid_citations": valid,
        "hallucinated_citations": hallucinated,
        "faithfulness": faithfulness,
        "expect_not_found": expect_not_found,
        "answer_found": answer_found,
        "not_found_correct": not_found_correct,
    }


def summarize(results: list[dict]) -> dict:
    hit_results = [r["hit_at_k"] for r in results if r["hit_at_k"] is not None]
    faithfulness_results = [
        r["faithfulness"] for r in results if r["faithfulness"] is not None
    ]
    not_found_results = [r["not_found_correct"] for r in results]

    hit_rate = sum(hit_results) / len(hit_results) if hit_results else None
    faithfulness = (
        sum(faithfulness_results) / len(faithfulness_results)
        if faithfulness_results
        else None
    )
    not_found_accuracy = (
        sum(not_found_results) / len(not_found_results) if not_found_results else None
    )

    return {
        "num_questions": len(results),
        "hit_rate_at_k": hit_rate,
        "faithfulness": faithfulness,
        "not_found_accuracy": not_found_accuracy,
        "hallucinated_citation_count": sum(
            len(r["hallucinated_citations"]) for r in results
        ),
    }


def passes_thresholds(summary: dict) -> bool:
    checks = [
        summary["hit_rate_at_k"] is None
        or summary["hit_rate_at_k"] >= HIT_RATE_THRESHOLD,
        summary["faithfulness"] is None
        or summary["faithfulness"] >= FAITHFULNESS_THRESHOLD,
        summary["not_found_accuracy"] is None
        or summary["not_found_accuracy"] >= NOT_FOUND_ACCURACY_THRESHOLD,
    ]
    return all(checks)


def main():
    parser = argparse.ArgumentParser(description="Run the Ask My Papers eval suite.")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Chunks to retrieve per question."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only run the first N questions."
    )
    args = parser.parse_args()

    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]

    results = []
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {q['question'][:70]}...")
        try:
            results.append(run_one(q, top_k=args.top_k))
        except (
            Exception
        ) as e:  # noqa: BLE001 - eval should keep going and report the failure
            results.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "error": str(e),
                }
            )
            print(f"  ERROR: {e}")

    summary = summarize([r for r in results if "error" not in r])
    errored = [r for r in results if "error" in r]

    report = {
        "summary": summary,
        "errored_count": len(errored),
        "passed_thresholds": passes_thresholds(summary) if not errored else False,
        "results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("\n--- Summary ---")
    print(f"Questions run:        {summary['num_questions']}")
    print(f"Errored:              {len(errored)}")
    hr = summary["hit_rate_at_k"]
    print(
        f"Hit-rate@k:           {hr:.2%}"
        if hr is not None
        else "Hit-rate@k:           N/A"
    )
    fa = summary["faithfulness"]
    print(
        f"Faithfulness:         {fa:.2%}"
        if fa is not None
        else "Faithfulness:         N/A"
    )
    nf = summary["not_found_accuracy"]
    print(
        f"Not-found accuracy:   {nf:.2%}"
        if nf is not None
        else "Not-found accuracy:   N/A"
    )
    print(f"Hallucinated citations total: {summary['hallucinated_citation_count']}")
    print(f"\nReport written to {REPORT_PATH}")

    if errored or not passes_thresholds(summary):
        print("\nEVAL FAILED — below threshold or errors occurred.")
        sys.exit(1)

    print("\nEVAL PASSED.")


if __name__ == "__main__":
    main()
