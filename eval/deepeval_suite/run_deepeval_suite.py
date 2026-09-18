"""
eval/deepeval_suite/run_deepeval_suite.py

Second offline eval suite, running alongside (not replacing)
eval/run_eval.py. Where run_eval.py checks retrieval hit-rate and a
citation-presence-based faithfulness score (cheap, deterministic,
reference-based), this suite fills the RAG-Triad gap that citation
checking can't: does a cited chunk's text actually ENTAIL the claim
next to it, and are the retrieved chunks relevant to the question at
all, independent of what the generator did with them.

Metrics (DeepEval, LLM-as-judge):
  - ContextualRelevancyMetric: are the retrieved chunks relevant to
    the input question? (the RAG-Triad box run_eval.py doesn't cover)
  - FaithfulnessMetric: does the generated answer's claims actually
    follow from the retrieved context? (judge-based upgrade path over
    citation_verifier.py's presence-only check)
  - AnswerRelevancyMetric: does the answer actually address the
    question asked, independent of grounding?

Gating mirrors run_eval.py's pattern deliberately (absolute thresholds
+ manual --write-baseline + regression-vs-baseline diff) so both
suites behave the same way in CI and in your head — same shape,
different metrics, kept as two files instead of merged into one so a
DeepEval dependency issue never blocks the cheaper citation-based
suite from running.

Run from the repo root:

    python -m eval.deepeval_suite.run_deepeval_suite
    python -m eval.deepeval_suite.run_deepeval_suite --write-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from backend.generation.groq_client import generate_answer
from backend.retrieval.hybrid import retrieve

HERE = Path(__file__).parent
DATASET_PATH = HERE / "golden_dataset.json"
REPORT_PATH = HERE.parent / "results" / "deepeval_report.json"
BASELINE_PATH = HERE.parent / "results" / "deepeval_baseline.json"

# Absolute floors -- same idea as run_eval.py's HIT_RATE_THRESHOLD etc.
# Tune these against your own golden set's actual score distribution,
# not against these placeholder numbers.
MIN_CONTEXTUAL_RELEVANCY = 0.6
MIN_FAITHFULNESS = 0.7
MIN_ANSWER_RELEVANCY = 0.7

REGRESSION_TOLERANCE = 0.03


def load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text())


def build_test_case(item: dict) -> tuple[LLMTestCase, list[dict]]:
    """Runs the REAL pipeline (retrieve + generate) for the question,
    rather than only replaying the hand-picked golden context — this
    is what makes it an end-to-end pipeline eval (RAG-Triad "Pipeline"
    level) instead of a component-only test. `retrieval_context` in
    the golden dataset is used as the judge's reference for what an
    ideal retrieval would have looked like, not fed to the generator."""
    chunks = retrieve(item["input"], chat_id=None)
    generation = generate_answer(item["input"], chunks)

    test_case = LLMTestCase(
        input=item["input"],
        actual_output=generation.text,
        expected_output=item.get("expected_output"),
        retrieval_context=[c.get("text", "") for c in chunks],
    )
    return test_case, chunks


def load_baseline() -> dict | None:
    if not BASELINE_PATH.exists():
        return None
    return json.loads(BASELINE_PATH.read_text())


def check_regression(summary: dict, baseline: dict | None) -> list[str]:
    if baseline is None:
        return []
    problems = []
    for metric in ("avg_contextual_relevancy", "avg_faithfulness", "avg_answer_relevancy"):
        current, prior = summary.get(metric), baseline.get(metric)
        if current is None or prior is None:
            continue
        if current < prior - REGRESSION_TOLERANCE:
            problems.append(
                f"{metric} regressed: {prior:.2f} (baseline) -> {current:.2f} (current)"
            )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DeepEval RAG-Triad suite.")
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset()
    contextual_metric = ContextualRelevancyMetric(threshold=MIN_CONTEXTUAL_RELEVANCY)
    faithfulness_metric = FaithfulnessMetric(threshold=MIN_FAITHFULNESS)
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=MIN_ANSWER_RELEVANCY)

    results = []
    for item in dataset:
        test_case, chunks = build_test_case(item)
        contextual_metric.measure(test_case)
        faithfulness_metric.measure(test_case)
        answer_relevancy_metric.measure(test_case)

        results.append(
            {
                "input": item["input"],
                "contextual_relevancy": contextual_metric.score,
                "faithfulness": faithfulness_metric.score,
                "answer_relevancy": answer_relevancy_metric.score,
                "retrieved_chunk_ids": [c.get("chunk_id") for c in chunks],
            }
        )

    def _avg(key: str) -> float | None:
        vals = [r[key] for r in results if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "avg_contextual_relevancy": _avg("contextual_relevancy"),
        "avg_faithfulness": _avg("faithfulness"),
        "avg_answer_relevancy": _avg("answer_relevancy"),
        "n_questions": len(results),
    }

    baseline = load_baseline()
    regressions = check_regression(summary, baseline)

    passed_thresholds = (
        (summary["avg_contextual_relevancy"] or 0) >= MIN_CONTEXTUAL_RELEVANCY
        and (summary["avg_faithfulness"] or 0) >= MIN_FAITHFULNESS
        and (summary["avg_answer_relevancy"] or 0) >= MIN_ANSWER_RELEVANCY
    )
    passed = passed_thresholds and not regressions

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps({"summary": summary, "regressions": regressions, "results": results}, indent=2)
    )

    print(json.dumps(summary, indent=2))
    if baseline is None:
        print("No baseline found -- run with --write-baseline once satisfied with a result.")
    elif regressions:
        print("Regressions vs. baseline:")
        for r in regressions:
            print(f"  - {r}")
    else:
        print("No regressions vs. baseline.")

    if not passed:
        print("DEEPEVAL SUITE FAILED.")
        sys.exit(1)

    print("DEEPEVAL SUITE PASSED.")
    if args.write_baseline:
        BASELINE_PATH.write_text(json.dumps(summary, indent=2))
        print(f"Baseline updated at {BASELINE_PATH}")


if __name__ == "__main__":
    main()
