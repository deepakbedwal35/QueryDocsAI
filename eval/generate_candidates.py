"""
eval/generate_candidates.py

Run from the project root:
    python -m eval.generate_candidates

Runs a bank of draft questions through your REAL retrieve() and
prints, for each, the top-5 paper titles that came back. Review the
output and use it to fill in `expected_paper_titles` in
test_questions.json -- this avoids hand-guessing chunk_ids or paper
titles that may not actually exist in your corpus.

Also writes eval/candidate_review.json with the same info, in case
it's easier to edit as a file than copy from terminal output.
"""

from __future__ import annotations

import json
from pathlib import Path

from real_retrieval import retrieve

# Draft questions spanning consciousness-theory topics your corpus
# likely covers (GWT, NCC, IIT, higher-order theories, anesthesia,
# machine consciousness, etc.) -- adjust/add your own freely.
CANDIDATE_QUESTIONS = [
    "How does Global Workspace Theory explain sustained conscious processing?",
    "What is the relationship between the prefrontal cortex and Global Workspace Theory?",
    "What is Integrated Information Theory?",
    "What are the neural correlates of consciousness?",
    "How does anesthesia affect conscious awareness?",
    "What triggers explicit awareness in implicit sequence learning?",
    "How does the 3D default space model explain conscious experience?",
    "What is Higher-Order Theory of consciousness?",
    "How does Recurrent Processing Theory explain perceptual consciousness?",
    "Can artificial intelligence systems be conscious?",
    "What role does the thalamus play in consciousness?",
    "How do different theories of consciousness compare in explaining sustained awareness?",
    "What is the global neuronal workspace?",
    "How is machine consciousness modeled using deep learning?",
    "What is meta-conscious processing?",
    "What are the functional contributions of consciousness?",
    "How does emotional intensity affect conscious processing?",
    "What is the ignition marker of conscious experience in the prefrontal cortex?",
    "How does phenomenology relate to the concept of consciousness?",
    "What is the relationship between spirituality and consciousness research?",
]

# Off-corpus negative controls (should trigger "not found")
NEGATIVE_QUESTIONS = [
    "Who won the most recent Super Bowl?",
    "What is the boiling point of water at sea level?",
    "What's the best recipe for chocolate chip cookies?",
    "What is the current stock price of Apple?",
]


def main():
    review = []

    print("=" * 70)
    print("POSITIVE QUESTIONS (should retrieve relevant papers)")
    print("=" * 70)
    for q in CANDIDATE_QUESTIONS:
        results = retrieve(q, top_k=5)
        titles = []
        seen = set()
        for r in results:
            if r["paper_title"] not in seen:
                titles.append(r["paper_title"])
                seen.add(r["paper_title"])

        print(f"\nQ: {q}")
        for t in titles:
            print(f"   - {t}")

        review.append(
            {
                "question": q,
                "expected_paper_titles": titles,  # pre-filled with what retrieval found;
                "expect_not_found": False,  # edit this list down to what's ACTUALLY correct
            }
        )

    print("\n" + "=" * 70)
    print("NEGATIVE QUESTIONS (should say 'cannot find the answer')")
    print("=" * 70)
    for q in NEGATIVE_QUESTIONS:
        print(f"\nQ: {q}")
        review.append(
            {
                "question": q,
                "expected_paper_titles": [],
                "expect_not_found": True,
            }
        )

    output_path = Path(__file__).parent / "candidate_review.json"
    output_path.write_text(json.dumps(review, indent=2))
    print(f"\n\nWritten to {output_path}")
    print(
        "Review each question's 'expected_paper_titles' list -- keep only the "
        "titles that are ACTUALLY correct for that question, delete the rest. "
        "Then hand this file to me (or run the conversion script) to produce "
        "the final test_questions.json."
    )


if __name__ == "__main__":
    main()
