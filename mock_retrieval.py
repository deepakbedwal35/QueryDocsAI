"""
mock_retrieval.py

Stand-in for Person A's retrieval function. Returns fake but
correctly-shaped data matching the contract in PRD.md Section 2
("Retrieval response contract — Person A → Person B interface").

Usage:
    from mock_retrieval import retrieve
    chunks = retrieve("How does Global Workspace Theory explain consciousness?")

Once Person A's real retrieve() is ready, swap the import in ask.py —
no other code should need to change, since everything downstream is
built against this same shape.
"""

from __future__ import annotations

# A small fake "corpus" of chunks, spanning a couple of different
# papers/topics so you can sanity-check that filtering/relevance-ish
# behavior looks reasonable during dev.
_MOCK_CHUNKS: list[dict] = [
    {
        "chunk_id": "paper027d_chunk014",
        "text": (
            "Conscious processing involves broadcasting information to "
            "several brain regions (Global Workspace Theory). Input that "
            "elicits relatively intense emotions is subjected to highly "
            "sustainable conscious processing, allowing it to remain "
            "accessible over longer time windows."
        ),
        "paper_title": "How sustainable are different levels of consciousness",
        "authors": ["E. Wiersma"],
        "year": 2017,
        "page": 4,
        "score": 0.91,
    },
    {
        "chunk_id": "paper027d_chunk015",
        "text": (
            "In contrast, low-intensity stimuli are processed locally and "
            "do not achieve global broadcast, resulting in only transient "
            "or unconscious representations."
        ),
        "paper_title": "How sustainable are different levels of consciousness",
        "authors": ["E. Wiersma"],
        "year": 2017,
        "page": 5,
        "score": 0.78,
    },
    {
        "chunk_id": "paper19a3_chunk002",
        "text": (
            "The neural correlates of consciousness (NCC) framework "
            "proposes that specific patterns of cortical activity are "
            "both necessary and sufficient for a given conscious "
            "experience to occur."
        ),
        "paper_title": "Neural Correlates of Consciousness: A Review",
        "authors": ["C. Koch", "N. Tononi"],
        "year": 2016,
        "page": 2,
        "score": 0.85,
    },
    {
        "chunk_id": "paper19a3_chunk007",
        "text": (
            "Integrated Information Theory (IIT) offers an alternative "
            "account, positing that consciousness corresponds to a "
            "system's capacity to integrate information, quantified as "
            "phi."
        ),
        "paper_title": "Neural Correlates of Consciousness: A Review",
        "authors": ["C. Koch", "N. Tononi"],
        "year": 2016,
        "page": 6,
        "score": 0.73,
    },
    {
        "chunk_id": "paper5f2c_chunk001",
        "text": (
            "Anesthetic agents such as propofol disrupt frontoparietal "
            "connectivity, which correlates with loss of behavioral "
            "responsiveness and reported awareness."
        ),
        "paper_title": "Anesthesia and the Breakdown of Cortical Connectivity",
        "authors": ["M. Alkire"],
        "year": 2019,
        "page": 3,
        "score": 0.69,
    },
]


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """
    Fake retrieval: ignores the actual question and returns the mock
    corpus (or a keyword-filtered subset), shaped exactly like Person
    A's real retrieval output will be.

    Args:
        question: The user's natural-language question (unused for
            real relevance here — this is just a stub).
        top_k: Max number of chunks to return.

    Returns:
        A list of chunk dicts, each with:
            chunk_id, text, paper_title, authors, year, page, score
    """
    q = question.lower()

    # Very naive "relevance" so different questions can surface
    # different mock chunks during manual testing.
    keyword_map = {
        "global workspace": {"paper027d_chunk014", "paper027d_chunk015"},
        "sustain": {"paper027d_chunk014", "paper027d_chunk015"},
        "neural correlates": {"paper19a3_chunk002"},
        "ncc": {"paper19a3_chunk002"},
        "integrated information": {"paper19a3_chunk007"},
        "iit": {"paper19a3_chunk007"},
        "phi": {"paper19a3_chunk007"},
        "anesthesia": {"paper5f2c_chunk001"},
        "propofol": {"paper5f2c_chunk001"},
    }

    matched_ids: set[str] = set()
    for keyword, chunk_ids in keyword_map.items():
        if keyword in q:
            matched_ids |= chunk_ids

    if matched_ids:
        results = [c for c in _MOCK_CHUNKS if c["chunk_id"] in matched_ids]
    else:
        # No keyword match -> just return the full mock set (sorted by
        # score) so the pipeline still has something to work with.
        results = list(_MOCK_CHUNKS)

    results.sort(key=lambda c: c["score"], reverse=True)
    return results[:top_k]


def retrieve_empty(question: str, top_k: int = 5) -> list[dict]:
    """
    Secondary mock for testing the "not found in corpus" path —
    always returns an empty list, so you can verify /ask correctly
    produces answer_found=False and the "cannot find the answer"
    message end-to-end.
    """
    return []


if __name__ == "__main__":
    # Quick manual sanity check: python mock_retrieval.py
    import json

    test_questions = [
        "How does Global Workspace Theory explain sustained conscious processing?",
        "What is Integrated Information Theory?",
        "How does propofol affect the brain?",
        "What's the capital of France?",  # should fall back to full set
    ]

    for tq in test_questions:
        print(f"\nQ: {tq}")
        print(json.dumps(retrieve(tq), indent=2))
