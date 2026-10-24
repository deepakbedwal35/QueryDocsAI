"""
evaluator.py - evaluation logic, separate from orchestration (main.py).

A RAG answer is CORRECT only if every applicable check passes, in order:

  1. refusal     : out-of-corpus question -> must say "cannot find the answer";
                   in-corpus question     -> must NOT refuse
  2. citations   : at least one citation, and none hallucinated
                   (reuses backend.generation.citation_verifier)
  3. grounding   : at least one valid citation points to the expected source PDF
                   (skipped when retrieval missed the source - the fact can live
                   in several papers, and retrieval is scored separately)
  4. correctness : every key-fact regex (keys_json) matches the answer
                   (strict, like value compare; token recall is a soft score only)

Citation formats: the verifier parses `[chunk: 28data_c79]`, but the current
SYSTEM_PROMPT asks for `[Document 28, Chunk 79]` (gpt-oss sometimes uses
`【...】` brackets). All are normalised to the verifier's format first.
<think>...</think> blocks from reasoning models are stripped.

Unicode hyphens/spaces (gpt-oss emits U+2011, U+202F) are normalised first.

Soft scores (citation_precision, answer_recall) are logged alongside the verdict.
"""

import json
import re
from collections import Counter

from backend.generation.citation_verifier import verify_citations

NOT_FOUND_PHRASE = "cannot find the answer"

_DOC_CHUNK = re.compile(r"[\[【]\s*Document\s+(\d+)\s*,\s*Chunk\s+(\d+)\s*[\]】]", re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_MALFORMED = re.compile(r"[\[【]\s*Document\b[^\]】]*[\]】]", re.IGNORECASE)
_ANY_TAG = re.compile(r"\[chunk:[^\]]*\]")
_UNICODE = str.maketrans({"‐": "-", "‑": "-", "–": "-", "—": "-",
                          " ": " ", " ": " "})
_STOP = set("""a an the of in on at to for and or is are was were be been by with as that this
which it its from into than then their there these those such can may not no""".split())


def normalise_citations(answer):
    """[Document 28, Chunk 79] -> [chunk: 28data_c79]"""
    return _DOC_CHUNK.sub(lambda m: f"[chunk: {m.group(1)}data_c{m.group(2)}]", answer)


def _tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def token_recall(pred, gold):
    """Fraction of the reference answer's tokens that appear in the prediction."""
    p, g = Counter(_tokens(pred)), Counter(_tokens(gold))
    if not g:
        return 0.0
    return sum((p & g).values()) / sum(g.values())


def evaluate_one(row, answer, chunks):
    """row: golden CSV row (dict-like). Returns dict(correct, reason, citation_precision, answer_recall)."""
    answer = normalise_citations(_THINK.sub("", answer or "").translate(_UNICODE)).strip()
    refused = NOT_FOUND_PHRASE in answer.lower()
    out = dict(correct=False, reason="", citation_precision=None, answer_recall=None)
    if not answer:
        out["reason"] = "empty_answer"
        return out

    # 1. refusal
    if str(row["not_found"]).upper() == "TRUE":
        out.update(correct=refused, reason="refusal_ok" if refused else "missed_refusal")
        return out
    if refused:
        out["reason"] = "false_refusal"
        return out

    # 2. citations
    v = verify_citations(answer, chunks)
    valid, bad = v["valid_citations"], v["hallucinated_citations"]
    n = len(valid) + len(bad)
    out["citation_precision"] = len(valid) / n if n else 0.0
    out["answer_recall"] = round(token_recall(_ANY_TAG.sub("", answer), row["expected_answer"]), 4)
    if n == 0:
        out["reason"] = "malformed_citation" if _MALFORMED.search(answer) else "no_citation"
        return out
    if bad:
        out["reason"] = "hallucinated_citation"
        return out

    # 3. grounding (only checkable when retrieval surfaced the source)
    if str(row["retrieval_hit"]).upper() == "TRUE" and \
            not any(c.startswith(f"{row['source']}_c") for c in valid):
        out["reason"] = "wrong_source"
        return out

    # 4. correctness
    text = _ANY_TAG.sub("", answer)
    if not all(re.search(k, text, re.IGNORECASE) for k in json.loads(row["keys_json"] or "[]")):
        out["reason"] = "missing_key_fact"
        return out

    out.update(correct=True, reason="match")
    return out
