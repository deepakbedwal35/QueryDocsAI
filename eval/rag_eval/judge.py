"""
judge.py - LLM-as-judge scores on top of the deterministic checks in evaluator.py.

  faithfulness     : share of the answer's claims supported by the retrieved context (0-1)
  answer_relevancy : how directly the answer addresses the question (0-1)

One Groq call per answer returns both scores as JSON. Refusals are not judged.
Pick a judge that is not the model under test (JUDGE_MODEL env var) to avoid
self-preference bias.
"""
import json
import os

from langfuse import observe

from backend.generation.groq_client import _get_client
from backend.generation.prompt_templates import format_context

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen/qwen3.8-27b")

_PROMPT = """You grade answers from a retrieval-augmented QA system.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
{answer}

Score two things, each from 0.0 to 1.0:
- faithfulness: split the answer into factual claims (ignore [chunk: ...] tags).
  Score = supported claims / total claims, where a claim is supported only if the
  RETRIEVED CONTEXT states or directly implies it.
- answer_relevancy: 1.0 if the answer directly and completely addresses the
  question, lower for partial, padded or off-topic answers. Judge relevancy
  independently of correctness: an on-topic answer that is factually wrong is
  still relevant (its wrongness belongs in faithfulness).

Reply with JSON only: {{"faithfulness": <float>, "answer_relevancy": <float>, "reason": "<one sentence>"}}"""


@observe(name="judge", as_type="evaluator")
def judge(question, answer, chunks):
    response = _get_client().chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        max_tokens=2000,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": _PROMPT.format(
            question=question, context=format_context(chunks), answer=answer)}],
    )
    out = json.loads(response.choices[0].message.content)
    return {k: round(min(max(float(out[k]), 0.0), 1.0), 3)
            for k in ("faithfulness", "answer_relevancy")}
