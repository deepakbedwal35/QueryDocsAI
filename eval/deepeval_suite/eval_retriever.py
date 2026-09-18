"""
eval/deepeval_suite/eval_retriever.py

Retrieval evaluation using:
- Contextual Recall
- Contextual Precision
- Custom OpenRouter Wrapper (Fixes AttributeError)
- Confident AI for evaluation tracking
"""
import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase


# --------------------------------------------------
# ENV
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

GOLDEN_PATH = PROJECT_ROOT / "eval/goldens/retriever_golden_dataset.json"
CACHE_PATH = PROJECT_ROOT / "eval/goldens/retrieval_cache.json"

JUDGE_MODEL = "google/gemini-2.5-flash:free"
THRESHOLD = 0.7

# --------------------------------------------------
# CUSTOM OPENROUTER CLASS WRAPPER
# --------------------------------------------------

class OpenRouterJudgeModel(DeepEvalBaseLLM):
    """
    Custom wrapper to safely manage OpenRouter endpoints within DeepEval,
    preventing json schema mapping response parsing failures.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        # self.client = OpenAI(
        #     base_url="https://openrouter.ai",
        #     api_key="..INSERT API KEY HERE WE YOU HAVE MONEY",
        # )

    def load_model(self):
        return self.client

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str, schema: BaseModel = None) -> str:
        # Fallback for synchronous execution steps if required
        return asyncio.run(self.a_generate(prompt, schema))

    async def a_generate(self, prompt: str, schema: BaseModel = None) -> str:
        # Construct parameters compatible with OpenRouter
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        
        # Handle structured JSON schemas if DeepEval requests it
        if schema:
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["messages"][0]["content"] = (
                f"{prompt}\n\nIMPORTANT: You must respond with a valid JSON object matching this schema template: {schema.model_json_schema()}"
            )

        # Run non-blocking completion request
        loop = asyncio.get_event_loop()
        completion = await loop.run_in_executor(
            None, 
            lambda: self.client.chat.completions.create(**kwargs)
        )
        
        # Return string clean text explicitly avoiding structural AttributeError
        return completion.choices[0].message.content


# --------------------------------------------------
# DATA LOADING & INITIALIZATION
# --------------------------------------------------

# Instantiate our custom openrouter model handler safely
judge = OpenRouterJudgeModel(model_name=JUDGE_MODEL)

goldens = json.loads(GOLDEN_PATH.read_text())
retrieval_cache = json.loads(CACHE_PATH.read_text())

# BUILD TEST CASES
test_cases = []
for g in goldens:
    context = retrieval_cache.get(g["question"])

    if not context:
        print(f"Skipping: {g['question']}")
        continue

    test_cases.append(
        LLMTestCase(
            input=g["question"],
            expected_output=g["expected_answer"],
            retrieval_context=context,
            actual_output="(generator not evaluated)",
        )
    )

if not test_cases:
    raise RuntimeError("No test cases found in retrieval cache.")

print(f"Running {len(test_cases)} test cases...")

# METRICS
metrics = [
    ContextualRecallMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=True,
    ),
    ContextualPrecisionMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=True,
    ),
]

# EVALUATE
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(
        run_async=True,
        max_concurrent=1, # Strict low value protects against OpenRouter free tier rate cuts
        throttle_value=1,
    ),
    hyperparameters={
        "retriever": "base_k5",
        "embedding_model": "bge-base-en-v1.5",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "judge_model": JUDGE_MODEL,
        "golden_set": str(GOLDEN_PATH),
    },
)
