# Ask My Papers — PRD & TRD

---

# Part 1: Product Requirements Document (PRD)

## 1. Overview

**Product name:** Ask My Papers
**One-liner:** A RAG (Retrieval-Augmented Generation) chatbot that answers questions about a curated set of neuroscience/consciousness research papers, grounding every claim in cited source passages.

**Problem it solves:** Reading dozens of research papers to answer a specific question is slow. General-purpose LLM chatbots can answer instantly but may hallucinate or use outdated/generic knowledge instead of the actual papers. Ask My Papers gives fast answers that are traceable back to real source text — or honestly says "not found" when the corpus doesn't cover it.

**Analogy:** Same core mechanism as Google NotebookLM (grounded Q&A with citations over a fixed document set), scoped down to a single research domain and built end-to-end to understand every layer.

## 2. Goals

- Answer natural-language questions about consciousness/neuroscience research using only a curated paper corpus.
- Every answer must cite the specific paper + location it came from.
- If the corpus doesn't contain the answer, say so explicitly — never fall back to general knowledge.
- Measurably better than naive "dump text into an LLM" — proven via a retrieval/faithfulness eval suite.

## 3. Non-Goals (v1)

- No arbitrary user-uploaded PDFs (fixed corpus only — see v2 below).
- No multi-turn conversational memory required (v1 is single-question-in, single-answer-out; can add chat history later).
- No authentication/user accounts.
- No support for non-English papers.

## 4. Target User

You (and your teammate) as builders/demoers — this is a portfolio project. Secondary "user" persona: someone researching consciousness/neuroscience topics who wants quick, trustworthy, cited answers instead of reading full papers.

## 5. Core User Stories

| #   | As a user, I want to...                                      | So that...                                  |
| --- | ------------------------------------------------------------ | ------------------------------------------- |
| 1   | Type a question into a search/chat box                       | I get an answer without reading full papers |
| 2   | See citations next to each claim in the answer               | I can verify the answer is accurate         |
| 3   | Click a citation to see the exact source excerpt             | I can trust (or fact-check) the answer      |
| 4   | Get told "not found in the provided papers" when appropriate | I don't get a confidently wrong answer      |
| 5   | See the answer stream/load quickly                           | The tool feels responsive, not sluggish     |

## 6. Functional Requirements

- **FR1:** System ingests ~50–200 papers on a focused sub-topic (e.g. neural correlates of consciousness) via Semantic Scholar/arXiv/PubMed APIs.
- **FR2:** System chunks papers by section/paragraph, embeds chunks, and stores them in Qdrant with metadata.
- **FR3:** Given a question, system retrieves relevant chunks using hybrid (dense + BM25) search, reranked by a cross-encoder.
- **FR4:** System generates an answer via Groq LLM using **only** retrieved chunks as context, with inline citations tied to chunk IDs.
- **FR5:** System verifies every citation in the generated answer actually corresponds to a retrieved chunk; strips/flags hallucinated citations.
- **FR6:** Frontend displays the answer with expandable citation cards showing the source excerpt, paper title, and metadata.
- **FR7:** An evaluation suite (30–50 Q&A pairs) scores retrieval hit-rate and answer faithfulness, runnable via CI.

## 7. Success Metrics

- **Retrieval hit-rate@5** ≥ 80% on the eval set (correct chunk appears in top 5 retrieved).
- **Faithfulness** ≥ 90% (answer claims are traceable to retrieved chunks, no hallucinated citations pass the verifier).
- **Latency:** end-to-end `/ask` response under ~5–8 seconds (CPU-based reranking + Groq generation).
- Demonstrable **before/after** improvement from naive dense-only retrieval → hybrid + rerank (qualitative, from your 5–10 saved comparison queries).

## 8. V2 / Future Scope

- Dynamic ingestion: users upload their own PDF, queried in an isolated session (not merged into the curated corpus).
- Multi-turn conversation with memory.
- Multi-domain corpora (swap the paper set).

---

# Part 2: Technical Requirements Document (TRD)

## 1. System Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────────────┐
│   React     │─────▶│   FastAPI    │─────▶│  Retrieval Layer         │
│  Frontend   │◀─────│  /ask route  │◀─────│  (Qdrant + BM25 + Rerank)│
└─────────────┘      └──────┬───────┘      └─────────────────────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │  Groq LLM    │
                      │  Generation  │
                      └──────────────┘
```

**Offline pipeline (run once / re-run on corpus update):**

```
Semantic Scholar/arXiv API → PDF download → PyMuPDF extraction →
chunking → embeddings → Qdrant index + BM25 index (saved to disk)
```

## 2. Data Model

### `metadata.csv` (per paper)

| field          | type   | notes                      |
| -------------- | ------ | -------------------------- |
| paper_id       | string | Semantic Scholar `paperId` |
| title          | string |                            |
| authors        | string | comma-separated            |
| year           | int    |                            |
| doi            | string |                            |
| pdf_url        | string | `openAccessPdf.url`        |
| citation_count | int    |                            |
| filename       | string | local PDF filename         |

### Chunk object (`chunks.jsonl`, and Qdrant payload)

```json
{
  "chunk_id": "paper027d_chunk014",
  "paper_id": "027d70631ba117229e54638fd373411ad2bacdea",
  "paper_title": "How sustainable are different levels of consciousness",
  "authors": ["E. Wiersma"],
  "year": 2017,
  "section": "discussion",
  "page": 4,
  "text": "...",
  "embedding": [0.021, -0.114, ...]
}
```

### Retrieval response contract (Person A → Person B interface)

```json
[
  {
    "chunk_id": "paper027d_chunk014",
    "text": "...",
    "paper_title": "How sustainable are different levels of consciousness",
    "authors": ["E. Wiersma"],
    "year": 2017,
    "page": 4,
    "score": 0.87
  }
]
```

## 3. API Design

### `POST /ask`

**Request:**

```json
{
  "question": "How does the Global Workspace theory explain sustained conscious processing?"
}
```

**Response:**

```json
{
  "answer": "According to Wiersma (2017), sustained conscious processing occurs when input elicits relatively intense emotions [chunk: paper027d_chunk014]...",
  "citations": [
    {
      "chunk_id": "paper027d_chunk014",
      "paper_title": "How sustainable are different levels of consciousness",
      "authors": ["E. Wiersma"],
      "year": 2017,
      "page": 4,
      "excerpt": "input that elicits relatively intense emotions is subjected to highly sustainable conscious processing..."
    }
  ],
  "answer_found": true
}
```

### `GET /health`

Simple liveness check — returns `{ "status": "ok" }`.

### `GET /papers` _(optional, nice-to-have)_

Returns the list of indexed papers (title, authors, year) for display in the UI ("what's in this corpus").

## 4. Prompt Contract (System Prompt for Groq)

```
You are a research assistant that answers questions using ONLY the provided
context chunks. Each chunk has a chunk_id.

Rules:
1. Answer only from the given context. Do not use outside knowledge.
2. Every factual claim must include an inline citation referencing the
   chunk_id it came from, e.g. [chunk: paper027d_chunk014].
3. If the answer is not contained in the provided context, respond exactly:
   "I cannot find the answer in the provided documents."
4. Do not fabricate chunk_ids. Only cite chunk_ids that appear in the context.
```

## 5. Citation Verification Logic (Person B owns this)

1. Parse the LLM's answer for `[chunk: <id>]` patterns (regex).
2. For each extracted chunk_id, check it exists in the retrieved-chunks list passed into that request.
3. Any chunk_id not found in that list → flag as hallucinated; strip the citation tag or annotate the answer with a warning.
4. Log verification results per request (useful for the eval suite's faithfulness score).

## 6. Non-Functional Requirements

- Reranker + embedding models run on CPU — no GPU dependency.
- Qdrant Cloud free tier or local Docker for dev.
- All secrets (`GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`) via `.env`, never committed.
- CI (GitHub Actions) runs eval suite on PRs; fails if hit-rate or faithfulness drops below threshold.

---

# Part 3: Your Guide (Person B — Generation + Product)

You own everything **after** retrieval returns chunks, plus the product surface (API + frontend + eval). You do **not** need to touch Qdrant indexing, embeddings, or BM25 internals — you just call Person A's retrieval function and trust its output matches the contract above.

## Step-by-step build order

### Step 0 — Agree on the contract (do this first, with Person A)

- Confirm the exact JSON shape retrieval will return (see Section 2 above).
- Ask Person A for a **stub/mock function** early: `def retrieve(question: str) -> List[dict]` that returns fake but correctly-shaped data. This unblocks you immediately — you don't wait for their pipeline to be done.

```python
# mock_retrieval.py — use until Person A's real retrieval is ready
def retrieve(question: str, top_k: int = 5) -> list[dict]:
    return [
        {
            "chunk_id": "mock_001",
            "text": "Conscious processing involves broadcasting information to several brain regions (Global Workspace Theory).",
            "paper_title": "How sustainable are different levels of consciousness",
            "authors": ["E. Wiersma"],
            "year": 2017,
            "page": 4,
            "score": 0.91
        }
    ]
```

### Step 1 — Groq client wrapper (`backend/generation/groq_client.py`)

- Reuse your FinScope pattern.
- Function signature: `def generate_answer(question: str, chunks: list[dict]) -> str`
- Build the prompt: system prompt (Section 4 above) + formatted chunks (include `chunk_id`, `paper_title`, `text` for each) + the user's question.
- Call Groq's chat completion endpoint, return raw text.

### Step 2 — Prompt templates (`backend/generation/prompt_templates.py`)

- Store the system prompt as a constant.
- Write a `format_context(chunks: list[dict]) -> str` helper that turns the chunk list into a numbered block like:

```
[chunk_id: paper027d_chunk014] (Wiersma, 2017, p.4)
"input that elicits relatively intense emotions is subjected to highly sustainable conscious processing..."
```

### Step 3 — Citation verifier (`backend/generation/citation_verifier.py`)

- `def verify_citations(answer: str, retrieved_chunks: list[dict]) -> dict`
- Regex-extract `[chunk: <id>]` tags from the answer.
- Cross-check against the set of retrieved `chunk_id`s.
- Return `{ "valid_citations": [...], "hallucinated_citations": [...], "clean_answer": "..." }`.

### Step 4 — Pydantic schemas (`backend/models/schemas.py`)

```python
class AskRequest(BaseModel):
    question: str

class Citation(BaseModel):
    chunk_id: str
    paper_title: str
    authors: list[str]
    year: int
    page: int | None = None
    excerpt: str

class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    answer_found: bool
```

### Step 5 — FastAPI route (`backend/routes/ask.py`)

```python
@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    chunks = retrieve(req.question)          # Person A's function (or your mock)
    raw_answer = generate_answer(req.question, chunks)
    result = verify_citations(raw_answer, chunks)
    citations = build_citation_list(result["valid_citations"], chunks)
    answer_found = "cannot find the answer" not in result["clean_answer"].lower()
    return AskResponse(
        answer=result["clean_answer"],
        citations=citations,
        answer_found=answer_found
    )
```

- Wire this router into `backend/main.py`.
- Add `GET /health` for a trivial sanity check.

### Step 6 — Test the backend in isolation

- Run FastAPI locally (`uvicorn backend.main:app --reload`).
- Hit `/ask` with curl/Postman using your mock retrieval — confirm the response shape matches the contract before touching the frontend.

### Step 7 — Eval suite (`eval/`)

- Write `eval/test_questions.json`: 30–50 entries, each with a question + the expected chunk_id(s)/paper(s) that should be retrieved.

```json
[
  {
    "question": "How does Global Workspace Theory explain sustained conscious processing?",
    "expected_chunk_ids": ["paper027d_chunk014"],
    "expected_paper": "How sustainable are different levels of consciousness"
  }
]
```

- `eval/run_eval.py`:
  - For each question: call `/ask` (or the retrieval + generation functions directly).
  - **Hit-rate@k:** did any `expected_chunk_ids` appear in the retrieved set?
  - **Faithfulness:** did the answer's citations all pass `verify_citations` (no hallucinated ones)?
  - Write results to `eval/results/eval_report.json` with aggregate scores.
- Wire into GitHub Actions: run `run_eval.py` on PR, fail the build if scores drop below a threshold you set (e.g. hit-rate < 75%).

### Step 8 — Frontend (`frontend/`)

Build in this order so you always have something visible:

1. **`App.jsx`** — page shell, holds the question input + submit handler + response state.
2. **`api/askApi.js`** — one function: `askQuestion(question) => fetch('/ask', {...})`.
3. **`ChatWindow.jsx`** — renders the conversation (or just the latest Q + A for v1).
4. **`MessageBubble.jsx`** — renders the answer text, with citation markers rendered inline or as footnote numbers.
5. **`CitationCard.jsx`** — expandable card: paper title, authors, year, page, and the excerpt text. Click to expand/collapse.
6. **`PaperSourceBadge.jsx`** — small badge/tag per citation (e.g. "Wiersma 2017") shown near the relevant claim, links to the matching `CitationCard`.
7. Loading state while `/ask` is in flight; empty state before first question; "not found" state styled distinctly (e.g. muted/gray) when `answer_found: false`.

### Step 9 — Integrate with real retrieval

- Swap your mock `retrieve()` for Person A's real function once it's ready — no other code should need to change, since you built to the contract.
- Re-run the eval suite against the real pipeline to get your actual hit-rate/faithfulness numbers.

### Step 10 — Deploy

- Backend → Render (same pattern as FinScope).
- Frontend → Render static site.
- Confirm `.env` values (`GROQ_API_KEY`, backend URL for frontend) are set in Render's environment config, not committed.

## Your File Checklist

```
backend/
├── main.py                      ← you
├── config.py                    ← you
├── generation/
│   ├── prompt_templates.py      ← you
│   ├── groq_client.py           ← you
│   └── citation_verifier.py     ← you
├── models/
│   └── schemas.py                ← you
├── routes/
│   └── ask.py                    ← you

eval/
├── test_questions.json           ← you (ideally co-written with Person A)
├── run_eval.py                   ← you
└── results/eval_report.json      ← generated

frontend/                         ← all you
├── src/App.jsx
├── src/api/askApi.js
├── src/components/ChatWindow.jsx
├── src/components/MessageBubble.jsx
├── src/components/CitationCard.jsx
└── src/components/PaperSourceBadge.jsx

.github/workflows/eval.yml        ← you (CI wiring)
```

## Order-of-operations summary

1. Lock the retrieval contract with Person A + get a mock function.
2. Build Groq client + prompt templates.
3. Build citation verifier.
4. Build `/ask` endpoint with schemas, test with mock data.
5. Build eval suite against mock data (validates your harness works).
6. Build frontend against the working mock-backed API.
7. Swap in real retrieval, re-run eval for real numbers.
8. Wire CI, deploy.
