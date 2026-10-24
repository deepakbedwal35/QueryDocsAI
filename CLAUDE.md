# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

QueryDocsAI is a RAG app for asking questions over a fixed corpus of research papers, plus PDFs that users upload into a chat. Answers must cite the chunks they came from. It has three parts: a FastAPI backend (`backend/`), a React + Vite + Tailwind v4 frontend (`frontend/`), and an offline indexing pipeline (`scripts/`). Qdrant runs in Docker.

## Commands

Run everything from the repo root, because Python imports use the `backend.` package prefix and data paths are relative (`DATA_DIR` defaults to `data`).

```bash
./start.sh                                    # Qdrant (docker compose) + uvicorn :8080 + vite dev
cd scripts && docker compose up -d            # Qdrant only (ports 6333/6334, storage in scripts/qdrant_storage)
uvicorn backend.main:app --reload --port 8080 # backend only

cd frontend && npm run dev | npm run build | npm run lint   # pnpm-lock.yaml also exists

pip install -r requirements.txt -r backend/requirements.txt  # pipeline deps + API deps (no deepeval)
```

Offline indexing pipeline: `scripts/01_fetch_papers.py` through `06_build_bm25_index.py`, with `07_running_pipeline.py` orchestrating steps 02–06. Step 01 is run manually. The notebooks in `scripts/notebook/` are the original versions of the same steps.

Evaluation (there are no unit tests). The primary suite is `eval/rag_eval/` (modelled on campusx `llm-sql-eval`). Retrieval is frozen into `golden_dataset.csv` once, then every model in `models.py` is scored on identical context with the deterministic checks in `evaluator.py` (refusal → citations → source grounding → token recall):

```bash
python -m eval.rag_eval.golden_dataset_generator          # validate GOLDEN sources
python -m eval.rag_eval.make_golden_dataset               # needs Qdrant; rebuild after re-indexing
python -m eval.rag_eval.main [--models "GPT-OSS 120B"] [--limit 5] [--threshold 0.6]   # -> eval/rag_eval/eval_results.csv
```

Older suite, still present:

```bash
python -m eval.deepeval_suite.eval_pipeline [--top-k 3] [--limit 5]   # retrieval hit-rate + citation faithfulness
python -m eval.deepeval_suite.run_deepeval_suite [--write-baseline]   # DeepEval LLM-as-judge (needs deepeval installed)
```

Many docstrings still refer to old module paths such as `eval.run_eval` and `eval.generate_candidates`. The real paths are under `eval/deepeval_suite/` and `eval/utils/`. `.github/workflows/eval.yml` runs `eval.rag_eval.main` and needs the committed `golden_dataset.csv`.

## Configuration

All configuration comes from the root `.env`. The key variables are:

- `GROQ_API_KEY` and `GROQ_MODEL` (default `openai/gpt-oss-120b`), loaded in `backend/config.py`.
- `DATABASE_URL`, which is required. `backend/db/session.py` calls `.startswith` on it and crashes if it is unset. It is a Postgres/Supabase URL in practice, though SQLite also works.
- `QDRANT_HOST` and `QDRANT_PORT`.
- `VITE_API_BASE_URL` for the frontend (default `http://localhost:8080`).

Tables are created with `create_all` at startup. Alembic is listed as a dependency but is not set up.

## Architecture

**Request flow for `POST /ask`** (`backend/routes/ask.py`): retrieve → generate → verify citations → save both messages → respond. The chat is also auto-titled from its first question.

**Identity:** there is no login. The frontend creates a UUID in localStorage (`frontend/src/api/deviceId.js`) and `apiFetch` (`frontend/src/api/client.js`) sends it as `X-Device-Id` on every request. `backend/db/deps.py:get_current_device` creates the device row the first time it sees an ID. Data ownership runs Device → Chat → Message/Document, with cascade deletes (`backend/db/models.py`). Citations are stored as a JSON column on `Message`. The ownership checks in `routes/documents.py` upload and list are currently commented out.

**Hybrid retrieval** (`backend/retrieval/hybrid.py`, which is the live module; the root `real_retrieval.py` is an older copy):
- Dense: Qdrant collection `docs_collections` with `BAAI/bge-base-en-v1.5`. Queries need `BGE_QUERY_PREFIX`; passages do not.
- Sparse: a static BM25 pickle (`data/bm25_index.pkl`) plus `data/mapping_table.pkl` (chunk_id ↔ text). BM25 only covers the corpus and never includes uploads.
- The two result lists are fused with RRF (k=60, a pool of 20 from each). When `chat_id` is given, uploaded chunks are swapped into the top_k, because their missing BM25 score would otherwise always rank them low.
- Every Qdrant point ID is `uuid5(NAMESPACE_DNS, chunk_id)`. Corpus chunks are mapped back to text through the mapping table. Uploaded chunks keep `chunk_id` and `text` in the payload.
- Chunk ID formats: corpus chunks are `<pdf_stem>_c<n>` and resolve to paper metadata via `data/pdf_to_paperid.json` → `data/metadata.csv` (the authors column is a Python repr, so it is parsed with `ast.literal_eval`). Uploaded chunks are `upload_<doc_id>_c<n>`.
- Qdrant payload `source` is either `corpus` or `upload`. On the first call, `_ensure_corpus_payload` backfills `source=corpus` onto every point. The search filter is `source=corpus OR chat_id=<this chat>`.
- `page` is always `None`, because the pipeline joins all pages together before chunking.

**Uploads** (`backend/ingestion/pipeline.py` + `qdrant_utils.py`): PyMuPDF → `RecursiveCharacterTextSplitter` (1200 characters, 250 overlap) → BGE embeddings → upsert to the same Qdrant collection. Deleting a document removes its Qdrant points first, then the DB row.

**Generation and citation contract:** `backend/generation/prompt_templates.py` tells the LLM to cite with `[chunk: <id>]` tags or reply exactly `I cannot find the answer in the provided documents.` `citation_verifier.py` keeps tags whose IDs were actually retrieved and strips the rest as hallucinated. `ask.py` sets `answer_found` by checking for the phrase "cannot find the answer". If you change the prompt wording, keep the verifier and `ask.py` in sync with it.

**Frontend:** `src/api/*Api.js` wrap `apiFetch`, and `src/hooks/use{Chats,Messages,Documents}.js` hold the state. `App.jsx` composes the chat-history sidebar, chat window, and documents sidebar. The `@` alias points to `src/`.

**Eval data** lives in `eval/goldens/` (golden datasets, a retrieval cache, and CI smoke questions). Reports and baselines are written to `eval/results/`.

## Gotchas

- `scripts/qdrant_storage/` and the `data/` indexes are committed to git, so running Qdrant locally makes the working tree dirty. Don't commit those changes unless you mean to.
- Both the embedding model and the BM25 index are loaded lazily inside each module (`hybrid.py` and `ingestion/pipeline.py` each have their own singleton). The first request is slow.
