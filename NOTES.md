use this api key for extract data/pdf :

https://www.semanticscholar.org/product/api/tutorial 


**The exact system you just mapped out is called RAG (Retrieval-Augmented Generation)** 
with an advanced Reranking and Citation pipeline.
Here is exactly how your description maps to how a PDF-based AI application works under the hood:
How Your Description Maps to Code
**"Reads all your papers once and organizes them..."**
*What happens:* The user uploads their PDFs. The system extracts the text, breaks it into small chunks (paragraphs), converts them into math vectors using a sentence transformer, and saves them in a Vector Database.
**"When you ask a question, it first finds the actual passages..."**
*What happens:* This is Retrieval. The system converts the user's question into a vector and pulls the top 10 or 20 paragraphs from the database that look conceptually similar.
"Then it double-checks itself — it picks out only the ones that truly answer your question...*"What happens:* This is called Reranking. A specialized AI model (like a Cross-Encoder) scores the retrieved paragraphs to filter out "keyword matches" and keeps only the chunks that actually possess the semantic answer.
**"Then it writes you an answer, but every claim points back to..."**
*What happens:* The system passes the final, clean paragraphs to a Large Language Model (LLM) along with strict instructions: "Answer the prompt using ONLY this context, and append the metadata source (Filename, Page Number) to the end of your sentences.
**""If your papers don't actually contain the answer, it tells you that honestly..."**
*What happens:* This is Hallucination Prevention. The prompt explicitly tells the LLM: "If the answer is not in the text, reply with 'I cannot find the answer in the provided documents.' Do not use outside training data."



ask-my-papers/
├── README.md
├── .env                          # QDRANT_URL, GROQ_API_KEY, etc.
├── .gitignore
│
├── data/
│   ├── raw_pdfs/                 # downloaded paper PDFs, untouched
│   ├── metadata.csv              # title, authors, year, source_url, filename
│   └── processed/
│       └── chunks.jsonl          # extracted + chunked text, one JSON per line
│
├── scripts/                      # one-off / re-runnable pipeline scripts
│   ├── 01_fetch_papers.py        # pulls PDFs from PubMed/arXiv/bioRxiv APIs
│   ├── 02_extract_text.py        # PyMuPDF: PDF → clean text
│   ├── 03_chunk_text.py          # RecursiveCharacterTextSplitter → chunks.jsonl
│   ├── 04_build_embeddings.py    # sentence-transformers → embeddings
│   ├── 05_index_qdrant.py        # push embeddings + metadata into Qdrant
│   ├── 06_build_bm25_index.py    # rank_bm25 sparse index, saved to disk
│   └── run_pipeline.py           # runs 01–06 in order
│
├── backend/
│   ├── main.py                   # FastAPI app entrypoint
│   ├── config.py                 # env vars, model names, constants
│   ├── requirements.txt
│   │
│   ├── retrieval/
│   │   ├── dense.py              # Qdrant query logic
│   │   ├── sparse.py             # BM25 query logic
│   │   ├── fusion.py             # Reciprocal Rank Fusion
│   │   └── reranker.py           # cross-encoder reranking
│   │
│   ├── generation/
│   │   ├── prompt_templates.py   # system prompt, citation-format instructions
│   │   └── groq_client.py        # Groq API wrapper (reuse FinScope's pattern)
│   │
│   ├── models/
│   │   └── schemas.py            # Pydantic request/response models
│   │
│   ├── routes/
│   │   └── ask.py                # POST /ask endpoint — ties retrieval + generation
│   │
│   └── utils/
│       └── pdf_utils.py          # shared PyMuPDF helpers (used by scripts too)
│
├── eval/
│   ├── test_questions.json       # your 30–50 hand-written Q&A pairs
│   ├── run_eval.py                # runs test set through the pipeline
│   └── results/
│       └── eval_report.json       # hit-rate@k, faithfulness scores per run
│
└── frontend/
    ├── package.json
    ├── src/
    │   ├── App.jsx
    │   ├── components/
    │   │   ├── ChatWindow.jsx
    │   │   ├── MessageBubble.jsx
    │   │   ├── CitationCard.jsx      # expandable source excerpt
    │   │   └── PaperSourceBadge.jsx
    │   ├── api/
    │   │   └── askApi.js             # calls backend /ask
    │   └── styles/
    └── public/



    Full stack, step by step
1. Paper collection

requests + PubMed E-utilities API for bulk metadata/full-text pulls
arxiv (Python package) for arXiv papers
Plain Python scripts, nothing fancy needed here

2. PDF parsing & chunking

PyMuPDF (fitz) — extracts text from PDFs; handles multi-column academic paper layouts better than pypdf
LangChain's text splitters (RecursiveCharacterTextSplitter) — off-the-shelf chunking that respects paragraph/section boundaries instead of blind character cuts

3. Embeddings

sentence-transformers library, model: bge-base-en-v1.5 or e5-base-v2 — both free, run locally (no API cost), and are the current strong open-source options for retrieval-quality embeddings
Runs on CPU fine for a corpus this size, no GPU needed

4. Vector database

Qdrant — run it via Docker locally (docker run qdrant/qdrant), or use Qdrant Cloud's free tier for deployment (avoids you having to self-host a stateful DB on Render, which is a pain)

5. Sparse (keyword) retrieval

rank_bm25 — lightweight, pure Python, no separate service needed (no Elasticsearch — overkill for this scale)

6. Hybrid fusion + reranking

Fusion: Reciprocal Rank Fusion — a simple formula, you can hand-write this in ~10 lines, no library needed
Reranker: sentence-transformers cross-encoder model, cross-encoder/ms-marco-MiniLM-L-6-v2 — free, runs locally, standard choice for this exact step

7. Generation

Groq API — same as FinScope, reuse your existing client setup. Model: whatever Llama/Mixtral you're already using there.

8. Backend service

FastAPI — same as your FinScope scanner service. One endpoint: POST /ask → runs steps 3–7 → returns answer + citations.

9. Evaluation

Just Python scripts + a CSV/JSON of your 30–50 test questions. No special library — write the hit-rate/faithfulness checks yourself, they're simple comparisons.

10. Frontend

React — same as FinScope, reuse your component patterns (you already have chat-adjacent UI experience from planning MeetSpace's message layer, and card-based result displays from FinScope's stock cards)

11. Deployment

FastAPI service → Render (same pattern as your existing scanner)
Qdrant → Qdrant Cloud free tier (simplest — avoids managing a stateful DB yourself)
React → Render static site, same as FinScope

One dependency note
sentence-transformers pulls in PyTorch — it's a heavier install than your typical Python project. First-time setup will take a few minutes and eat some disk space; not a blocker, just don't be alarmed by the install time.
Want me to scaffold the project folder structure and a requirements.txt next, or start with Step 1 (pulling a real sample of papers) to get actual data in hand first?{'paperId': '027d70631ba117229e54638fd373411ad2bacdea',
 'externalIds': {'MAG': '2614358123',
  'ArXiv': '1705.04742',
  'DOI': '10.31234/osf.io/e6497',
  'CorpusId': 25130814},
 'title': 'How sustainable are different levels of consciousness',
 'year': 2017,
 'citationCount': 5,
 'openAccessPdf': {'url': 'https://arxiv.org/pdf/1705.04742',
  'status': 'GREEN',
  'license': None,
  'disclaimer': 'Notice: Paper or abstract available at https://arxiv.org/abs/1705.04742, which is subject to the license by the author or copyright owner provided with this content. Please go to the source to verify the license and copyright information for your use.'},
 'fieldsOfStudy': ['Mathematics', 'Biology', 'Psychology'],
 'authors': [{'authorId': '32783505', 'name': 'E. Wiersma'}],
 'abstract': 'The human brain processes a wide variety of inputs and does so either consciously or subconsciously. According to the Global Workspace theory, conscious processing involves broadcasting of information to several regions of the brain and subconscious processing involves more localized processing. This theoretical paper aims to expand on some of the aspects of the Global Workspace theory: how the properties of incoming information result in it being processed subconsciously or consciously; why processing can be either be sustained or short-lived; how the Global Workspace theory may apply both to real-time sensory input as well as to internally retained information. This paper proposes that: familiar input which does not elicit intense emotions becomes processed subconsciously and such processing can be continuous and sustained; input that elicits relatively intense emotions is subjected to highly sustainable conscious processing; input can also undergo meta-conscious processing. Such processing is not very sustainable but can exert control over other cognitive processes. This paper also discusses possible benefits of regulating cognitive processes this way.'} it's the data column give should I also download pdf or only use abstract a training model just explain what to do nextShow moreClarified PDF necessity over abstracts for effective model trainingClarified PDF necessity over abstracts for effective model trainingDownload the PDF — abstract alone isn't enough for RAG.
Here's the distinction:

Abstract = a 150-word summary of the conclusion. Good for one thing only: deciding "is this paper relevant enough to include" — which is exactly what you're doing right now, looking at it before committing.
Full PDF text = the actual methods, detailed arguments, evidence, nuances, competing points — this is what your RAG system needs to answer real questions. If someone asks "how does this paper explain why some processing is sustained vs short-lived," that answer lives in the body text, not the abstract.

If you only index abstracts, your chatbot can only ever give surface-level, generic answers — it'd basically become a fancier abstract-search tool, not a paper-reading assistant. That defeats the point of the whole project.
What to do with this paper specifically
Since openAccessPdf.status is "GREEN" (legally available, author-hosted or repository-hosted copy) and the URL is a direct arXiv PDF link — keep it, this one's a good open-and-shut case:

Judge relevance from the abstract (which you just did) — this one's clearly on-topic (Global Workspace Theory, consciousness).
Log it in metadata.csv: paperId, title, year, authors, DOI, openAccessPdf.url, citationCount.
Download the actual PDF from that url into data/raw_pdfs/ — that's what Step 2 (PyMuPDF extraction) will later chunk and embed.

Practical flow going forward

Loop through all your bulk-search results
For each: read the abstract → decide keep/discard → if keep, append metadata row + queue the PDF URL for download
Batch-download all queued PDFs in one script run at the end (so you don't mix "deciding" and "downloading" logic together)

One caution on this specific paper: it has only 5 citations and is a self-published/theoretical piece (DOI prefix 10.31234/osf.io = OSF preprints, not a peer-reviewed journal). Not disqualifying — theoretical/preprint papers are fine and often good for a consciousness-theory corpus — but you'll want a mix, not only low-citation preprints, so your assistant isn't over-representing fringe or preliminary ideas as if they were settled science.
Want me to write the loop that processes a full batch of search results (keep/discard logic + metadata.csv writer + PDF downloader) next?bro without much thinking only tell the field of study do specific this or it is independent of field of stduyThought for 2sThought for 2sIndependent — don't filter strictly on it. This paper's tagged Mathematics/Biology/Psychology, not even "Neuroscience," despite clearly being a consciousness/Global Workspace Theory paper. Semantic Scholar's auto-tagging is unreliable, so judge relevance from the title + abstract, not the fieldsOfStudy field. Use fieldsOfStudy only as a loose extra filter in your search query, never as the deciding factor.means user can send any pdf and our model give good result hnaClarified distinction between curated corpus and general upload featureClarified distinction between curated corpus and general upload featureNot quite — and this distinction matters, so let's be clear.
What we've scoped so far: a fixed corpus. You (the builder) curate 50–80 papers upfront, index them once, and the chatbot answers questions only from that fixed set. Users can't add their own PDF — they're querying your pre-built neuroscience library.
"User uploads any PDF and gets good answers" is a different, additional feature — usually called dynamic/on-the-fly ingestion. It's not hard given what you've already built, but it's a distinct capability:

User uploads a PDF → your Steps 2–4 (extract, chunk, embed) run live, right then on just that one paper, into a separate temporary space in Qdrant (not mixed with your curated 50–80)
Their questions then retrieve only from their uploaded paper(s), not your fixed corpus
Quality depends entirely on the one PDF they gave you — if it's messy scanned text or a bad-quality PDF, results degrade, since there's no longer a curated, quality-checked corpus behind it

Recommendation
Don't build this yet. Get the fixed-corpus version working end-to-end first (including your eval step — that's the harder, more valuable part). Once that's solid, adding "upload your own PDF" is a relatively small extension: you're just running your existing pipeline on-demand instead of as a batch job.
If you do want it eventually, it's genuinely a good v2 feature — it turns "Ask My Papers" from your fixed library into a general research tool, which is a stronger interview story ("built a fixed-corpus RAG, then extended it to arbitrary user documents").
Want to keep going on the fixed-corpus build first, or are you deciding right now whether to design for both from the start?


Full flow (step by step)
Step 1 — Collect papers
Pull PDFs via the APIs above into a folder. Keep a simple metadata sheet: title, authors, year, source URL.
Step 2 — Extract & chunk text
Parse PDF → clean text → split into small pieces (chunk = a small paragraph-sized slice of text, usually 200–500 words. You chunk because an LLM can only "read" a limited amount at once, so you break papers into bite-sized pieces it can search through). Split on section headings/paragraphs, not blindly every N words — a chunk that cuts a sentence in half is useless.
Step 3 — Turn chunks into embeddings
Embedding = a way of converting a piece of text into a list of numbers that captures its meaning, so a computer can compare "how similar" two pieces of text are, even if they use different words. Every chunk gets converted this way and stored.
Step 4 — Store in a vector database
Vector database = a database built specifically to store embeddings and quickly find "which stored chunks are most similar to this new question." You're using Qdrant for this.
Step 5 — Also build a keyword index (sparse retrieval)
Sparse retrieval (e.g. BM25) = old-school keyword search — good at exact terms (drug names, gene names, acronyms) that meaning-based search sometimes fuzzes over. Meaning-based search alone misses these.
Dense retrieval = the embedding-based similarity search from Step 3–4 (called "dense" because the numbers are packed with meaning info).
Hybrid retrieval = combining both — dense catches meaning, sparse catches exact terms, together they cover more ground than either alone.
Step 6 — Merge and rerank the results
When a question comes in, both searches return a pile of "maybe relevant" chunks. A cross-encoder reranker = a second, more careful (but slower) model that looks at the question and each candidate chunk together and re-scores how relevant it really is — filtering out chunks that only superficially matched. You run this on your top ~20 candidates to pick the real top 5.
Step 7 — Generate the answer
Send the question + the final selected chunks to your LLM (Groq, same as FinScope) with instructions: answer only from the given chunks, cite [Paper, page], and say "not found in the provided papers" if nothing fits.
Step 8 — Build an evaluation set
Write ~30–50 questions yourself with known correct answers. Run them through the system and measure:

Hit-rate@k = out of your questions, what % of the time did the right chunk actually appear in the top-k retrieved results (checks retrieval, before the LLM even answers).
Faithfulness = does the generated answer actually stick to what the source chunks said, or did it drift/hallucinate.

This step is what separates "I built a RAG demo" from "I built and measured a RAG system" — the second one is what actually stands out in an interview.
Step 9 — Frontend + deploy
Simple chat UI showing the answer with clickable citations expanding to the source excerpt. Deploy same as FinScope (Render).
Want me to start with Step 1–2 now — pulling a sample set of papers and testing chunking on a real PDF — or do you want the full tech stack (libraries/tools per step) laid out first