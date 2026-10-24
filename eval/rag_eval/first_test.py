"""
First test: send ONE question to ONE model with live retrieval, print the raw
answer. No loop, no scoring.

Run (from repo root, Qdrant up):  python -m eval.rag_eval.first_test
"""

from backend.generation.groq_client import generate_answer
from backend.retrieval.hybrid import retrieve

# ---------- FILL THESE IN ----------
MODEL = "openai/gpt-oss-120b"     # Groq model slug
question = "What does 'ignition' mean in the context of the Global Workspace Theory (GWT)?"
# -----------------------------------

chunks = retrieve(question, top_k=5)
answer = generate_answer(question, chunks, model=MODEL, temperature=0)

print("=" * 60)
print("QUESTION:", question)
print("RETRIEVED:", [c["chunk_id"] for c in chunks])
print("=" * 60)
print("RAW MODEL OUTPUT:")
print(answer)
print("=" * 60)
