from real_retrieval import retrieve

results = retrieve("How does Global Workspace Theory explain consciousness?", top_k=5)
for r in results:
    print(r["chunk_id"], "|", r["paper_title"], "|", r["score"])
    print(r["text"][:150])
    print()
