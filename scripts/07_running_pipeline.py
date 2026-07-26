"""
scripts/run_pipeline.py

Orchestrates the offline indexing pipeline end-to-end, calling each
step's function directly (no subprocess) since every script now
exposes a plain callable.
"""

import time

from importlib import import_module

STEPS = [
    ("02_extract_text", "extract_all"),
    ("03_chunk_text", "chunk_all"),
    ("04_build_embeddings", "embed_all"),
    ("05_index_qdrant", "index_all"),
    ("06_build_bm25_index", "build_bm25_index"),
    # 01_fetch_papers is run separately/manually per search query,
    # not part of the automatic rebuild -- see its __main__ block.
]


def main():
    overall_start = time.time()

    for module_name, func_name in STEPS:
        print(f"\n{'='*50}\nRunning: {module_name}.{func_name}()\n{'='*50}")
        start = time.time()
        try:
            module = import_module(module_name)
            getattr(module, func_name)()
        except Exception as e:
            print(f"\n❌ FAILED at {module_name}.{func_name}(): {e}")
            raise
        print(f"✅ Done: {module_name} ({time.time() - start:.1f}s)")

    print(f"\n🎉 Pipeline completed successfully in {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()