"""
The models under test, as Groq slugs.
Kept as a list of (display_name, slug) so output tables show friendly names.
"""
MODELS = [
    ("GPT-OSS 120B",  "openai/gpt-oss-120b"),
    ("GPT-OSS 20B",   "openai/gpt-oss-20b"),
    ("Qwen3.8 27B",   "qwen/qwen3.8-27b"),
    ("ALLaM 2 7B",    "allam-2-7b"),
]

# Per-model completion cap (default 2000: reasoning models think before answering).
# ALLaM has a 4k context window, so the 5-chunk prompt leaves little room.
MAX_TOKENS = {"allam-2-7b": 400}

if __name__ == "__main__":
    print(f"{len(MODELS)} models under test:")
    for name, slug in MODELS:
        print(f"  {name:15s} -> {slug}")
