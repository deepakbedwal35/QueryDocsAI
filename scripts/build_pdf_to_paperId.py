# scripts/build_pdf_to_paperid.py
import json
import pandas as pd

metadata = pd.read_csv("data/metadata.csv")

# row index N -> filename "{N}data.pdf" -> stem "{N}data" -> paperId
mapping = {
    f"{idx}data": row["paperId"]
    for idx, row in metadata.iterrows()
}

with open("data/pdf_to_paperid.json", "w") as f:
    json.dump(mapping, f, indent=2)

print(f"Built mapping for {len(mapping)} papers")