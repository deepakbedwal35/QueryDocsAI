"""
scripts/01_fetch_papers.py

Fetches candidate papers from the Semantic Scholar bulk search API,
saves metadata to data/metadata.csv, and downloads the open-access
PDFs to data/raw_pdfs/.

Can be run directly (`python 01_fetch_papers.py`) or imported:
    from scripts.fetch_papers import fetch_and_download
"""

from __future__ import annotations

import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

SEMANTIC_URL = os.getenv("SEMANTIC_URL", "https://api.semanticscholar.org/graph/v1/paper/search/bulk")
METADATA_PATH = "../../data/metadata.csv"
RAW_PDFS_PATH = "../../data/raw_pdfs"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def search_papers(query: str, year_range: str = "2010-2026") -> pd.DataFrame:
    """Search Semantic Scholar and return a flat DataFrame of results."""
    params = {
        "query": query,
        "fields": "title,abstract,year,authors,citationCount,openAccessPdf,externalIds,fieldsOfStudy",
        "fieldsOfStudy": "Neuroscience,Psychology,Biology,Computer Science,Philosophy",
        "year": year_range,
        "openAccessPdf": "",  # presence of this param = filter to open-access only
    }
    response = requests.get(SEMANTIC_URL, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data)
    df = pd.json_normalize(df["data"])
    return df


def save_metadata(df: pd.DataFrame, path: str = METADATA_PATH) -> None:
    """Append the relevant metadata columns to the master CSV."""
    req_cols = ["paperId", "title", "year", "authors", "externalIds.DOI", "openAccessPdf.url", "citationCount"]
    meta_data = df[req_cols]

    file_exists = os.path.exists(path)
    meta_data.to_csv(path, mode="a", header=not file_exists, index=False)


def download_pdfs(df: pd.DataFrame, out_dir: str = RAW_PDFS_PATH) -> None:
    """Download each open-access PDF, named by row index (matches original notebook)."""
    os.makedirs(out_dir, exist_ok=True)
    pdf_urls = df["openAccessPdf.url"]

    for index, url in pdf_urls.dropna().items():
        try:
            res = requests.get(url, headers=HEADERS, stream=True, timeout=15)
            if res.status_code == 200:
                file_path = os.path.join(out_dir, f"{index}data.pdf")
                with open(file_path, "wb") as pdf_file:
                    for chunk in res.iter_content(chunk_size=4096):
                        pdf_file.write(chunk)
                print(f"Success: Index {index} saved")
            else:
                print(f"Skipped index {index}: HTTP {res.status_code}")
        except Exception as e:
            print(f"Error downloading index {index} ({url}): {e}")


def fetch_and_download(query: str, year_range: str = "2010-2026") -> None:
    """Full flow: search -> save metadata -> download PDFs."""
    df = search_papers(query, year_range)
    print(f"Found {df.shape[0]} papers for query: '{query}'")
    save_metadata(df)
    download_pdfs(df)


if __name__ == "__main__":
    # Edit this list to run multiple theory-specific searches in one go
    queries = [
        "global workspace theory consciousness",
    ]
    for q in queries:
        fetch_and_download(q)