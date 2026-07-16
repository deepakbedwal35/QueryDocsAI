import dotenv
from dotenv import load_dotenv
import pandas as pd
import requests
import os
import json
load_dotenv()

semantic_scholor_url = os.getenv("SEMANTIC_URL")
params = {
    "query": "global workspace theory consciousness",
    "fields": "title,abstract,year,authors,citationCount,openAccessPdf,externalIds,fieldsOfStudy",
    "fieldsOfStudy": "Neuroscience,Psychology",
    "year": "2010-2026",
    "openAccessPdf": ""   
}

response = requests.get(semantic_scholor_url, params=params)
data = response.json()
df = pd.DataFrame(data)
all_datas = df['data']
# print(data)
print(df['data'].columns)
# # print(all_datas)
# for paper in all_datas:
#     if paper.get("openAccessPdf"):
#         print(paper["title"], "→", paper["openAccessPdf"]["url"])

    