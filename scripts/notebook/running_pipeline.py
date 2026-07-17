from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
import pickle
import json 
import re
import os
import numpy as np
import uuid

def tokenise_query_text(text):
    clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
    return clean_text.split()

model = SentenceTransformer("BAAI/bge-base-en-v1.5")


def get_query_ans(query:str):
    
    query_embeddings = model.encode( query, normalize_embeddings=True)
    
    client = QdrantClient(host = "localhost", port=6333)
    search_results = client.query_points(
        collection_name="sample_docs_collections",
        query=query_embeddings,
        limit=4,    
    ).points
    
    for point in search_results:
        print(f"Score: {point.score:.4f} | id: {point.id}")
    
    tokenise_query = tokenise_query_text(query)
    
    with open("./data/sample_bm25_index.pkl" , "rb") as f_bm25:
        sample_bm25 = pickle.load(f_bm25)

    with open("./data/sample_mapping_table.pkl" , "rb") as f_map:
        sample_mapping_table = pickle.load(f_map)
    
    doc_scores = sample_bm25.get_scores(tokenise_query)
    top_indices = np.argsort(doc_scores)[::-1][:4]
    
    results = []
    for rank, idx in enumerate(top_indices, start=1):
            score = doc_scores[idx]
            if score == 0:
                continue
                
            data_mapping = sample_mapping_table[idx]
            
            results.append({
                "rank": rank,
                "score": round(score, 4),
                "uuid": data_mapping.get("chunk_id"),
                "text":data_mapping.get("text")
            })
    
    search_hit_scores = {str(point.id): point.score for point in search_results}

    matched_results = []


    for item in sample_mapping_table:
        chunk_id = item.get("chunk_id")
        
    
        valid_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
        if valid_uuid in search_hit_scores:
        
            qdrant_score = search_hit_scores[valid_uuid]
            
    
            matched_results.append({
                "uuid": valid_uuid,
                "chunk_id": chunk_id,
                "qdrant_score": round(float(qdrant_score), 4), 
                "text": item.get("text")
            })

    matched_results = sorted(matched_results, key=lambda x: x["qdrant_score"], reverse=True)
    
    return {
        "bm25":results,
        "qdrant":matched_results
    }
  

query = """If AI achieves autonomous behavior, what two factors will 
determine if it is granted legal personhood (right-subjectivity)"""
print(get_query_ans(query))
    
    
    
    