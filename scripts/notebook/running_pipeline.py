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
  




def get_top_n_chunks(query, n=5, k=60):
    results = get_query_ans(query)
    bm25_ = results.get("bm25")
    qdrant_ = results.get("qdrant")

    fused_scores = {}   

    for item in bm25_:
        chunk_id = item["uuid"]
        rank = item["rank"]
        score = 1 / (k + rank)

        if chunk_id in fused_scores:
            fused_scores[chunk_id]["score"] += score
        else:
            fused_scores[chunk_id] = {"score": score, "text": item["text"]}


    for idx, item in enumerate(qdrant_):
        chunk_id = item["chunk_id"]
        rank = idx + 1
        score = 1 / (k + rank)

        if chunk_id in fused_scores:
            fused_scores[chunk_id]["score"] += score
        else:
            fused_scores[chunk_id] = {"score": score, "text": item["text"]}


    ranked = sorted(fused_scores.items(), key=lambda x: x[1]["score"], reverse=True)

    top_n = [
        {"chunk_id": chunk_id, "score": data["score"], "text": data["text"]}
        for chunk_id, data in ranked[:n]
    ]
    return top_n

query = """ Imagine a robotics lab is trying to build an advanced "embodied AI" capable of engaging in genuine social dialogue and demonstrating ethical responsibility, as discussed by A.V. Nedyak. To achieve this, the engineers realize the AI needs a central control mechanism that can dynamically shift its computational resources between processing the external world (listening to the human) and accessing its own internal "lifeworld" (its artificial self-awareness and simulated emotions).
Based on the neurobiological research provided in the sources, which specific human brain network should these AI engineers attempt to mimic? Furthermore, if the engineers fail to build this network correctly, causing the AI to lack introspection and fail at understanding the human's motives, what specific human neurodevelopmental disorder would the AI's behavior resemble"""
chunks = [{'chunk_id': '28data_c78', 'score': 0.03149801587301587, 'text': '5 \nIntegration Theory (IIT; [11, 50]) and Global Neuronal Workspace (GNW; [7,8]). Unfortunately, these theories \n6 \nhave seldomly been tested neurophysiologically, and never within the same dataset. A series of concrete \n7 \npredictions can be generated from these theories, and in the current work we sought to generate such \n8 \npredictions and test them.   \n9 \n \n0 \nStarting from IIT mathematics, a strong prediction that can be made is that as an organism transitions from \n1 \nconscious to unconscious states, central integrative hubs (vs. convergent hubs) of neural networks should be \n2 \nmost impacted. Indeed, IIT states that the greater theinformation possessed by a network above and beyond \n3 \nits constituent parts, the more conscious the system [11, 50]. Level of consciousness may be calculated and \n4 \nrepresented as the variable phi (Φ). We demonstrate that within a simple three-node network, if the central \n5 \nnode is an integrative (“AND” gate) node as opposed to a convergent (“XOR” gate) node, the value of Φ triples \n6 \n(see SI). In evaluating the neuronal data, we categorized neurons as either convergent or integrative and \n7'}, {'chunk_id': '28data_c15', 'score': 0.01639344262295082, 'text': '2 \nof neural processing according to these theories, and are mute regarding “what it feels like” [24] or the “Hard \n3 \nProblem” [25] of consciousness. With these caveats in mind, first we formalize the role of multisensory neurons \n4 \nthat integrate information from multiple sensory modalities (operationalized as being driven by multisensory \n5 \nstimulation than to unisensory stimulation, “AND” gates) vs. those that converge yet do not integrate \n6 \n(operationalized as responding to multiple sensory modalities but not being further driven by multisensory \n7 \nconditions, “XOR” gates = “OR” gates – “AND” gates; see [26] for an early characterization of multisensory \n8 \nneurons as Boolean gates). Interestingly, IIT mathematics suggests that a simple 3-node network (e.g., \n9 \nunisensory audio node, unisensory tactile node, and multisensory audio-tactile node) merging on an “AND” \n0 \ngate bears a greater degree of integrated information than one converging on an “XOR” gate (Φ = 0.78 vs. Φ = \n1 \n.\nCC-BY-NC-ND 4.0 International license\na \ncertified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available under'}, {'chunk_id': '28data_c14', 'score': 0.01639344262295082, 'text': '3 \naccess to a limited purview of the system (see [9] for more detail). Regretably, computing this measure in \n4 \ncomplex biological systems is impossible from a practical standpoint due to its combinatorial search problem \n5 \n(but see [22, 23] for interesting approaches circumventing current computing limitations).  \n6 \n \n7 \nIn an effort to provide empirical evidence germane to theories of consciousness, we propose here simple \n8 \nneurophysiological benchmarks for consciousness as derived from the GNW and IIT, and test them empirically \n9 \nin single unit recordings in non-human primates. Of note, we must emphasize that the predictions derived \n0 \nbelow, are according to IIT mathematics and are logical consequence to IIT and GNW literature, yet are not \n1 \nnecessarily put forward explicitly by either IIT or GNW theorists. Further, these predictions bear on the nature \n2 \nofneural processing according to these theories, and are mute regarding “what it feels like” [24] or the “Hard \n3 \nProblem” [25] of consciousness. With these caveats in mind, first we formalize the role of multisensory neurons \n4'}, {'chunk_id': '28data_c40', 'score': 0.016129032258064516,'text': '12\nintegrative, or “XOR” gates). Conversely, neurons that responded more vigorously to the multisensory \n9 \ncombination were labeled as integrative neurons, and were operationally categorized as “AND” gates. This \n0 \nbifurcation of neurons into exclusive groups is important from a statistical perspective (in order not to create \n1 \ngroups that are partially overlapping and overlapping to different extents across states of consciousness and \n2 \nrecordings areas), and most importantly, from a theoretical perspective, creating “AND” and “XOR” neuronal \n3 \npools. However, given the initial number of recorded neurons in S1 and vPM, this categorization scheme \n4 \nyielded a sufficient number of convergent(N=125) and integrative (N=64) neurons in S1, but not in vPM \n5 \n(convergent, N = 61; integrative, N = 8). Thus, for the analyses specifically probing the difference between \n6 \nconvergent and integrative neurons in light of IIT (Predictions #1 and #2), analyses are restricted to S1. \n7 \n \n8 \n \nTesting Consciousness Theory in Multisensory Neurons; Information Integration Theory \n9 \n \n0 \nPrediction #1; Are integrative neurons most readily impacted by loss of consciousness? A first \n1'}, {'chunk_id': '29data_c4', 'score': 0.016129032258064516, 'text': '3 \nIntroduction \nDespite a substantial effort, which has dramatically increased over the past several \ndecades, a satisfactory scientific theory of consciousness remains elusive. Spectacular advances \nin our ability to study the behavior of neuronal systems in situ and at scale have not yet yielded \nthe theoretical bridge that connects the behavior of these systems with an explanation of \nphenomenal consciousness; as yet, the hard problem of consciousness has not been solved even \nas proffered physicalist theories of consciousness continue to multiply.(Seth & Bayne 2022) \nThe problems that continue to plague theories of consciousness have been well described; \na recent paper by Doerig and colleagues offers a reasonable starting point for assessing the \nviability of these and proposed theories.(Doerig et al. 2021) In it, the authors identify that \ntheories must be constructed in such a way as to cope with a number of key questions and \ncriteria. These elements represent a summary of the major neuroscientific, analytical, and \nphilosophical considerations and quandaries affecting the viability of various candidate theories \nof consciousness. \n \nPresentation of the Theory'}, {'chunk_id': '29data_c5', 'score': 0.015873015873015872, 'text': "criteria. These elements represent a summary of the major neuroscientific, analytical, and \nphilosophical considerations and quandaries affecting the viability of various candidate theories \nof consciousness. \n \nPresentation of the Theory \nThis paper aims to develop a theory of consciousness that aims to fulfill these and other \nkey criteria. The main ideas put forth in this paper - the State Space Theory - are as follows: \n1. Generic computation - The computational processing that occurs in the cortex is \nsomewhat generic rather than highly specialized. This is evidenced through \nneuroscientific observations including interindividual variability in normal brain function \n(primary visual cortex; Broca's area) as well as plasticity in response to congenital"}, {'chunk_id': '28data_c79', 'score': 0.015625, 'text': '5 \nnode is an integrative (“AND” gate) node as opposed to a convergent (“XOR” gate) node, thevalue of Φ triples \n6 \n(see SI). In evaluating the neuronal data, we categorized neurons as either convergent or integrative and \n7 \nexaminedwhich class was most impacted by propofol administration. The assumption here is that cross-modal \n8 \nneurons in S1 and vPM receive informationregarding the different senses from upstream areas, S1 and vPM \n9 \nin turn being the central node composed of “AND” and “XOR” functionality. Ofcourse, this is an over-simplified \n0 \nbiological neural network, but one that permits testing predictions derived from the IIT from a \n1 \nneurophysiological perspective. Contrary to our IIT-derived predictions, convergent, as opposed to integrative, \n2 \nneurons were most impacted by the administration of anesthesia. To further test predictions derived from IIT, \n3 \nwe reasoned that when organisms were conscious, integrative neurons should exhibit neurophysiological \n4 \nproperties of consciousness to a greater extent than convergent neurons (i.e., supporting lower phi-values). \n5'}]
print(get_top_n_chunks(query))


    
    
    
    
    
    
    