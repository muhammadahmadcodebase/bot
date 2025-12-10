# api/recommend.py
import pickle
import numpy as np
from fastapi import FastAPI
from mangum import Mangum
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load saved product embeddings
# -------------------------
with open("../products_with_embeddings.pkl", "rb") as f:
    df = pickle.load(f)

# Load SBERT model
model = SentenceTransformer("all-mpnet-base-v2")

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI()

class Query(BaseModel):
    query: str
    top_k: int = 5

# -------------------------
# Semantic search function
# -------------------------
def semantic_search(query, top_k=10):
    query_emb = model.encode(query).reshape(1, -1)
    item_embs = np.vstack(df["embedding"].values)
    sims = cosine_similarity(query_emb, item_embs)[0]

    df_local = df.copy()
    df_local["semantic_score"] = sims
    return df_local.sort_values("semantic_score", ascending=False).head(top_k)

# -------------------------
# Simple ranking
# -------------------------
def rank_items(df_candidates):
    df_candidates["price_score"] = 1 / (df_candidates["Variant Price"] + 1)
    df_candidates["final_score"] = (
        df_candidates["semantic_score"] * 0.7 +
        df_candidates["price_score"] * 0.3
    )
    return df_candidates.sort_values("final_score", ascending=False)

# -------------------------
# API route
# -------------------------
@app.post("/")
def recommend(data: Query):
    candidates = semantic_search(data.query)
    ranked = rank_items(candidates).head(data.top_k)

    results = []
    for _, row in ranked.iterrows():
        results.append({
            "title": row["Title"],
            "price": float(row["Variant Price"]),
            "tags": row["Tags"],
            "image": row["Image Src"],
            "score": float(row["final_score"])
        })
    return {"query": data.query, "recommendations": results}

# -------------------------
# Convert to Vercel serverless handler
# -------------------------
handler = Mangum(app)
