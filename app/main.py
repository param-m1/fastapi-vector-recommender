import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer

from app.schemas import RecommendRequest, RecommendResponse, Recommendation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "products.csv"
EMBEDDINGS_PATH = PROJECT_ROOT / "artifacts" / "embeddings.npy"
IDS_PATH = PROJECT_ROOT / "artifacts" / "ids.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = SentenceTransformer(MODEL_NAME, device = "cpu")
    app.state.embeddings = np.load(EMBEDDINGS_PATH)
    ids = np.load(IDS_PATH)
    app.state.products = pd.read_csv(DATA_PATH)

    if not np.array_equal(ids, app.state.products["product_id"].to_numpy()):
        raise ValueError("Embedding rows do not match products.csv order")

    yield

app = FastAPI(lifespan = lifespan)

@app.get("/health")
def health(request: Request):
    return{"products": len(request.app.state.products), "status": "ok"}

@app.post(r"/recommend", response_model = RecommendResponse)
def recommend(body: RecommendRequest, request: Request):
    start_time = time.perf_counter()
    state = request.app.state

    query_vec = state.model.encode_query(body.query, normalize_embeddings = True)
    scores = state.embeddings @ query_vec

    k = body.top_k
    top = np.argpartition(scores, -k)[-k:]
    top = top[np.argsort(scores[top])][::-1]

    results = []
    for i in top:
        row = state.products.iloc[i]
        results.append(Recommendation(
            product_id = int(row["product_id"]),
            title = row["title"],
            category = row["category"],
            price = float(row["price"]),
            score = float(scores[i]),
        ))

    latency_ms = (time.perf_counter() - start_time) * 1000
    return RecommendResponse(query = body.query, results = results, latency_ms = latency_ms)