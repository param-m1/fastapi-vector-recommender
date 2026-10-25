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
