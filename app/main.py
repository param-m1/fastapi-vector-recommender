import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from app.schemas import RecommendRequest, RecommendResponse, Recommendation

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "products.csv"
EMBEDDINGS_PATH = PROJECT_ROOT / "artifacts" / "embeddings.npy"
IDS_PATH = PROJECT_ROOT / "artifacts" / "ids.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}
