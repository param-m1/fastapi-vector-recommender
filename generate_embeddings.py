from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "products.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
IDS_PATH = ARTIFACTS_DIR / "ids.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64


def build_text(title: str, description: str) -> str:
    return f"{title}. {description}"


def load_products(path: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path)
    ids = df["product_id"].to_numpy()
    texts = [build_text(title, description) for title, description in zip(df["title"], df["description"])]
    return ids, texts

def main() -> None:
    ids, texts = load_products(DATA_PATH)
    # print(len(ids))
    # print(ids[:3])
    # print(texts[0])

    model = SentenceTransformer(MODEL_NAME, device = "cpu")
    start_time = time.perf_counter()
    embeddings = embed(model, texts)
    end_time = time.perf_counter()
    print(f"Embedding time: {end_time - start_time:.2f} seconds")
    print(embeddings.shape, embeddings.dtype)

def embed(model, texts) -> np.ndarray:
    return model.encode_document(texts, batch_size = BATCH_SIZE, normalize_embeddings = True, show_progress_bar = True).astype(np.float32, copy = False)

if __name__ == "__main__":
    main()




