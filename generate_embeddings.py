from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
IDS_PATH = ARTIFACTS_DIR / "ids.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64

def main() -> None:
    print(DATA_PATH, DATA_PATH.exists())

if __name__ == "__main__":
    main()


