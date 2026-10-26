# fastapi-vector-recommender

Semantic search for a small product catalog. You type what you want ("a tent that survives heavy rain") and it returns the closest products by meaning instead of by matching keywords.

I'm a first-year CS student and I built this to understand how vector search works from start to finish: embeddings, similarity, loading a model once at startup, and (still to do) measuring latency and putting it in Docker. I wrote the backend myself. The search page in `app/static` is generated, and it's only there so there's something to click.

## Run it

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 for the search page or http://127.0.0.1:8000/docs for the API docs. The first start downloads the model (about 90 MB).

The data and embeddings are already committed. To regenerate them: `python data/generate_dataset.py`, then `python generate_embeddings.py`.

## API

`POST /recommend` takes a JSON body:

| Field | Type | Rules |
|---|---|---|
| `query` | string | 1 to 200 characters |
| `top_k` | integer | 1 to 20, default 5 |

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "a tent that survives heavy rain", "top_k": 5}'
```

It returns `query`, `latency_ms` (time spent inside the handler) and a list of `results`, each with `product_id`, `title`, `category`, `price` and `score` (cosine similarity). Bad input gets a 422. `GET /health` returns the number of products loaded.

## How it works

The 1,000 products in `data/products.csv` are fake. I generate them from a fixed seed, so I get the exact same file every time (I checked that the file hash doesn't change when I regenerate it). `generate_embeddings.py` turns each title and description into a 384-number vector with `all-MiniLM-L6-v2` and saves them in `artifacts/`, so the server doesn't have to do that every time it starts.

When the server starts, it loads the model, the vectors and the CSV once. On each request it only encodes the query and compares it to all 1,000 vectors in one NumPy multiply. That works because I normalize every vector to length 1 ahead of time, which makes cosine similarity the same as a dot product. I use `np.argpartition` to get the top results without sorting all 1,000 scores.

At this size I just compare against every product. An approximate index like FAISS would be overkill for 1,000 items. The endpoint is a regular `def` instead of `async def` because encoding the query is CPU work, and I didn't want it blocking the event loop.

On startup the server also checks that the saved IDs match the CSV's row order. If they ever drifted, results would quietly point at the wrong products and nothing would crash, so I made it refuse to start instead.

Most of my debugging time went on small things: a stray character that broke the dataset script, a `__main__` guard sitting above the functions it called, and a path that was one folder level off.

## Not done yet

Latency benchmarks, Locust load tests and Docker. The data is synthetic, so it's good for checking the pipeline works but doesn't say much about quality on a real catalog. There's also no filtering by category or price.
