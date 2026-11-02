// Semantic product search, fully in the browser.
// Product vectors are precomputed (data/embeddings.bin). Only the query is embedded here.

const MODEL_ID = "onnx-community/all-MiniLM-L6-v2-ONNX";
const LIB_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
const DIM = 384;
const TOP_K = 5;

const form = document.getElementById("form");
const input = document.getElementById("query");
const go = document.getElementById("go");
const statusEl = document.getElementById("status");
const list = document.getElementById("results");

const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

let products = []; // rows of [id, title, category, price], in the same order as the vectors
let vectors = null; // Float32Array, products.length * DIM, every vector has length 1
let extractorPromise = null;
let latestSearch = 0; // ignore results from searches that were replaced by a newer one

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.className = isError ? "error" : "";
}

// ---- Data ------------------------------------------------------------------

const dataReady = (async () => {
  const [productsRes, vectorsRes] = await Promise.all([
    fetch("data/products.json"),
    fetch("data/embeddings.bin"),
  ]);
  if (!productsRes.ok || !vectorsRes.ok) throw new Error("Could not load the data files");
  const catalog = await productsRes.json();
  const floats = new Float32Array(await vectorsRes.arrayBuffer());
  if (floats.length !== catalog.rows.length * DIM) throw new Error("Data files do not match");
  products = catalog.rows;
  vectors = floats;
})();

dataReady.catch(() => setStatus("Couldn't load the product data. Reload the page to try again.", true));

// ---- Model (loaded on the first search) --------------------------------------

function getExtractor() {
  if (!extractorPromise) {
    const files = new Map();
    const onProgress = (e) => {
      if (e.status !== "progress" || !e.total) return;
      files.set(e.file, { loaded: e.loaded, total: e.total });
      let loaded = 0;
      let total = 0;
      for (const f of files.values()) { loaded += f.loaded; total += f.total; }
      setStatus("Loading the search model… " + Math.round((100 * loaded) / total) + "%");
    };
    extractorPromise = import(LIB_URL).then(({ pipeline }) =>
      pipeline("feature-extraction", MODEL_ID, { progress_callback: onProgress })
    );
    extractorPromise.catch(() => { extractorPromise = null; }); // allow a retry
  }
  return extractorPromise;
}

// ---- Ranking -----------------------------------------------------------------

// Vectors have length 1, so cosine similarity is just the dot product.
function topK(query, k, skip = -1) {
  const n = products.length;
  const scores = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    const offset = i * DIM;
    for (let d = 0; d < DIM; d++) sum += vectors[offset + d] * query[d];
    scores[i] = sum;
  }
  const order = [];
  for (let i = 0; i < n; i++) if (i !== skip) order.push(i);
  order.sort((a, b) => scores[b] - scores[a]);
  return order.slice(0, k).map((i) => ({ i, score: scores[i] }));
}

// ---- Rendering ---------------------------------------------------------------

// Uses textContent only, so data can never inject markup.
function render(items) {
  list.replaceChildren();
  for (const { i, score } of items) {
    const [, title, category, price] = products[i];
    const li = document.createElement("li");

    const name = document.createElement("div");
    name.textContent = title;

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = category + " · " + money.format(price) +
      (score === undefined ? "" : " · score " + score.toFixed(2));

    const more = document.createElement("button");
    more.type = "button";
    more.className = "similar";
    more.textContent = "More like this";
    more.addEventListener("click", () => showSimilar(i));

    li.append(name, meta, more);
    list.append(li);
  }
}

function showSimilar(i) {
  latestSearch++; // cancels any search still waiting on the model
  go.disabled = false;
  const started = performance.now();
  const items = topK(vectors.subarray(i * DIM, (i + 1) * DIM), TOP_K, i);
  const ms = performance.now() - started;
  setStatus(TOP_K + " products similar to “" + products[i][1] + "” (" + ms.toFixed(1) + " ms)");
  render(items);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ---- Search ------------------------------------------------------------------

async function search(query) {
  if (!query) return;
  const mine = ++latestSearch;
  go.disabled = true;
  list.replaceChildren();

  try {
    await dataReady;
    setStatus("Loading the search model…");

    let extractor;
    try {
      extractor = await getExtractor();
    } catch (err) {
      if (mine !== latestSearch) return;
      setStatus("Couldn't load the search model. It needs a connection to jsDelivr and Hugging Face. " +
        "You can still browse below and use “More like this”.", true);
      render(products.slice(0, 8).map((_, i) => ({ i })));
      return;
    }

    setStatus("Searching…");
    const output = await extractor(query, { pooling: "mean", normalize: true });
    if (mine !== latestSearch) return;
    if (output.data.length !== DIM) throw new Error("Unexpected embedding size");

    const started = performance.now();
    const items = topK(output.data, TOP_K);
    const ms = performance.now() - started;
    setStatus(items.length + " results for “" + query + "” (ranked " + products.length +
      " products in " + ms.toFixed(1) + " ms)");
    render(items);
  } catch (err) {
    if (mine === latestSearch) setStatus("Something went wrong: " + err.message, true);
  } finally {
    if (mine === latestSearch) go.disabled = false;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  search(input.value.trim());
});

for (const chip of document.querySelectorAll(".try button")) {
  chip.addEventListener("click", () => {
    input.value = chip.textContent;
    search(chip.textContent);
  });
}
