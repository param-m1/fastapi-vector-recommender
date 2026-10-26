  const form = document.getElementById("form");
  const input = document.getElementById("query");
  const go = document.getElementById("go");
  const status = document.getElementById("status");
  const results = document.getElementById("results");

  function show(message, isError) {
    status.textContent = message;
    status.className = isError ? "error" : "";
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    go.disabled = true;
    results.replaceChildren();
    show("Searching…");

    try {
      const res = await fetch("/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      if (!res.ok) {
        let reason = "The API returned an error (" + res.status + ").";
        if (res.status === 422) {
          const body = await res.json().catch(() => null);
          if (body && body.detail && body.detail[0]) reason = body.detail[0].msg + ".";
        }
        return show(reason, true);
      }

      const data = await res.json();
      show(data.results.length + " results in " + data.latency_ms.toFixed(1) + " ms");

      for (const r of data.results) {
        const li = document.createElement("li");
        const title = document.createElement("div");
        const meta = document.createElement("div");
        title.textContent = r.title;
        meta.className = "meta";
        meta.textContent = r.category + " · $" + r.price.toFixed(2) + " · score " + r.score.toFixed(2);
        li.append(title, meta);
        results.append(li);
      }
    } catch (err) {
      show("Can't reach the API. Is the server running?", true);
    } finally {
      go.disabled = false;
    }
  });