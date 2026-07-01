const form = document.getElementById("query-form");
const answerEl = document.getElementById("answer");
const metadataEl = document.getElementById("metadata");
const resultEl = document.getElementById("result");
const errorEl = document.getElementById("error");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultEl.classList.add("hidden");
  errorEl.classList.add("hidden");
  answerEl.textContent = "";
  metadataEl.textContent = "";

  const question = document.getElementById("question").value.trim();
  if (!question) {
    errorEl.textContent = "Please enter a legal question.";
    errorEl.classList.remove("hidden");
    return;
  }

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.detail || "Unable to fetch answer.");
    }

    const data = await response.json();
    answerEl.textContent = data.answer;
    metadataEl.innerHTML = `
      <div class="metadata-item"><strong>Latency:</strong> ${data.latency_ms.toFixed(0)} ms</div>
      <div class="metadata-item"><strong>Retrieval:</strong> ${data.retrieval_ms.toFixed(0)} ms</div>
      <div class="metadata-item"><strong>Rerank:</strong> ${data.rerank_ms.toFixed(0)} ms</div>
      <div class="metadata-item"><strong>Generation:</strong> ${data.generation_ms.toFixed(0)} ms</div>
      <div class="metadata-item"><strong>Cost estimate:</strong> $${data.cost_usd.toFixed(6)}</div>
      <div class="metadata-item"><strong>Request id:</strong> ${data.request_id}</div>
    `;
    resultEl.classList.remove("hidden");
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.classList.remove("hidden");
  }
});
