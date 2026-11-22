const chatEl = document.getElementById("chat");
const statusEl = document.getElementById("status");
const composer = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const apiKeyInput = document.getElementById("apiKey");
const clearBtn = document.getElementById("clearBtn");
const healthPill = document.getElementById("healthPill");
const suggestions = document.getElementById("suggestions");

const STORAGE_KEY = "healthcare-rag-api-key";
const DEFAULT_KEY = "dev-secret-change-me";

/** @type {{role: 'user'|'assistant', content: string}[]} */
let messages = [];

apiKeyInput.value = localStorage.getItem(STORAGE_KEY) || DEFAULT_KEY;
apiKeyInput.addEventListener("change", () => {
  localStorage.setItem(STORAGE_KEY, apiKeyInput.value.trim());
});

function setStatus(text) {
  statusEl.textContent = text || "";
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatAnswer(text) {
  // Preserve full answer text; render paragraphs instead of clipping.
  return escapeHtml(text)
    .split(/\n{2,}/)
    .map((para) => `<p class="answer-p">${para.replaceAll("\n", "<br>")}</p>`)
    .join("");
}

function renderSources(sources) {
  if (!sources?.length) return "";
  const items = sources
    .map((s) => {
      const title = escapeHtml(s.title || s.doc_id);
      const org = escapeHtml(s.source || "Unknown source");
      const license = escapeHtml(s.license || "n/a");
      const excerpt = escapeHtml(s.text || "");
      const link = s.url
        ? `<a href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer">Open source</a>`
        : "";
      return `<details>
        <summary>${title} · ${org}</summary>
        <p>${excerpt}</p>
        <p>Licence: ${license}${link ? ` · ${link}` : ""}</p>
      </details>`;
    })
    .join("");
  return `<div class="sources">${items}</div>`;
}

function appendMessage(role, content, sources = null, meta = null) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  const label = role === "user" ? "You" : "Assistant";
  const metaLine = meta
    ? `<div class="label">${label} · ${meta}</div>`
    : `<div class="label">${label}</div>`;
  const body =
    role === "assistant"
      ? `<div class="answer-body">${formatAnswer(content)}</div>`
      : `<div>${escapeHtml(content)}</div>`;
  div.innerHTML = `${metaLine}${body}${
    role === "assistant" ? renderSources(sources) : ""
  }`;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

async function refreshHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    healthPill.innerHTML = `<strong>online</strong> · ${data.indexed_chunks} chunks · ${data.generation_mode}`;
  } catch {
    healthPill.textContent = "offline";
  }
}

async function sendQuestion(question) {
  const q = question.trim();
  if (!q) return;

  const apiKey = apiKeyInput.value.trim();
  if (!apiKey) {
    setStatus("Add an API key first.");
    return;
  }
  localStorage.setItem(STORAGE_KEY, apiKey);

  messages.push({ role: "user", content: q });
  appendMessage("user", q);
  input.value = "";
  sendBtn.disabled = true;
  setStatus("Thinking over the corpus…");

  try {
    const res = await fetch("/v1/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": apiKey,
      },
      body: JSON.stringify({ messages }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    messages.push({ role: "assistant", content: data.answer });
    const meta = `${Math.round(data.latency_ms)} ms · ${data.generation_mode}`;
    appendMessage("assistant", data.answer, data.sources, meta);
    setStatus("");
  } catch (err) {
    messages.pop();
    setStatus(err.message || "Something went wrong.");
    appendMessage(
      "assistant",
      "Sorry — I could not complete that request. Check the API key and that the server is running."
    );
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuestion(input.value);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendQuestion(input.value);
  }
});

clearBtn.addEventListener("click", () => {
  messages = [];
  chatEl.innerHTML = "";
  setStatus("Chat cleared.");
});

suggestions.addEventListener("click", (event) => {
  const btn = event.target.closest("button[data-q]");
  if (!btn) return;
  sendQuestion(btn.dataset.q);
});

appendMessage(
  "assistant",
  "Hi — ask me about AI ethics in health, mammography, GMLP, or transparency for ML medical devices. I only answer from the indexed open sources."
);
refreshHealth();
