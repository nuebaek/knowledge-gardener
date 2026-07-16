const transcript = document.getElementById("transcript");
const transcriptInner = document.getElementById("transcript-inner");
const emptyState = document.getElementById("empty-state");
const examples = document.getElementById("examples");
const composer = document.getElementById("composer");
const input = document.getElementById("message");
const submitBtn = document.getElementById("submit-btn");
const announcer = document.getElementById("announcer");
const corpusTag = document.getElementById("corpus-tag");

const TOOL_LABELS = {
  write_daily: "데일리노트",
  write_weekly: "위클리노트",
  write_til: "TIL",
};

marked.setOptions({ breaks: true, gfm: true });

// ---- thread_id: created once, kept for the life of this browser's journal,
// but "새 채팅" can mint a fresh one without reloading the page ----
const THREAD_KEY = "kaia:thread_id";

function getThreadId() {
  let id = localStorage.getItem(THREAD_KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(THREAD_KEY, id);
  }
  return id;
}

let threadId = getThreadId();

// ---- markdown + LaTeX rendering (shared by chat answers and the document reader) ----
function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// CommonMark 규칙상 닫는 **가 문장부호 바로 뒤에 오고 공백 없이 한글이 이어지면
// "닫는 델리미터"로 인정되지 않아 **가 그대로 노출된다 — 공백 하나를 넣어 우회한다.
function fixEmphasisSpacing(text) {
  return text.replace(/([)\]"'”’])\*\*(?=[가-힣])/g, "$1** ");
}

function renderMarkdownInto(container, rawText) {
  const mathTokens = [];
  const protectedText = fixEmphasisSpacing(rawText).replace(/\$\$[\s\S]+?\$\$|\$[^\n$]+?\$/g, (match) => {
    mathTokens.push(match);
    return `@@MATH${mathTokens.length - 1}@@`;
  });

  let html = DOMPurify.sanitize(marked.parse(protectedText));
  html = html.replace(/@@MATH(\d+)@@/g, (_, i) => escapeHtml(mathTokens[Number(i)]));

  container.innerHTML = html;

  if (window.renderMathInElement) {
    renderMathInElement(container, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  }
}

// ---- corpus: seeds the empty state with example prompts, tags the rail with doc count ----
function pickSample(list, n) {
  if (list.length <= n) return list;
  const step = list.length / n;
  return Array.from({ length: n }, (_, i) => list[Math.floor(i * step)]);
}

async function loadCorpus() {
  try {
    const res = await fetch("/corpus");
    if (!res.ok) return;
    const { count, topics } = await res.json();

    if (count > 0) {
      corpusTag.textContent = `${count}개 파일`;
      corpusTag.hidden = false;
    }

    pickSample(topics, 3).forEach((topic) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip-example";
      btn.textContent = `${topic}에 대해 설명해줘`;
      examples.appendChild(btn);
    });
    if (topics.length > 0) examples.hidden = false;
  } catch (err) {
    // 코퍼스 정보는 부가 정보라 실패해도 대화 흐름엔 영향 없음
  }
}

// =====================================================================
// panel shell: rail tabs switch which <section class="panel"> is visible.
// standard roving-tabindex tab pattern — one tab is reachable at a time,
// arrow keys move focus and activate the panel together.
// =====================================================================
const railTabs = Array.from(document.querySelectorAll(".rail-tab"));
const panels = {
  chat: document.getElementById("panel-chat"),
  documents: document.getElementById("panel-documents"),
  search: document.getElementById("panel-search"),
};

function activatePanel(name) {
  railTabs.forEach((tab) => {
    const isActive = tab.dataset.panel === name;
    tab.setAttribute("aria-selected", String(isActive));
    tab.tabIndex = isActive ? 0 : -1;
  });
  Object.entries(panels).forEach(([key, el]) => {
    el.hidden = key !== name;
  });
  if (name === "documents") ensureDocumentsLoaded().catch(() => {});
  if (name === "search") searchInput.focus({ preventScroll: true });
  history.replaceState(null, "", name === "chat" ? "#" : `#${name}`);
}

railTabs.forEach((tab, i) => {
  tab.addEventListener("click", () => activatePanel(tab.dataset.panel));
  tab.addEventListener("keydown", (e) => {
    if (!["ArrowDown", "ArrowUp"].includes(e.key)) return;
    e.preventDefault();
    const dir = e.key === "ArrowDown" ? 1 : -1;
    const next = railTabs[(i + dir + railTabs.length) % railTabs.length];
    next.focus();
    activatePanel(next.dataset.panel);
  });
});

// ---- documents panel: list + reader ----
const docListEl = document.getElementById("doc-list");
const docReaderEmpty = document.getElementById("doc-reader-empty");
const docReaderBody = document.getElementById("doc-reader-body");
const docReaderScroll = document.getElementById("doc-reader");
const docCountTag = document.getElementById("doc-count-tag");

let documentsPromise = null;

// shaped like the doc-card it's about to become, not a spinner — so the list doesn't
// visually jump when the real cards land
function docListSkeleton(count = 4) {
  return Array.from({ length: count })
    .map(
      () => `
      <div class="skeleton-card" aria-hidden="true">
        <div class="skeleton-line w-60"></div>
        <div class="skeleton-line w-90"></div>
        <div class="skeleton-line w-40"></div>
      </div>`
    )
    .join("");
}

function ensureDocumentsLoaded() {
  if (documentsPromise) return documentsPromise;
  docListEl.innerHTML = docListSkeleton();
  documentsPromise = fetch("/documents")
    .then((res) => {
      if (!res.ok) throw new Error("failed to load documents");
      return res.json();
    })
    .then((docs) => {
      renderDocList(docs);
      docCountTag.textContent = `${docs.length}개 문서`;
      return docs;
    })
    .catch((err) => {
      docListEl.innerHTML = `<p class="doc-list-status status-error">문서를 불러오지 못했습니다.</p>`;
      documentsPromise = null;
      throw err;
    });
  return documentsPromise;
}

function renderDocList(docs) {
  docListEl.innerHTML = "";
  docs.forEach((doc) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "doc-card";
    card.dataset.docId = doc.id;
    card.setAttribute("aria-current", "false");
    card.innerHTML = `
      <p class="doc-card-title"></p>
      <p class="doc-card-excerpt"></p>
      <p class="doc-card-meta"></p>
    `;
    card.querySelector(".doc-card-title").textContent = doc.title;
    card.querySelector(".doc-card-excerpt").textContent = doc.excerpt;
    card.querySelector(".doc-card-meta").textContent = `${doc.char_count.toLocaleString()}자`;
    card.addEventListener("click", () => openDocument(doc.id));
    docListEl.appendChild(card);
  });
}

function markActiveDocCard(docId) {
  Array.from(docListEl.children).forEach((card) => {
    if (!card.dataset) return;
    card.setAttribute("aria-current", String(card.dataset.docId === docId));
  });
}

function docReaderSkeleton() {
  return `
    <div class="skeleton-card skeleton-card--plain" aria-hidden="true">
      <div class="skeleton-line w-40"></div>
      <div class="skeleton-line w-90"></div>
      <div class="skeleton-line w-90"></div>
      <div class="skeleton-line w-60"></div>
    </div>`;
}

async function openDocument(docId) {
  markActiveDocCard(docId);
  docReaderEmpty.hidden = true;
  docReaderBody.hidden = false;
  docReaderBody.innerHTML = docReaderSkeleton();

  try {
    const res = await fetch(`/documents/${encodeURIComponent(docId)}`);
    if (!res.ok) throw new Error("failed to load document");
    const doc = await res.json();

    docReaderBody.innerHTML = `
      <header class="doc-reader-head">
        <h2 class="doc-reader-title"></h2>
        <p class="doc-reader-file"></p>
      </header>
      <div class="agent-text doc-reader-content"></div>
    `;
    docReaderBody.querySelector(".doc-reader-title").textContent = doc.title;
    docReaderBody.querySelector(".doc-reader-file").textContent =
      `${doc.id}.md · ${doc.char_count.toLocaleString()}자`;
    renderMarkdownInto(docReaderBody.querySelector(".doc-reader-content"), doc.content);
    docReaderScroll.scrollTop = 0;
  } catch (err) {
    docReaderBody.innerHTML = `<p class="doc-reader-status status-error">문서를 불러오지 못했습니다.</p>`;
  }
}

async function goToDocument(docId) {
  activatePanel("documents");
  railTabs.forEach((tab) => {
    const isActive = tab.dataset.panel === "documents";
    tab.setAttribute("aria-selected", String(isActive));
    tab.tabIndex = isActive ? 0 : -1;
  });
  await ensureDocumentsLoaded().catch(() => {});
  openDocument(docId);
}

// ---- search panel ----
const searchInput = document.getElementById("search-input");
const searchResultsEl = document.getElementById("search-results");
let searchDebounce = null;

function highlightSnippet(snippet, start, end) {
  const safeStart = Math.max(0, Math.min(start, snippet.length));
  const safeEnd = Math.max(safeStart, Math.min(end, snippet.length));
  return (
    escapeHtml(snippet.slice(0, safeStart)) +
    "<mark>" + escapeHtml(snippet.slice(safeStart, safeEnd)) + "</mark>" +
    escapeHtml(snippet.slice(safeEnd))
  );
}

function renderSearchResults(hits, query) {
  if (hits.length === 0) {
    searchResultsEl.innerHTML = `<p class="search-hint">“${escapeHtml(query)}”에 대한 결과가 없습니다.</p>`;
    return;
  }
  searchResultsEl.innerHTML = "";
  hits.forEach((hit) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "search-hit";
    card.innerHTML = `
      <p class="search-hit-title"></p>
      <p class="search-hit-section"></p>
      <p class="search-hit-snippet"></p>
    `;
    card.querySelector(".search-hit-title").textContent = hit.title;
    card.querySelector(".search-hit-section").textContent = hit.section || "";
    card.querySelector(".search-hit-snippet").innerHTML = highlightSnippet(
      hit.snippet,
      hit.match_start,
      hit.match_end
    );
    card.addEventListener("click", () => goToDocument(hit.doc_id));
    searchResultsEl.appendChild(card);
  });
}

function searchSkeleton() {
  return Array.from({ length: 3 })
    .map(
      () => `
      <div class="skeleton-card" aria-hidden="true">
        <div class="skeleton-line w-40"></div>
        <div class="skeleton-line w-90"></div>
        <div class="skeleton-line w-60"></div>
      </div>`
    )
    .join("");
}

async function runSearch(query) {
  if (!query) {
    searchResultsEl.innerHTML = `<p class="search-hint">궁금한 개념을 입력해보세요.</p>`;
    return;
  }
  searchResultsEl.innerHTML = searchSkeleton();
  try {
    const res = await fetch(`/search?q=${encodeURIComponent(query)}`);
    if (!res.ok) throw new Error("search failed");
    const data = await res.json();
    renderSearchResults(data.hits, query);
  } catch (err) {
    searchResultsEl.innerHTML = `<p class="search-hint status-error">검색에 실패했습니다.</p>`;
  }
}

document.getElementById("search-form").addEventListener("submit", (e) => e.preventDefault());
searchInput.addEventListener("input", () => {
  clearTimeout(searchDebounce);
  const q = searchInput.value.trim();
  searchDebounce = setTimeout(() => runSearch(q), 300);
});

// =====================================================================
// chat panel — unchanged behavior, just lives inside #panel-chat now.
// every turn is a timeline row: a growth-line cell (dot + connector) driven purely by
// `kind`, and a content cell the caller fills in. the dot color is the single place
// turn-type is color-coded — keeps the content itself uncluttered.
// =====================================================================
function scrollToBottom() {
  transcript.scrollTop = transcript.scrollHeight;
}

function dismissEmptyState() {
  if (!emptyState.hidden) emptyState.hidden = true;
}

function createTurn(kind) {
  const turn = document.createElement("div");
  turn.className = `turn kind-${kind}`;
  turn.innerHTML = `
    <div class="turn-rail" aria-hidden="true"><span class="rail-dot"></span><span class="rail-line"></span></div>
    <div class="turn-content"></div>
  `;
  return turn;
}

function appendTurn(turn) {
  dismissEmptyState();
  transcriptInner.appendChild(turn);
  scrollToBottom();
  return turn;
}

function addUserTurn(text) {
  const turn = createTurn("user");
  const content = turn.querySelector(".turn-content");
  content.innerHTML = `<p class="turn-body"></p>`;
  content.querySelector(".turn-body").textContent = text;
  appendTurn(turn);
}

function addPendingTurn() {
  const turn = createTurn("pending");
  turn.querySelector(".turn-content").innerHTML = `<span class="pending-text">판단하는 중</span>`;
  return appendTurn(turn);
}

function addAgentTurn(data) {
  // /converse pushes every ToolMessage into saved_documents, including answer_question's —
  // only entries whose type is an actual writer tool represent a real filed note.
  const filedDocs = (data.saved_documents || []).filter((doc) => doc.type in TOOL_LABELS);
  const hasFiled = filedDocs.length > 0;
  const hasAnswer = Array.isArray(data.tools_used) && data.tools_used.includes("answer_question");
  const kind = hasFiled ? "filed" : hasAnswer ? "answer" : "plain";

  const turn = createTurn(kind);
  const content = turn.querySelector(".turn-content");

  if (hasFiled) {
    const stack = document.createElement("div");
    stack.className = "filed-stack";
    filedDocs.forEach((doc) => {
      const tab = document.createElement("div");
      tab.className = "filed-tab";
      tab.innerHTML = `<span class="filed-type"></span><span class="filed-name"></span>`;
      tab.querySelector(".filed-type").textContent = TOOL_LABELS[doc.type] || doc.type;
      tab.querySelector(".filed-name").textContent = doc.file_name;
      stack.appendChild(tab);
    });

    content.innerHTML = `<p class="turn-label">filed</p>`;
    content.appendChild(stack);
    const note = document.createElement("p");
    note.className = "agent-note";
    note.textContent = data.answer;
    content.appendChild(note);
  } else if (hasAnswer) {
    content.innerHTML = `
      <p class="turn-label is-answer">answer</p>
      <div class="agent-text"></div>
    `;
    renderMarkdownInto(content.querySelector(".agent-text"), data.answer);
  } else {
    content.innerHTML = `<div class="agent-text"></div>`;
    renderMarkdownInto(content.querySelector(".agent-text"), data.answer);
  }

  appendTurn(turn);
  announcer.textContent = hasFiled
    ? `저장을 완료했습니다. ${filedDocs.length}건.`
    : "답변을 표시했습니다.";
}

function addErrorTurn(message, retryText) {
  const turn = createTurn("error");
  const content = turn.querySelector(".turn-content");
  content.innerHTML = `<p class="error-text"></p><button type="button" class="retry-btn">다시 보내기</button>`;
  content.querySelector(".error-text").textContent = message;
  content.querySelector(".retry-btn").addEventListener("click", () => {
    turn.remove();
    requestReply(retryText);
  });
  appendTurn(turn);
  announcer.textContent = message;
}

// ---- send flow ----
let isPending = false;

// Talks to /converse for a given message. Does NOT add a user turn — the user turn
// for `message` must already be on screen (either just typed, or from the failed attempt
// a retry is re-trying), so retries don't duplicate the user bubble.
async function requestReply(message) {
  if (isPending) return;
  isPending = true;
  submitBtn.disabled = true;

  const pending = addPendingTurn();
  announcer.textContent = "에이전트가 판단하는 중입니다.";

  try {
    const res = await fetch("/converse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, thread_id: threadId }),
    });

    pending.remove();

    if (res.ok) {
      const data = await res.json();
      addAgentTurn(data);
    } else {
      addErrorTurn("응답 생성에 실패했습니다. 잠시 후 다시 시도하세요.", message);
    }
  } catch (err) {
    pending.remove();
    addErrorTurn("서버에 연결하지 못했습니다. 서버가 실행 중인지 확인하세요.", message);
  } finally {
    isPending = false;
    submitBtn.disabled = input.value.trim().length === 0;
  }
}

function sendMessage(message) {
  if (isPending) return;
  addUserTurn(message);
  requestReply(message);
}

// grows the textarea up to the CSS max-height (~3 lines) as the user types past one
// line, then lets it scroll internally — mirrors a chat composer, not a fixed-height field
function autoResizeInput() {
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message || isPending) return;
  input.value = "";
  submitBtn.disabled = true;
  autoResizeInput();
  sendMessage(message);
});

input.addEventListener("input", () => {
  submitBtn.disabled = isPending || input.value.trim().length === 0;
  autoResizeInput();
});

// Enter sends, Shift+Enter inserts a newline — textarea's own default (always a
// newline) would otherwise never submit the form
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

examples.addEventListener("click", (e) => {
  const btn = e.target.closest(".chip-example");
  if (!btn || isPending) return;
  sendMessage(btn.textContent);
});

// ---- new chat: fresh thread_id, cleared transcript. leaves the previous thread's
// history on the server untouched — it just stops being the one this tab talks to ----
const newChatBtn = document.getElementById("new-chat-btn");

function resetChat() {
  if (isPending) return;

  Array.from(transcriptInner.children).forEach((child) => {
    if (child !== emptyState) child.remove();
  });
  emptyState.hidden = false;

  threadId = crypto.randomUUID();
  localStorage.setItem(THREAD_KEY, threadId);

  input.value = "";
  autoResizeInput();
  submitBtn.disabled = true;
  announcer.textContent = "새 채팅을 시작했습니다.";
  input.focus();
}

newChatBtn.addEventListener("click", resetChat);

submitBtn.disabled = true;
loadCorpus();

// deep-linkable panels: #documents / #search open the panel, #documents/<id> opens a document
const [initialPanel, initialDocId] = location.hash.replace("#", "").split("/");
if (initialPanel === "documents" || initialPanel === "search") {
  activatePanel(initialPanel);
  if (initialPanel === "documents" && initialDocId) {
    ensureDocumentsLoaded().then(() => openDocument(initialDocId));
  }
}
