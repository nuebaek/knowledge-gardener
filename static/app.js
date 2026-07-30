const transcript = document.getElementById("transcript");
const transcriptInner = document.getElementById("transcript-inner");
const emptyState = document.getElementById("empty-state");
const examples = document.getElementById("examples");
const composer = document.getElementById("composer");
const input = document.getElementById("message");
const submitBtn = document.getElementById("submit-btn");
const announcer = document.getElementById("announcer");
const corpusTag = document.getElementById("corpus-tag");

marked.setOptions({ breaks: true, gfm: true });

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

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// CommonMark: closing ** right after punctuation with no space before Hangul isn't
// treated as a closer — add one to work around it.
function fixEmphasisSpacing(text) {
  return text.replace(/([)\]"'”’])\*\*(?=[가-힣])/g, "$1** ");
}

function formatDocDate(isoString) {
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString("ko-KR", { year: "numeric", month: "2-digit", day: "2-digit" });
}

function markRecallBlocks(container) {
  const isHeading = (el) => /^H[1-4]$/.test(el.tagName);
  Array.from(container.children).forEach((el) => {
    if (el.parentElement !== container) return; // already swept into an earlier wrapper
    if (!el.textContent.trim().startsWith("🌱")) return;

    const group = [el];
    if (isHeading(el)) {
      let next = el.nextElementSibling;
      while (next && next.tagName !== "HR" && !isHeading(next)) {
        group.push(next);
        next = next.nextElementSibling;
      }
    }

    const wrapper = document.createElement("div");
    wrapper.className = "recall-block";
    el.before(wrapper);
    group.forEach((node) => wrapper.appendChild(node));
  });
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
  markRecallBlocks(container);

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

// writer's TOC items are plain numbered text, not links — wire clicks to scroll to
// the matching heading.
function wireTocLinks(container) {
  const headings = Array.from(container.querySelectorAll("h1, h2, h3, h4, h5, h6"));
  if (headings.length === 0) return;

  const slugCounts = new Map();
  const labelOf = (text) => text.trim().replace(/^\d+(?:\.\d+)*\.?\s*/, "").toLowerCase();
  // li.textContent includes nested <ol> text too, so a parent item with children
  // never matched — strip nested lists first.
  const ownLabel = (li) => {
    const clone = li.cloneNode(true);
    clone.querySelectorAll("ol, ul").forEach((nested) => nested.remove());
    return labelOf(clone.textContent);
  };

  const headingByLabel = new Map();
  headings.forEach((h) => {
    const base = h.textContent.trim().toLowerCase().replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-+|-+$/g, "") || "section";
    const n = (slugCounts.get(base) || 0) + 1;
    slugCounts.set(base, n);
    h.id = n === 1 ? base : `${base}-${n}`;
    const label = labelOf(h.textContent);
    if (!headingByLabel.has(label)) headingByLabel.set(label, h);
  });

  headings.forEach((tocHeading) => {
    if (!/^(목차|table of contents)/i.test(tocHeading.textContent.trim())) return;
    let list = tocHeading.nextElementSibling;
    while (list && !["OL", "UL"].includes(list.tagName)) {
      if (/^H[1-6]$/.test(list.tagName)) return;
      list = list.nextElementSibling;
    }
    if (!list) return;
    list.querySelectorAll("li").forEach((li) => {
      const target = headingByLabel.get(ownLabel(li));
      if (!target) return;
      li.classList.add("toc-link");
      li.addEventListener("click", (e) => {
        // prevents a child click's scroll target from being overwritten by the
        // parent li's own handler
        e.stopPropagation();
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });
  });
}

function pickSample(list, n) {
  if (list.length <= n) return list;
  const step = list.length / n;
  return Array.from({ length: n }, (_, i) => list[Math.floor(i * step)]);
}

async function loadCorpus() {
  try {
    const [corpusRes, docsRes] = await Promise.all([fetch("/corpus"), fetch("/documents")]);

    if (docsRes.ok) {
      const docs = await docsRes.json();
      if (docs.length > 0) {
        corpusTag.textContent = `${docs.length}개 문서`;
        corpusTag.hidden = false;
      }
    }

    if (!corpusRes.ok) return;
    const { topics } = await corpusRes.json();

    pickSample(topics, 3).forEach((topic) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip-example";
      btn.textContent = `${topic}에 대해 설명해줘`;
      examples.appendChild(btn);
    });
    if (topics.length > 0) examples.hidden = false;
  } catch (err) {
    // corpus info is supplementary — a failed fetch shouldn't break the chat
  }
}

const railTabs = Array.from(document.querySelectorAll(".rail-tab"));
const railIndicator = document.getElementById("rail-indicator");
const panels = {
  chat: document.getElementById("panel-chat"),
  documents: document.getElementById("panel-documents"),
};

function activatePanel(name) {
  railTabs.forEach((tab) => {
    const isActive = tab.dataset.panel === name;
    tab.setAttribute("aria-selected", String(isActive));
    tab.tabIndex = isActive ? 0 : -1;
    if (isActive) moveRailIndicatorTo(tab);
  });
  Object.entries(panels).forEach(([key, el]) => {
    el.hidden = key !== name;
  });
  if (name === "documents") ensureDocumentsLoaded().catch(() => {});
  history.replaceState(null, "", name === "documents" ? "#" : `#${name}`);
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

function makeSpring({ damping = 1, response = 0.32 } = {}) {
  const angularFreq = (2 * Math.PI) / response;
  const stiffness = angularFreq * angularFreq;
  const dampingCoef = 2 * damping * Math.sqrt(stiffness);
  let value = 0;
  let velocity = 0;
  let target = 0;
  return {
    reset(v) {
      value = v;
      velocity = 0;
      target = v;
    },
    set(t) {
      target = t;
    },
    step(dt) {
      const force = -stiffness * (value - target) - dampingCoef * velocity;
      velocity += force * dt;
      value += velocity * dt;
      return value;
    },
    get value() {
      return value;
    },
    settled() {
      return Math.abs(target - value) < 0.5 && Math.abs(velocity) < 0.5;
    },
  };
}

const railSpring = makeSpring();
let railAnimHandle = null;
let railAxis = "y";
const railMobileQuery = window.matchMedia("(max-width: 40rem)");

function railTargetFor(tab) {
  if (!tab) return 0;
  return railAxis === "y"
    ? tab.offsetTop + tab.offsetHeight / 2 - railIndicator.offsetHeight / 2
    : tab.offsetLeft + tab.offsetWidth / 2 - railIndicator.offsetWidth / 2;
}

function paintRailIndicator(v) {
  railIndicator.style.transform = railAxis === "y" ? `translateY(${v}px)` : `translateX(${v}px)`;
}

function runRailSpring() {
  if (railAnimHandle != null) return;
  let last = performance.now();
  const frame = (now) => {
    const dt = Math.min((now - last) / 1000, 1 / 30);
    last = now;
    paintRailIndicator(railSpring.step(dt));
    railAnimHandle = railSpring.settled() ? null : requestAnimationFrame(frame);
  };
  railAnimHandle = requestAnimationFrame(frame);
}

const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function moveRailIndicatorTo(tab) {
  const target = railTargetFor(tab);
  if (prefersReducedMotion.matches) {
    railSpring.reset(target);
    paintRailIndicator(target);
    return;
  }
  railSpring.set(target);
  runRailSpring();
}

// axis flips (vertical <-> horizontal) snap instead of springing — animating
// across the jump would look like a diagonal glitch
function snapRailIndicator() {
  railAxis = railMobileQuery.matches ? "x" : "y";
  railIndicator.classList.toggle("rail-indicator--horizontal", railAxis === "x");
  const active = railTabs.find((t) => t.getAttribute("aria-selected") === "true") || railTabs[0];
  const target = railTargetFor(active);
  railSpring.reset(target);
  paintRailIndicator(target);
}

window.addEventListener("resize", snapRailIndicator);
railMobileQuery.addEventListener("change", snapRailIndicator);
snapRailIndicator();
if (document.fonts && document.fonts.ready) document.fonts.ready.then(snapRailIndicator);

const docListEl = document.getElementById("doc-list");
const docReaderEmpty = document.getElementById("doc-reader-empty");
const docReaderBody = document.getElementById("doc-reader-body");
const docReaderScroll = document.getElementById("doc-reader");
const docCountTag = document.getElementById("doc-count-tag");
const docFolderTabs = document.getElementById("doc-folder-tabs");
const tagFilterPill = document.getElementById("tag-filter-pill");
const tagFilterLabel = document.getElementById("tag-filter-label");

// doc_type comes straight from the storage path — "processed" isn't always CS231n
// (any file through scripts/preprocess.py lands there), so don't hardcode a label.
const DOC_TYPE_LABELS = { dailynote: "데일리노트", weeklynote: "위클리노트", til: "TIL", processed: "코퍼스" };

let documentsPromise = null;
let allDocuments = [];
let activeDocType = null;
let activeTag = null;
let activeQuery = "";
let searchDebounce = null;
let searchRequestId = 0;

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
      allDocuments = docs;
      renderFolderTabs();
      applyFilters();
      return docs;
    })
    .catch((err) => {
      docListEl.innerHTML = `<p class="doc-list-status status-error">문서를 불러오지 못했습니다.</p>`;
      documentsPromise = null;
      throw err;
    });
  return documentsPromise;
}

function renderFolderTabs() {
  const types = [...new Set(allDocuments.map((d) => d.doc_type))];
  docFolderTabs.innerHTML = "";
  const makeTab = (value, label) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "folder-tab";
    btn.textContent = label;
    btn.setAttribute("aria-selected", String(activeDocType === value));
    btn.addEventListener("click", () => {
      clearSearch();
      activeDocType = value;
      renderFolderTabs();
      applyFilters();
    });
    docFolderTabs.appendChild(btn);
  };
  makeTab(null, "전체");
  types.forEach((t) => makeTab(t, DOC_TYPE_LABELS[t] || t));
}

// search and folder/tag browsing are mutually exclusive modes
function applyFilters() {
  if (activeQuery) {
    runDocSearch(activeQuery);
    return;
  }
  const filtered = allDocuments.filter((d) => {
    if (activeDocType && d.doc_type !== activeDocType) return false;
    if (activeTag && !d.tags.includes(activeTag)) return false;
    return true;
  });
  renderDocList(filtered);
  docCountTag.textContent = `${filtered.length}개 문서`;
}

function renderTagFilterPill() {
  if (!activeTag) {
    tagFilterPill.hidden = true;
    return;
  }
  tagFilterLabel.textContent = `#${activeTag}`;
  tagFilterPill.hidden = false;
}

// resets the folder filter to "전체" so the document being read doesn't
// disappear from the list
function setTagFilter(tag) {
  clearSearch();
  activeTag = tag;
  activeDocType = null;
  renderFolderTabs();
  renderTagFilterPill();
  applyFilters();
}

function clearTagFilter() {
  activeTag = null;
  renderTagFilterPill();
  applyFilters();
}

tagFilterPill.addEventListener("click", clearTagFilter);

function renderDocList(docs) {
  docListEl.innerHTML = "";
  if (docs.length === 0) {
    docListEl.innerHTML = `<p class="doc-list-status">이 폴더에는 문서가 없습니다.</p>`;
    return;
  }
  docs.forEach((doc, i) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "doc-card";
    card.dataset.docId = doc.id;
    card.setAttribute("aria-current", "false");
    card.style.setProperty("--i", Math.min(i, 10));
    card.innerHTML = `
      <p class="doc-card-title"></p>
      <p class="doc-card-excerpt"></p>
      <p class="doc-card-meta"></p>
      <div class="doc-card-tags"></div>
    `;
    card.querySelector(".doc-card-title").textContent = doc.title;
    card.querySelector(".doc-card-excerpt").textContent = doc.excerpt;
    card.querySelector(".doc-card-meta").textContent = formatDocDate(doc.created_at);
    const tagsEl = card.querySelector(".doc-card-tags");
    doc.tags.forEach((tag) => {
      const chip = document.createElement("span");
      chip.className = "tag-chip";
      chip.textContent = tag;
      tagsEl.appendChild(chip);
    });
    card.addEventListener("click", () => openDocument(doc.id));
    docListEl.appendChild(card);
  });
}

function syncDocTagsCache(docId, tags) {
  const cached = allDocuments.find((d) => d.id === docId);
  if (cached) cached.tags = tags;
  applyFilters(); // clears the active-card highlight, so re-mark it right after
  markActiveDocCard(docId);
}

function renderReaderTags(docId, tags, newTag) {
  const listEl = docReaderBody.querySelector("#reader-tag-list");
  listEl.innerHTML = "";
  tags.forEach((tag) => {
    const chip = document.createElement("span");
    chip.className = "tag-chip tag-chip--removable";
    if (tag === activeTag) chip.classList.add("is-active-filter");
    if (tag === newTag) chip.classList.add("is-new");
    chip.innerHTML = `<button type="button" class="tag-chip-label"></button><button type="button" class="tag-chip-remove" aria-label="${tag} 태그 삭제">×</button>`;
    chip.querySelector(".tag-chip-label").textContent = tag;
    chip.querySelector(".tag-chip-label").setAttribute(
      "aria-label",
      `${tag} 태그가 달린 문서만 보기`
    );
    chip.querySelector(".tag-chip-label").addEventListener("click", () => {
      if (tag === activeTag) clearTagFilter();
      else setTagFilter(tag);
      renderReaderTags(docId, tags);
    });
    chip.querySelector(".tag-chip-remove").addEventListener("click", async () => {
      const res = await fetch(`/documents/${encodeURIComponent(docId)}/tags/${encodeURIComponent(tag)}`, {
        method: "DELETE",
      });
      if (res.ok) {
        const updated = await res.json();
        if (tag === activeTag) clearTagFilter();
        renderReaderTags(docId, updated);
        syncDocTagsCache(docId, updated);
      }
    });
    listEl.appendChild(chip);
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

async function openDocument(docId, scrollQuery) {
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
        <p class="doc-reader-eyebrow"></p>
        <h2 class="doc-reader-title"></h2>
      </header>
      <div class="doc-reader-tags">
        <div class="tag-chip-list" id="reader-tag-list"></div>
        <form class="tag-add-form" id="tag-add-form">
          <button type="button" class="tag-add-chip" id="tag-add-trigger">+ 태그</button>
          <input type="text" id="tag-add-input" class="tag-add-input" placeholder="태그 이름" maxlength="30" autocomplete="off" hidden />
        </form>
      </div>
      <div class="agent-text doc-reader-content"></div>
    `;
    docReaderBody.querySelector(".doc-reader-title").textContent = doc.title;
    docReaderBody.querySelector(".doc-reader-eyebrow").textContent =
      `${DOC_TYPE_LABELS[doc.doc_type] || doc.doc_type} · ${formatDocDate(doc.created_at)}`;
    const contentEl = docReaderBody.querySelector(".doc-reader-content");
    renderMarkdownInto(contentEl, doc.content);
    wireTocLinks(contentEl);
    renderReaderTags(doc.id, doc.tags);
    // force a reflow so removing+re-adding the class restarts the animation
    docReaderBody.classList.remove("is-entering");
    void docReaderBody.offsetWidth;
    docReaderBody.classList.add("is-entering");

    const tagAddForm = docReaderBody.querySelector("#tag-add-form");
    const tagAddTrigger = docReaderBody.querySelector("#tag-add-trigger");
    const tagAddInput = docReaderBody.querySelector("#tag-add-input");

    const showTagInput = () => {
      tagAddTrigger.hidden = true;
      tagAddInput.hidden = false;
      tagAddInput.focus();
    };
    const resetTagInput = () => {
      tagAddInput.value = "";
      tagAddInput.hidden = true;
      tagAddTrigger.hidden = false;
    };

    tagAddTrigger.addEventListener("click", showTagInput);
    tagAddInput.addEventListener("keydown", (e) => {
      if (e.key === "Escape") resetTagInput();
    });
    tagAddInput.addEventListener("blur", () => {
      if (!tagAddInput.value) resetTagInput();
    });

    tagAddForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const name = tagAddInput.value.trim();
      if (!name) return;
      const res = await fetch(`/documents/${encodeURIComponent(doc.id)}/tags`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (res.ok) {
        const updated = await res.json();
        resetTagInput();
        renderReaderTags(doc.id, updated, name);
        syncDocTagsCache(doc.id, updated);
      }
    });
    docReaderScroll.scrollTop = 0;
    if (scrollQuery) scrollToQueryMatch(contentEl, scrollQuery);
  } catch (err) {
    docReaderBody.innerHTML = `<p class="doc-reader-status status-error">문서를 불러오지 못했습니다.</p>`;
  }
}

async function goToDocument(docId, scrollQuery) {
  activatePanel("documents");
  railTabs.forEach((tab) => {
    const isActive = tab.dataset.panel === "documents";
    tab.setAttribute("aria-selected", String(isActive));
    tab.tabIndex = isActive ? 0 : -1;
  });
  await ensureDocumentsLoaded().catch(() => {});
  openDocument(docId, scrollQuery);
}

// walks text nodes instead of regexing the HTML string, which would break tags
function scrollToQueryMatch(container, query) {
  const q = query.trim().toLowerCase();
  if (!q) return;
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    const idx = node.textContent.toLowerCase().indexOf(q);
    if (idx === -1) continue;
    const range = document.createRange();
    range.setStart(node, idx);
    range.setEnd(node, idx + q.length);
    const mark = document.createElement("mark");
    mark.className = "jump-highlight";
    range.surroundContents(mark);
    mark.scrollIntoView({ behavior: "smooth", block: "center" });
    setTimeout(() => {
      const parent = mark.parentNode;
      if (!parent) return;
      parent.replaceChild(document.createTextNode(mark.textContent), mark);
      parent.normalize();
    }, 2600);
    return;
  }
}

const docSearchInput = document.getElementById("doc-search-input");

function highlightSnippet(snippet, start, end) {
  const safeStart = Math.max(0, Math.min(start, snippet.length));
  const safeEnd = Math.max(safeStart, Math.min(end, snippet.length));
  return (
    escapeHtml(snippet.slice(0, safeStart)) +
    "<mark>" + escapeHtml(snippet.slice(safeStart, safeEnd)) + "</mark>" +
    escapeHtml(snippet.slice(safeEnd))
  );
}

function clearSearch() {
  activeQuery = "";
  docSearchInput.value = "";
}

function renderDocSearchHits(hits, query) {
  if (hits.length === 0) {
    docListEl.innerHTML = `<p class="doc-list-status">“${escapeHtml(query)}”에 대한 결과가 없습니다.</p>`;
    return;
  }
  docListEl.innerHTML = "";
  hits.forEach((hit, i) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "doc-search-hit";
    card.style.setProperty("--i", Math.min(i, 10));
    card.innerHTML = `
      <p class="doc-search-hit-title"></p>
      <p class="doc-search-hit-section"></p>
      <p class="doc-search-hit-snippet"></p>
    `;
    card.querySelector(".doc-search-hit-title").textContent = hit.title;
    card.querySelector(".doc-search-hit-section").textContent = hit.section || "";
    card.querySelector(".doc-search-hit-snippet").innerHTML = highlightSnippet(
      hit.snippet,
      hit.match_start,
      hit.match_end
    );
    card.addEventListener("click", () => goToDocument(hit.doc_id, query));
    docListEl.appendChild(card);
  });
}

// tags each request so a slow response can't overwrite a newer query's results
async function runDocSearch(query) {
  const requestId = ++searchRequestId;
  docListEl.innerHTML = docListSkeleton(3);
  try {
    const res = await fetch(`/search?q=${encodeURIComponent(query)}`);
    if (!res.ok) throw new Error("search failed");
    const data = await res.json();
    if (requestId !== searchRequestId) return;
    renderDocSearchHits(data.hits, query);
    docCountTag.textContent = `${data.hits.length}개 결과`;
  } catch (err) {
    if (requestId !== searchRequestId) return;
    docListEl.innerHTML = `<p class="doc-list-status status-error">검색에 실패했습니다.</p>`;
  }
}

docSearchInput.addEventListener("input", () => {
  clearTimeout(searchDebounce);
  const q = docSearchInput.value.trim();
  searchDebounce = setTimeout(() => {
    activeQuery = q;
    applyFilters();
  }, 300);
});

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
  content.innerHTML = `<p class="turn-label">you</p><p class="turn-body"></p>`;
  content.querySelector(".turn-body").textContent = text;
  appendTurn(turn);
}

function addPendingTurn() {
  const turn = createTurn("pending");
  turn.querySelector(".turn-content").innerHTML = `
    <span class="typing-dots" aria-hidden="true"><span></span><span></span><span></span></span>
    <span class="pending-text">생각하는 중</span>
  `;
  return appendTurn(turn);
}

function addAgentTurn(data) {
  const filedDocs = data.saved_documents || [];
  const hasFiled = filedDocs.length > 0;
  const hasAnswer = Array.isArray(data.tools_used) && data.tools_used.includes("answer_question");
  const hasMindmap = Boolean(data.mindmap_plaintext);
  const kind = hasMindmap ? "mindmap" : hasFiled ? "filed" : hasAnswer ? "answer" : "plain";

  const turn = createTurn(kind);
  const content = turn.querySelector(".turn-content");

  if (hasMindmap) {
    content.innerHTML = `
      <p class="turn-label">mindmap</p>
      <div class="mindmap-mount"></div>
    `;
    const mount = content.querySelector(".mindmap-mount");
    // render is async, so scroll again once it's actually mounted
    window.renderMindmapInto(mount, data.mindmap_plaintext).then(scrollToBottom);
  } else if (hasFiled) {
    const stack = document.createElement("div");
    stack.className = "filed-stack";
    filedDocs.forEach((doc) => {
      const tab = document.createElement("div");
      tab.className = "filed-tab";
      tab.innerHTML = `<span class="filed-type"></span><span class="filed-name"></span>`;
      tab.querySelector(".filed-type").textContent = DOC_TYPE_LABELS[doc.type] || doc.type;
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
    const sources = data.sources || [];
    if (sources.length) {
      const list = document.createElement("div");
      list.className = "sources";
      sources.forEach((src) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "source-chip";
        chip.innerHTML = `<span class="source-chip-dot"></span>`;
        chip.append(src);
        // doc_id is the same source_path chain.py stamps on vector chunks — opens
        // directly, no lookup needed
        chip.addEventListener("click", () => goToDocument(src));
        list.appendChild(chip);
      });
      content.appendChild(list);
    }
  } else {
    content.innerHTML = `
      <p class="turn-label">gardener</p>
      <div class="agent-text"></div>
    `;
    renderMarkdownInto(content.querySelector(".agent-text"), data.answer);
  }

  const recall = data.recall || [];
  if (recall.length) {
    const block = document.createElement("div");
    block.className = "recall-block";
    block.innerHTML = `<span aria-hidden="true">🌱</span> <b>다시 꺼내볼 것</b> — `;
    block.append(recall.join(", "));
    content.appendChild(block);
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

let isPending = false;

// doesn't add the user turn itself — caller must already have it on screen
// (typed, or a retry) so retries don't duplicate the bubble
async function requestReply(message) {
  if (isPending) return;
  isPending = true;
  submitBtn.disabled = true;

  const pending = addPendingTurn();
  announcer.textContent = "에이전트가 생각하는 중입니다.";

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

// isComposing / keyCode 229: an IME (e.g. Hangul) also fires Enter to confirm the
// character being composed — without this guard, submitting mid-composition drops
// the last character
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

examples.addEventListener("click", (e) => {
  const btn = e.target.closest(".chip-example");
  if (!btn || isPending) return;
  sendMessage(btn.textContent);
});

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

const [initialPanel, initialDocId] = location.hash.replace("#", "").split("/");
if (initialPanel === "chat") {
  activatePanel("chat");
} else {
  ensureDocumentsLoaded().catch(() => {});
  if (initialDocId) ensureDocumentsLoaded().then(() => openDocument(initialDocId));
}
