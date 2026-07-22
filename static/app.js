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

// catalog.py는 created_at을 UTC ISO 문자열로 저장한다 — 화면엔 글자수 대신 이 날짜를 보여준다.
function formatDocDate(isoString) {
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString("ko-KR", { year: "numeric", month: "2-digit", day: "2-digit" });
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

// writer 문서는 "# 목차" 헤딩 아래 평문 번호 목록("1. fp32")으로 목차를 쓰고, 그 아래
// 본문 헤딩도 같은 텍스트("# 1. fp32")로 나온다 — 마크다운 링크가 아니라서 클릭해도 아무 일도
// 안 났다. 헤딩마다 id를 붙이고 목차 항목 텍스트를 헤딩 텍스트와 매칭해서 클릭 시 스크롤되게 한다.
function wireTocLinks(container) {
  const headings = Array.from(container.querySelectorAll("h1, h2, h3, h4, h5, h6"));
  if (headings.length === 0) return;

  const slugCounts = new Map();
  const labelOf = (text) => text.trim().replace(/^\d+(?:\.\d+)*\.?\s*/, "").toLowerCase();
  // daily note의 중첩 목차(부모 항목 아래 하위 개념)에서 li.textContent는 중첩된 <ol>의
  // 텍스트까지 그대로 이어붙여 돌려준다(예: "Docker" 항목 밑에 "Dockerfile"/"Image"가 중첩돼
  // 있으면 "DockerDockerfileImage..."가 됨) — 그래서 하위 항목이 있는 부모 항목만 매칭이
  // 항상 실패했다. 중첩 리스트를 떼어낸 사본에서 읽어야 그 li 자신의 라벨만 나온다.
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
      if (/^H[1-6]$/.test(list.tagName)) return; // 목차 헤딩 바로 다음에 리스트가 없으면 포기
      list = list.nextElementSibling;
    }
    if (!list) return;
    list.querySelectorAll("li").forEach((li) => {
      const target = headingByLabel.get(ownLabel(li));
      if (!target) return;
      li.classList.add("toc-link");
      li.addEventListener("click", (e) => {
        // 하위 항목 클릭이 부모 li까지 버블링되면 부모 항목의 핸들러가 나중에 또 실행되면서
        // 스크롤 위치가 부모 섹션으로 덮어써진다 — 부모 항목도 이제 정상 매칭되므로 막아야 함.
        e.stopPropagation();
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });
  });
}

// ---- corpus: seeds the empty state with example prompts, tags the rail with doc count ----
function pickSample(list, n) {
  if (list.length <= n) return list;
  const step = list.length / n;
  return Array.from({ length: n }, (_, i) => list[Math.floor(i * step)]);
}

// rail-foot 태그는 /corpus의 count(= RAG에 색인된 코퍼스 문서 수, doc_type="processed"만
// 집계 — 아직 재인덱싱되지 않는 데일리/위클리/TIL은 여기 안 잡힘, README 로드맵의
// "학습 기록 재인덱싱" 항목 참고)가 아니라 /documents 전체 개수를 쓴다. 안 그러면
// 매일 노트를 써도 이 숫자가 그대로라 "업데이트가 안 된다"는 오해를 산다.
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
    // 코퍼스 정보는 부가 정보라 실패해도 대화 흐름엔 영향 없음
  }
}

// =====================================================================
// panel shell: rail tabs switch which <section class="panel"> is visible.
// standard roving-tabindex tab pattern — one tab is reachable at a time,
// arrow keys move focus and activate the panel together.
// =====================================================================
const railTabs = Array.from(document.querySelectorAll(".rail-tab"));
const railIndicator = document.getElementById("rail-indicator");
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
    if (isActive) moveRailIndicatorTo(tab);
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

// ---- rail indicator: one shared bar springs to the active tab, instead of a fresh
// element popping in/out per tab — the physical "selection glides over" feel of a
// segmented control or a Dock highlight. Apple's damping/response model (see
// apple-design skill notes), implemented as a tiny critically-damped spring since this
// project has no animation library. Retargeting mid-flight (a second click before the
// first settles) just moves `target` — the existing velocity carries through, so it's
// interruptible rather than restarting from a hard stop. ----
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
let railAxis = "y"; // "y" = desktop vertical rail, "x" = mobile top-bar
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

// resize/orientation changes swap the indicator's axis entirely (top-to-bottom vs
// left-to-right) — springing across that jump would look like a diagonal glitch, so
// these snap instead of animating.
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
// webfonts finishing their swap can nudge tab widths by a px or two after first paint
if (document.fonts && document.fonts.ready) document.fonts.ready.then(snapRailIndicator);

// ---- documents panel: list + reader ----
const docListEl = document.getElementById("doc-list");
const docReaderEmpty = document.getElementById("doc-reader-empty");
const docReaderBody = document.getElementById("doc-reader-body");
const docReaderScroll = document.getElementById("doc-reader");
const docCountTag = document.getElementById("doc-count-tag");
const docFolderTabs = document.getElementById("doc-folder-tabs");
const tagFilterPill = document.getElementById("tag-filter-pill");
const tagFilterLabel = document.getElementById("tag-filter-label");

// doc_type은 저장 경로(data/writer/dailynote 등)에서 그대로 온 값 — 화면 표시용 한글 라벨만 매핑.
// "processed"는 scripts/preprocess.py가 data/raw/* 를 무엇이든 markitdown으로 변환해
// data/processed/*.md 에 넣으면 그대로 붙는 이름이라 "강의자료"로 못 박으면 안 됨 —
// 지금까지 CS231n 강의노트만 넣어봐서 우연히 그렇게 보였을 뿐, 논문이든 뭐든 같은
// 파이프라인을 타면 여기로 들어온다. 그래서 문서함 subtitle과 같은 용어인 "코퍼스"로 표기.
const DOC_TYPE_LABELS = { dailynote: "데일리노트", weeklynote: "위클리노트", til: "TIL", processed: "코퍼스" };

let documentsPromise = null;
let allDocuments = []; // 폴더 탭을 서버 왕복 없이 즉시 전환하려고 전체를 한 번만 받아 클라이언트에서 필터링
let activeDocType = null; // null = 전체
let activeTag = null; // null = 태그 필터 없음, 리더에서 태그를 클릭하면 설정됨

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
      activeDocType = value;
      renderFolderTabs();
      applyFilters();
    });
    docFolderTabs.appendChild(btn);
  };
  makeTab(null, "전체");
  types.forEach((t) => makeTab(t, DOC_TYPE_LABELS[t] || t));
}

// 폴더(doc_type)와 태그, 두 축을 AND로 겹쳐서 걸러낸다 — 리더에서 태그를 클릭해도
// 폴더 탭 상태는 그대로 두고 목록만 좁혀지도록.
function applyFilters() {
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

// 리더에서 태그 라벨을 클릭하면 호출됨 — 지금 읽고 있는 문서가 필터 결과에서 사라지지
// 않도록 폴더 필터는 "전체"로 풀어준다 (그 문서가 다른 폴더에 있었더라도 보이게).
function setTagFilter(tag) {
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

// 문서를 읽는 그 자리에서 바로 태그를 붙이고 뗄 수 있게 — 목록으로 안 돌아가도 됨.
// allDocuments는 세션당 한 번만 받아온 캐시라, 여기서 바꾼 태그를 그 캐시에도 반영해줘야
// 목록으로 돌아갔을 때 카드의 태그 칩이 방금 편집한 내용과 어긋나지 않는다.
function syncDocTagsCache(docId, tags) {
  const cached = allDocuments.find((d) => d.id === docId);
  if (cached) cached.tags = tags;
  applyFilters(); // 카드 태그 칩을 최신 상태로 다시 그리는데, 이때 현재 열려있는 카드의 강조가 풀림
  markActiveDocCard(docId); // 그래서 바로 다시 표시해줌
}

function renderReaderTags(docId, tags, newTag) {
  const listEl = docReaderBody.querySelector("#reader-tag-list");
  listEl.innerHTML = "";
  tags.forEach((tag) => {
    const chip = document.createElement("span");
    chip.className = "tag-chip tag-chip--removable";
    if (tag === activeTag) chip.classList.add("is-active-filter");
    if (tag === newTag) chip.classList.add("is-new"); // 방금 추가된 칩만 pill-in으로 등장
    chip.innerHTML = `<button type="button" class="tag-chip-label"></button><button type="button" class="tag-chip-remove" aria-label="${tag} 태그 삭제">×</button>`;
    chip.querySelector(".tag-chip-label").textContent = tag;
    chip.querySelector(".tag-chip-label").setAttribute(
      "aria-label",
      `${tag} 태그가 달린 문서만 보기`
    );
    chip.querySelector(".tag-chip-label").addEventListener("click", () => {
      if (tag === activeTag) clearTagFilter();
      else setTagFilter(tag);
      renderReaderTags(docId, tags); // 방금 누른 칩의 강조 상태(is-active-filter)를 즉시 반영
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
    docReaderBody.querySelector(".doc-reader-file").textContent =
      `${doc.id} · ${formatDocDate(doc.created_at)}`;
    const contentEl = docReaderBody.querySelector(".doc-reader-content");
    renderMarkdownInto(contentEl, doc.content);
    wireTocLinks(contentEl);
    renderReaderTags(doc.id, doc.tags);
    // restart the settle-in animation on every load — the class was already removed
    // by the skeleton swap above, so a fresh reflow + re-add is enough to retrigger it
    docReaderBody.classList.remove("is-entering");
    void docReaderBody.offsetWidth;
    docReaderBody.classList.add("is-entering");

    // 태그 추가 UI: "+ 태그" 고스트 칩을 클릭하면 같은 자리에서 칩 모양 입력으로 바뀜 —
    // 별도 제출 버튼 없이 Enter로 저장, Escape/빈 채로 blur하면 다시 고스트 칩으로 되돌아감
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
      if (!tagAddInput.value) resetTagInput(); // 타이핑 중인 내용은 blur로 지우지 않음
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
  hits.forEach((hit, i) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "search-hit";
    card.style.setProperty("--i", Math.min(i, 10));
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
    // renderMindmapInto가 mind-elixir를 처음 쓸 때만 동적으로 로드하는데, 그 다운로드가
    // 끝나기 전에 이 턴이 스크롤 밖으로 나가있을 수 있어 렌더 완료 후 다시 스크롤해준다.
    window.renderMindmapInto(mount, data.mindmap_plaintext).then(scrollToBottom);
  } else if (hasFiled) {
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
