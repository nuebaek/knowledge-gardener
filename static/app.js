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

// ---- thread_id: created once, kept for the life of this browser's journal ----
function getThreadId() {
  const KEY = "kaia:thread_id";
  let id = localStorage.getItem(KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(KEY, id);
  }
  return id;
}

const threadId = getThreadId();

// ---- markdown + LaTeX rendering (same escape hatch as the old /ask UI) ----
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

// ---- corpus: seeds the empty state with example prompts ----
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
      corpusTag.textContent = `색인 문서 ${count}개`;
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

// ---- transcript rendering ----
// every turn is a timeline row: a growth-line cell (dot + connector) driven purely by
// `kind`, and a content cell the caller fills in. the dot color is the single place
// turn-type is color-coded — keeps the content itself uncluttered.
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

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message || isPending) return;
  input.value = "";
  submitBtn.disabled = true;
  sendMessage(message);
});

input.addEventListener("input", () => {
  submitBtn.disabled = isPending || input.value.trim().length === 0;
});

examples.addEventListener("click", (e) => {
  const btn = e.target.closest(".chip-example");
  if (!btn || isPending) return;
  sendMessage(btn.textContent);
});

submitBtn.disabled = true;
loadCorpus();
