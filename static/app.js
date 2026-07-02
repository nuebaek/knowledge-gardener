const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const submitBtn = document.getElementById("submit-btn");
const examples = document.getElementById("examples");
const flow = document.getElementById("flow");
const status = document.getElementById("status");
const result = document.getElementById("result");
const answerText = document.getElementById("answer-text");
const sources = document.getElementById("sources");
const statePanel = document.getElementById("state-panel");
const stateLabel = document.getElementById("state-label");
const stateText = document.getElementById("state-text");
const announcer = document.getElementById("announcer");
const corpusNote = document.getElementById("corpus-note");

marked.setOptions({ breaks: true, gfm: true });

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// CommonMark 규칙상 닫는 **가 문장부호(예: ")") 바로 뒤에 오고 공백 없이
// 한글이 바로 이어지면 "닫는 델리미터"로 인정되지 않아 **가 그대로 노출된다
// (예: "**A(B)**입니다" → 굵게 안 됨). 공백 하나를 넣어 우회한다.
function fixEmphasisSpacing(text) {
  return text.replace(/([)\]"'”’])\*\*(?=[가-힣])/g, "$1** ");
}

// LaTeX($...$, $$...$$)는 markdown 파서가 모르는 문법이라 그대로 파싱하면
// 밑줄(_)이 이탤릭으로 오인되는 등 깨질 수 있어, 마크다운 변환 전에 토큰으로
// 빼뒀다가 렌더링 후 복원하고 KaTeX로 별도 typeset한다.
function renderAnswerMarkdown(rawText) {
  const mathTokens = [];
  const protectedText = fixEmphasisSpacing(rawText).replace(/\$\$[\s\S]+?\$\$|\$[^\n$]+?\$/g, (match) => {
    mathTokens.push(match);
    return `@@MATH${mathTokens.length - 1}@@`;
  });

  let html = DOMPurify.sanitize(marked.parse(protectedText));
  html = html.replace(/@@MATH(\d+)@@/g, (_, i) => escapeHtml(mathTokens[Number(i)]));

  answerText.innerHTML = html;

  if (window.renderMathInElement) {
    renderMathInElement(answerText, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  }
}

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
      corpusNote.textContent = `현재 색인 문서: ${count}개`;
      corpusNote.hidden = false;
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
    // 코퍼스 정보는 부가 정보라 실패해도 질문 흐름엔 영향 없음
  }
}

function resetPanels() {
  flow.classList.remove("is-loading", "is-settled");
  flow.hidden = false;
  status.hidden = true;
  result.hidden = true;
  statePanel.hidden = true;
  statePanel.classList.remove("is-error");
}

function sourceLabel(path) {
  const stem = path.split("/").pop().replace(/\.md$/i, "");
  return stem
    .split(/[-_]/)
    .map((word) => (/^\d+$/.test(word) ? word : word.charAt(0).toUpperCase() + word.slice(1)))
    .join(" ");
}

function showResult(data) {
  flow.classList.remove("is-loading");
  flow.classList.add("is-settled");
  status.hidden = true;

  renderAnswerMarkdown(data.answer);
  sources.innerHTML = "";
  data.sources.forEach((src, i) => {
    const chip = document.createElement("span");
    chip.className = "source-chip";
    chip.innerHTML = `<span class="idx">[${i + 1}]</span> ${sourceLabel(src)}`;
    sources.appendChild(chip);
  });
  result.hidden = false;
  announcer.textContent = `답변을 표시했습니다. 출처 ${data.sources.length}건.`;
}

function showState({ label, text, isError }) {
  flow.classList.remove("is-loading", "is-settled");
  flow.hidden = true;
  status.hidden = true;
  stateLabel.textContent = label;
  stateText.textContent = text;
  statePanel.classList.toggle("is-error", Boolean(isError));
  statePanel.hidden = false;
  announcer.textContent = text;
}

async function ask(question) {
  resetPanels();
  flow.classList.add("is-loading");
  status.hidden = false;
  submitBtn.disabled = true;
  announcer.textContent = "검색 중입니다.";

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (res.ok) {
      const data = await res.json();
      showResult(data);
      return;
    }

    if (res.status === 404) {
      showState({
        label: "no match",
        text: "이 질문에 대한 근거를 색인된 문서에서 찾지 못했습니다. 다른 표현으로 다시 물어보세요.",
      });
      return;
    }

    showState({
      label: "pipeline error",
      text: "답변 생성 파이프라인에 문제가 발생했습니다. 잠시 후 다시 시도해주세요.",
      isError: true,
    });
  } catch (err) {
    showState({
      label: "connection error",
      text: "서버에 연결하지 못했습니다. 서버가 실행 중인지 확인해주세요.",
      isError: true,
    });
  } finally {
    submitBtn.disabled = input.value.trim().length === 0;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  ask(question);
});

input.addEventListener("input", () => {
  submitBtn.disabled = input.value.trim().length === 0;
});

examples.addEventListener("click", (e) => {
  const btn = e.target.closest(".chip-example");
  if (!btn) return;
  input.value = btn.textContent;
  submitBtn.disabled = false;
  ask(btn.textContent);
});

submitBtn.disabled = true;
loadCorpus();
