# Knowledge Gardener

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C)

> 개인의 학습 기록을 지속적으로 축적하고, 그 기록을 다시 검색 가능한 지식으로 활용하는 AI 학습 에이전트

Knowledge Gardener는 질문에는 근거 있는 답을 주고, 배운 내용은 잊지 않도록 기록해두는 개인 학습 에이전트다. 질문·일일 학습·주간 회고·프로젝트 TIL을 에이전트가 스스로 구분해 처리하고, 새로 쌓인 문서를 다시 지식베이스에 편입시켜 쓸수록 답변이 좋아지는 구조를 목표로 한다.

---

## Why

대부분의 개인용 RAG는 문서를 검색해 답을 찾는 것에서 끝난다. 하지만 실제 학습은 검색보다 배운 내용을 다시 꺼내 설명하고, 기존 지식과 연결하는 과정에서 이루어진다.

Dunlosky 외(2013)의 학습 기법 메타분석(*Improving Students' Learning With Effective Learning Techniques*)에서도 **인출 연습(practice testing)** 과 **분산 복습(distributed practice)** 은 학습 효과가 높은 전략으로 평가된 반면, 요약이나 재독처럼 많이 사용하는 방법은 효과가 상대적으로 낮았다.

Knowledge Gardener는 이러한 학습 원칙을 서비스 설계에 반영했다.

- 질문을 하면 근거 문서를 검색해 답변한다.
- 하루를 마칠 때는 `write_daily`가 오늘 배운 내용을 자신의 말로 정리하게 한다.
- 단순 요약이 아니라 '이 개념이 무엇과 연결되는가?'를 함께 기록해 기존 지식과 연결하도록 유도한다.
- 주간 단위에서는 `write_weekly`가 일일 기록을 다시 묶어 회고하도록 하여 자연스럽게 분산 복습이 이루어진다.

이렇게 만들어진 학습 기록은 다시 검색 가능한 지식베이스에 편입된다. 즉, 이 프로젝트는 **질문에 답하는 RAG**와 **학습 기록을 축적하는 시스템**을 하나의 에이전트 안에서 연결해, 사용할수록 개인의 지식이 함께 성장하는 구조를 목표로 한다.

---

## 주요 기능

**근거 기반 질의응답** (`answer_question`)  
검색된 문서에 근거해서만 답변하고 출처를 함께 반환한다. 관련도가 낮으면 질문을 다시 써서 재검색한다(최대 2회).

**학습 기록 자동 저장** (`write_daily` / `write_weekly` / `write_til`)  
회고형 발화에서 언급된 내용만 구조화해 일간·주간·프로젝트 단위로 저장한다. 언급 안 된 내용은 지어내지 않고 되묻는다.

**질문 vs 회고, 에이전트가 스스로 판단**  
LLM이 tool description을 보고 스스로 호출을 판단한다(tool-calling). `thread_id`로 멀티턴 맥락이 유지된다.

---

## 데모

```
POST /converse {"message": "Transformer의 Encoder와 Decoder 차이가 뭐야?", "thread_id": "t1"}
   ↓
Agent가 answer_question 호출 → retrieve → grade_docs
   ├─ score ≥ 0.4 → generate: 문서 근거로 답변 생성
   └─ score < 0.4 → rewrite_query → retrieve 재시도 (최대 2회)
   ↓
{"answer": "...", "tools_used": [], "saved_documents": []}

POST /converse {"message": "오늘 LangGraph 공부했고 tool-calling 배웠어. 정리해줘", "thread_id": "t1"}
   ↓
Agent가 write_daily 호출 (인자 검증 실패 시 에러 메시지 보고 자동 재시도)
   ↓
data/writer/dailynote/*.md 저장
   ↓
{"answer": "...", "tools_used": ["write_daily"], "saved_documents": [{"type": "write_daily", "file_name": "..."}]}
```

---

## 아키텍처

![Architecture](docs/knowledge-gardener-architecture.svg)

서버 시작 시 에이전트 그래프를 한 번만 구성해 재사용한다.   
사용자 메시지가 오면 LLM이 시스템 프롬프트와 대화 맥락을 보고 `answer_question` (검색) 또는 `write_daily` / `write_weekly` / `write_til` (저장) 중 하나를 부르거나, 되묻는다.  
모든 실행은 LangSmith로 추적, 평가된다.

---

## 기술 스택

| 영역 | 선택 | 이유 |
|---|---|---|
| 에이전트 오케스트레이션 | LangGraph StateGraph + tool-calling (`bind_tools`, `InMemorySaver`) | 조건 분기·재시도·대화 메모리는 단일 체인(LCEL)으로 표현 불가 |
| 문서 전처리 | [markitdown](https://github.com/microsoft/markitdown) + Markdown Header Splitter | 형식(PDF/HTML/DOCX) 통일 + 헤더 단위 분할로 검색 시 의미 보존 |
| 검색 재시도 | score 기반 `grade_docs` + `rewrite_query` (최대 2회) | top1 relevance score < 0.4면 재검색, 상한으로 무한루프 방지 |
| 임베딩 · 벡터 DB | 로컬 `BAAI/bge-m3` + Chroma `PersistentClient` | API/Cloud 무료 한도 초과 → 로컬 전환으로 무제한·무료 |
| 생성 LLM | `gemini-2.5-flash` · `gemma4:e2b`(Ollama) · `gemma-4-31b`(Cerebras, 기본) | provider만 바꿔 같은 코드로 품질·속도·비용 A/B 비교 |
| 서빙 · 평가 | FastAPI(`lifespan`으로 그래프 1회 구성) + LangSmith | 요청마다 `invoke()`만 호출, 전 실행 추적·평가 |

---

## 프로젝트 구조

```
kaia-project/
├── main.py                    # FastAPI 진입점 (lifespan에서 에이전트 그래프 1회 구성)
├── graph.py                   # QA 그래프(재검색 루프) / 에이전트 그래프(tool-calling+메모리)
├── nodes.py                   # retrieve / generate / grade_docs / rewrite_query 노드
├── tools.py                   # answer_question / write_daily / write_weekly / write_til
├── writer.py                  # 노트 저장 로직 (프론트매터 + LLM 본문)
├── rag.py                     # 임베딩·벡터스토어·LLM 빌더
├── state.py                   # GraphState 정의
├── routers/converse.py        # POST /converse, GET /threads/{thread_id}
├── routers/corpus.py          # 코퍼스 조회
├── controllers/rag.py         # 그래프 호출 + 응답 변환 + 에러 처리
├── schemas.py                 # 요청/응답 모델
├── preprocess.py              # data/raw → markitdown → data/processed
├── baseline.py                # LangSmith 평가 스크립트
├── prompts/                   # 콘텐츠 프롬프트
├── static/                    # 프론트엔드 (대화형 UI, /converse 기반)
├── data/, chroma_data/        # ← gitignore
└── docs/
```

---

## 실행 방법

```bash
uv sync
cp .env.example .env          # LLM_PROVIDER=google|ollama|cerebras, 실제 키 커밋 금지

uv run python preprocess.py   # data/raw/* → data/processed/*.md
uv run python rag.py          # 인덱싱 (chroma_data 생성, 최초 1회)
uv run fastapi dev main.py    # 서버 (http://localhost:8000/docs)
```

`/docs`에서 API 스펙과 예시 요청을 바로 확인할 수 있다.

---

## 로드맵

| 구성 | 계획 | 시점 |
|---|---|---|
| Graph RAG | 개체·관계를 그래프로 저장해 벡터 검색과 함께 다중 홉 질문에 답변 | 미정 |
| 학습 기록 재인덱싱 | 저장된 학습일지를 기존 인덱싱 파이프라인에 그대로 편입 — 쓸수록 코퍼스가 늘고 답변 근거도 풍부해지는 구조 | 미정 |
| Hub 고도화 | 재검색을 LLM 채점으로 교체, RAGAS 평가 도입, BM25 하이브리드 검색 | 미정 |
| Input Layer | 판서 인식(OCR), 북마크 흡수 | 미정 |
| Output Layer | Graph RAG 기반 마인드맵 시각화 | 미정 |

---

## 회고

- [7주차 회고](docs/retro-week7.md) — FastAPI 서빙 + LangSmith 평가 도입, 평가 결과 수치(모델 비교, latency 실측)
