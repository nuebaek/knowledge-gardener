# 🗓️ week-08-TIL — 직선(Chain)에서 그래프(Graph)로

> LCEL의 직선형 파이프라인이 분기·반복·병렬을 감당하지 못하는 지점에서 랭그래프를 만나고, State·Node·Edge라는 최소 3요소로 그래프를 조립한 뒤 Checkpointer로 "멈췄다 이어가는" 영속성까지 확보한 주차

**기간:** 2026-06-29 ~ 2026-07-03

**키워드:** `LangGraph` `Chain vs Workflow vs Agent` `State/Node/Edge` `Reducer` `ToolNode` `Command` `Send` `ReAct` `Checkpointer` `Thread` `Time Travel` `Durable Execution` `멱등성` `Human-in-the-loop` `MCP` `대용량 트래픽`

---

## 📋 이번 주 학습 지도

| 날짜 | 주제 | 핵심 개념 |
|------|------|-----------|
| 06/29 (월) | LangGraph 기초 | LangGraph, Chain/Workflow/Agent, Graph API 구축 6단계, State/Reducer, Node/Edge, Tool Calling/ToolNode, Routing/Command, Parallelization/Send, Loop/ReAct |
| 06/30 (화) | LangGraph Persistence | Functional vs Graph API, ReAct 미니프로젝트, Checkpoint/Checkpointer, Thread, Time Travel, Durable Execution/Determinism, 멱등성 |
| 07/01 (수) | LangGraph 고급 + 멘토링 | Human-in-the-loop, Architecture(Planner/Executor) |
| 07/02 (목) | Checkpointer vs Memory, MCP | Checkpointer/Memory 구분, MCP, 랭체인→랭그래프 마이그레이션 |
| 07/03 (금) | 카카오 특강 | 대용량 트래픽 처리, 비기능 요구사항, MSA |

---

## 📖 학습 내용

### Day 1 (06/29) — LangGraph 기초

#### 1-1. LangGraph

> 체인(직선)으로는 표현할 수 없는 분기·반복·병렬을 상태 기반 그래프(방향 그래프)로 표현하는 랭체인 생태계의 오케스트레이션 프레임워크.

**한줄정의**
> 노드(작업 단위)와 엣지(흐름 규칙)로 이루어진 그래프를 실행하며, 공유된 State를 여러 노드가 함께 읽고 쓰는 구조.

**왜 쓰는가**
- LCEL 파이프(`prompt | model | parser`)는 A→B→C로만 흐르는 직선이라, "조건에 따라 다른 노드로 가기"·"같은 노드를 몇 번이고 반복하기"·"여러 경로를 동시에 타기"를 표현할 수 없다
- 실제 에이전트는 도구를 쓰고, 실패하면 재시도하고, 상황에 따라 다른 경로를 타야 하므로 직선이 아니라 그래프가 필요하다

**비유:** 체인 = 컨베이어 벨트(한 방향), 그래프 = 지하철 노선도(분기·순환·환승 가능).

---

#### 1-2. Chain vs Workflow vs Agent

> LLM 애플리케이션의 자율성 정도에 따라 나뉘는 3단계 구조.

| 구조 | 흐름 결정 주체 | 특징 |
|---|---|---|
| **Chain** | 개발자가 코드로 100% 고정 | A→B→C 직선, 예측 가능하지만 유연성 없음 |
| **Workflow** | 개발자가 분기 조건을 미리 설계 | 조건부 분기·병렬은 있지만 "어떤 경로를 탈지"는 사전 정의된 로직이 결정 |
| **Agent** | LLM이 매 스텝 스스로 판단 | 다음에 뭘 할지(어떤 도구를 쓸지, 언제 멈출지)를 LLM이 실시간으로 결정 |

**핵심**
- 자율성이 높아질수록 유연하지만 예측 불가능성과 비용(토큰/지연)도 함께 커진다
- 실무에서는 무조건 Agent가 아니라, 필요한 만큼만 자율성을 부여하는 게 정답 — "쓸 수 있다"와 "써야 한다"는 다르다

---

#### 1-3. Graph API 구축 6단계

> 랭그래프 그래프를 만드는 표준 절차.

**핵심**
1. **State Schema 정의**: 그래프 전체가 공유할 상태(데이터 구조)를 TypedDict/Pydantic으로 선언
2. **StateGraph 인스턴스 생성**: `StateGraph(State)`
3. **Node 추가**: `add_node("이름", 함수)` — 함수는 State를 받아 변경분을 반환
4. **Edge 연결**: `add_edge`, `add_conditional_edges`로 노드 간 흐름 규칙 정의
5. **시작/종료 지정**: `START`, `END` 특수 노드로 진입점/종료점 명시
6. **Compile**: `.compile()`로 실행 가능한 그래프 객체 생성 (이때 Checkpointer도 함께 부착 가능)

---

#### 1-4. State, Reducer, MessagesState

> State = 그래프의 모든 노드가 공유하는 데이터 저장소. Reducer = 여러 노드의 State 변경을 어떻게 병합할지 정하는 규칙.

**핵심**
- 기본 동작은 "덮어쓰기(overwrite)" — 노드가 반환한 값이 이전 값을 대체
- **Reducer**를 지정하면 병합 방식을 바꿀 수 있음. 대표적으로 `add_messages`(또는 `operator.add`)는 덮어쓰지 않고 **리스트에 추가(append)** — 대화 메시지처럼 누적이 필요한 값에 필수
- **MessagesState**: 대화형 에이전트에서 자주 쓰는 State 스키마로, `messages` 필드에 `add_messages` Reducer가 기본 적용된 사전 정의 클래스

**비유:** Reducer 없는 State = 화이트보드를 지우고 새로 쓰기 / `add_messages` Reducer = 화이트보드 밑에 계속 이어 쓰기(대화 로그처럼).

---

#### 1-5. Node, Edge, START/END, Compile

> Node = 실제 작업을 수행하는 함수(또는 Runnable). Edge = 노드 간 이동 규칙.

**핵심**
- **Node**: State를 입력받아 State의 일부를 변경해 반환하는 파이썬 함수
- **Edge**: `add_edge(A, B)`는 무조건 A→B / `add_conditional_edges(A, 라우팅함수)`는 라우팅 함수의 반환값에 따라 다음 노드가 갈림
- **START/END**: 그래프의 진입점과 종료점을 나타내는 특수 상수 노드
- **Compile**: 정의만 해둔 그래프를 실제 실행 가능한 객체로 확정 짓는 단계 — 이때 Checkpointer, 인터럽트 지점 등 실행 옵션도 함께 설정

---

#### 1-6. Tool Calling과 ToolNode

> LLM이 "이 도구를 이렇게 호출해줘"라고 요청하면, 실제로 그 도구를 실행하고 결과를 다시 State에 넣어주는 전용 노드.

**핵심**
- LLM 자체는 도구를 실행할 수 없다 — 실행 요청(함수명 + 인자)을 구조화된 형태로 "제안"만 할 뿐
- **ToolNode**가 이 제안을 받아 실제 파이썬 함수를 실행하고, 결과를 `ToolMessage`로 만들어 State(`messages`)에 추가
- 이 사이클(모델 호출 → 도구 요청 판단 → ToolNode 실행 → 결과 반영 → 다시 모델 호출)이 반복되는 게 에이전트의 기본 루프

---

#### 1-7. Routing과 Command

> Routing = 조건부 엣지로 다음 노드를 결정하는 방식. Command = 노드 함수 안에서 State 업데이트와 다음 노드 이동을 동시에 지시하는 객체.

**핵심**
- **Routing(conditional edge)**: 별도의 라우팅 함수가 현재 State를 보고 문자열(다음 노드 이름)을 반환 → 그래프 구조 자체에 분기가 선언됨
- **Command**: 노드 함수가 `return Command(update={...}, goto="다음노드")` 형태로, State 변경과 이동을 한 번에 반환 — 별도 라우팅 함수 없이 노드 내부 로직만으로 분기 가능
- 단순한 분기는 Routing이 선언적이라 읽기 쉽고, 노드 로직과 분기 판단이 밀접하게 얽혀 있으면 Command가 더 간결

---

#### 1-8. Parallelization과 Send

> 여러 노드를 동시에 실행하고 결과를 모으는 병렬 처리 메커니즘.

**핵심**
- 하나의 노드에서 여러 엣지로 동시에 뻗어나가면 병렬 실행됨(팬아웃) → 각 브랜치가 끝나면 자동으로 팬인(집계)
- **Send**: 입력 리스트의 개수만큼 동일 노드를 동적으로 여러 번 병렬 실행해야 할 때 사용 (map-reduce 패턴) — 몇 개를 병렬 실행할지 컴파일 시점이 아니라 런타임에 결정

**비유:** Parallelization(고정 브랜치) = 미리 정해진 몇 개 창구를 동시에 여는 것 / Send = 대기줄 인원 수만큼 창구를 그때그때 동적으로 늘리는 것.

---

#### 1-9. Loop와 ReAct Pattern

> Loop = 조건이 만족될 때까지 같은 노드(들)를 반복 실행하는 그래프 구조. ReAct = Reasoning(추론) + Acting(행동)을 번갈아 반복하는 에이전트 설계 패턴.

**핵심**
- 그래프에서 조건부 엣지가 자기 자신(또는 이전 노드)으로 되돌아가도록 연결하면 Loop가 만들어짐
- **ReAct**: "생각한다 → 도구를 쓴다 → 관찰한다 → 다시 생각한다"를 종료 조건(더 이상 도구가 필요 없음)까지 반복 — LangGraph의 모델 노드 ↔ ToolNode 순환이 바로 ReAct의 전형적 구현

---

#### 1-10. Functional API (예고)

> `StateGraph` 없이 `@entrypoint`, `@task` 데코레이터로 일반 파이썬 함수처럼 랭그래프 워크플로우를 작성하는 대안 인터페이스. (다음 날 상세 비교 예정)

---

### Day 2 (06/30) — LangGraph Persistence

#### 2-1. Functional API vs Graph API

> 동일한 랭그래프 실행 엔진 위에서, 그래프를 명시적으로 그리는 대신(Graph API) 일반 함수 호출처럼 작성하는(Functional API) 두 가지 진입 방식.

| | Graph API | Functional API |
|---|---|---|
| 작성 방식 | `StateGraph` + `add_node`/`add_edge` 명시적 선언 | `@entrypoint`, `@task` 데코레이터로 일반 함수처럼 작성 |
| 흐름 가시성 | 그래프 구조가 눈에 보임(시각화 용이) | 파이썬 제어문(if/for)으로 자연스럽게 분기·반복 |
| 적합한 경우 | 구조가 복잡하고 분기가 많아 시각화·유지보수가 중요할 때 | 로직이 단순 순차적이거나 파이썬스럽게 빠르게 짜고 싶을 때 |
| 공통점 | 둘 다 동일하게 Checkpointer, State 영속성 지원 | 좌동 |

**핵심**
- 두 API는 경쟁 관계가 아니라 같은 엔진의 다른 인터페이스 — 팀 컨벤션과 그래프 복잡도에 따라 선택

---

#### 2-2. ReAct 미니프로젝트 (팩트체커)

> 사용자의 주장에 대해 모델이 "검색이 필요한가?"를 스스로 판단(Reasoning) → 필요하면 검색 도구 호출(Acting) → 검색 결과를 관찰(Observation) → 판정까지 반복하는 실습.

**핵심**
- ReAct 루프(모델↔ToolNode)를 실제로 손으로 짜보며, 종료 조건(도구 호출 없이 최종 답변만 나올 때) 설계가 생각보다 까다롭다는 걸 체감
- 검색 결과가 애매하면 모델이 재검색을 반복하다가 무한루프에 빠질 수 있어, 최대 반복 횟수 같은 안전장치가 실무에는 필요

---

#### 2-3. Persistence, Checkpoint, Checkpointer

> Persistence = 그래프 실행 상태를 영구 저장해 중단·재개·복원이 가능하게 하는 랭그래프의 핵심 기능. Checkpoint = 특정 시점의 State 스냅샷. Checkpointer = 그 스냅샷을 저장/조회하는 백엔드(메모리, SQLite, Postgres 등).

**왜 쓰는가**
- 인간 개입(Human-in-the-loop)으로 중간에 멈춰 승인받거나, 서버가 재시작돼도 대화가 끊기지 않게 하려면 State가 어딘가에 저장되어 있어야 한다

**핵심**
- 그래프가 각 슈퍼스텝(노드 실행 묶음)마다 자동으로 Checkpoint를 남김
- Checkpointer를 `.compile(checkpointer=...)`로 부착하면 이 저장이 활성화됨 (미부착 시 실행 상태는 휘발)

---

#### 2-4. Thread (thread_id)

> 하나의 독립된 대화/실행 세션을 식별하는 키.

**핵심**
- 같은 그래프라도 `thread_id`가 다르면 완전히 별개의 State 히스토리로 취급됨 — 멀티 유저 서비스에서 유저(or 세션)별 격리의 기본 단위
- `config={"configurable": {"thread_id": "..."}}` 형태로 매 호출마다 지정

**비유:** Day1의 대화 이력 세션 관리(멀티 유저 문맥 분리)와 같은 문제의식이 랭그래프 레벨에서는 Thread로 구현된다.

---

#### 2-5. Time Travel

> 특정 과거 Checkpoint 시점으로 되돌아가 그 지점부터 다시 실행(재생 또는 분기 실행)하는 기능.

**핵심**
- 모든 슈퍼스텝이 Checkpoint로 남기 때문에, 특정 시점의 `checkpoint_id`를 지정해 그 상태에서부터 다시 시작 가능
- 디버깅("이 시점에서 뭐가 잘못됐지?")이나 "다른 선택을 했다면?" 실험(what-if 분기)에 활용

---

#### 2-6. Durable Execution, Determinism, 멱등성

> Durable Execution = 중단되어도 마지막 저장 지점부터 안전하게 재개할 수 있는 실행 방식. Determinism(결정성) = 같은 입력이면 항상 같은 결과. 멱등성(Idempotency) = 같은 연산을 여러 번 실행해도 결과가 한 번 실행한 것과 같음.

**왜 쓰는가**
- Checkpoint에서 재개할 때, 이미 실행됐던 노드가 재실행되며 부작용(예: 중복 결제, 중복 이메일 발송)을 일으키면 안 된다 → 노드 로직이 멱등적이어야 안전하게 재개 가능

**핵심**
- Durable Execution이 실무에서 신뢰받으려면 각 노드가 **결정적(같은 입력→같은 출력)이거나 멱등적(재실행해도 안전)**이어야 한다는 전제가 깔려 있다
- 랭그래프가 재개를 지원한다고 해서 노드 내부 로직의 부작용까지 자동으로 안전해지는 건 아니다 — 이건 개발자가 설계로 책임져야 하는 부분

**비유:** 결제 API를 멱등키(idempotency key) 없이 재시도하면 중복 결제가 나는 것과 동일한 문제의식.

---

### Day 3 (07/01) — LangGraph 고급 + 멘토링

> 이 날짜는 정식 배움일지 없이 짧은 노트(`26-07-01.md`)만 남아 있다. 노트 대부분이 헤딩만 있고 내용이 비어 있어(멱등성, 부수효과, 메모리 항목), 실제로 채워져 있던 두 개념(Human-in-the-loop, Architecture)과 멘토링 메모만 정리한다.

#### 3-1. Human-in-the-loop — 내적 멈춤 / 외적 멈춤

> 그래프 실행 도중 사람의 판단을 받기 위해 일시적으로 멈추는 패턴. LangGraph의 Checkpointer 위에서 구현된다.

**핵심**
- **외적 멈춤**: 내(개발자/시스템)가 내 바깥의 무언가(그래프 실행)를 멈추게 하는 것
- **내적 멈춤**: 그래프 자신이 스스로 멈춰서 생각을 하는 것(예: 도구 호출 전 승인 대기)
- 둘 다 그래프의 **종료(End)가 아니다** — 멈췄다가 다시 이어갈 수 있다는 점에서 Persistence(Day2)와 같은 문제의식

#### 3-2. Architecture — Planner / Executor

> 에이전트 아키텍처를 설계할 때 "왜 이렇게 짰는지 설명이 가능해야 한다"는 원칙 아래, 계획을 세우는 역할과 그 계획을 실행하는 역할을 분리하는 설계 패턴.

**핵심**
- **Planner**: 목표를 여러 단계로 쪼개는 역할
- **Executor**: Planner가 세운 단계를 실제로 수행하는 역할
- 역할을 분리하면 "지금 어느 단계에서 왜 이 행동을 했는지"를 설명할 수 있어, 에이전트가 블랙박스가 되는 것을 막는다

**멘토링 메모**
- 로컬 환경(16GB 메모리 맥북)에서 Gemma 수준 모델을 mlx로 돌리는 실험 방향 논의
- 범용 RAG를 화이트라벨/OEM 형태로 만드는 전략 — 정의·사용법·이유를 서브노트 문서로 먹여 RAG를 구성하고, 여기에 웹서치를 더해 최신성을 보완하는 방향

---

### Day 4 (07/02) — Checkpointer vs Memory, MCP

> 이 날짜도 배움일지 없이 노트(`26-07-02.md`)만 남아 있지만, Checkpointer/Memory 개념 정리와 MCP 정의, 실습(alex-rag) 메모가 있어 정리한다.

#### 4-1. Checkpointer vs Memory

> Checkpointer는 "저장하는 행위/저장 레이어"에 초점을 둔 개념이고, Memory는 "저장된 정보/기억의 범위"에 초점을 둔 개념 — 같은 저장 메커니즘을 서로 다른 층위에서 부르는 이름이다.

**핵심**
- Checkpointer = 영속성(persistence)을 위한 인터페이스, **저장 행위** 자체
- Memory = 저장된 **정보 자체**(단기 메모리/장기 메모리)
- 단기 메모리: 한 세션 안의 대화처럼, 스레드가 끝나면 증발(삭제)되는 정보
- 장기 메모리: 한 유저에 대한 정보처럼, 세션이 끝나도 유지되어야 하는 정보
- OOP의 부모-자식 클래스 상속(`B(A)`)에 빗대 "체크포인터가 메모리의 상위 개념인가?"를 스스로 질문해봤지만, 정확히는 상속 관계가 아니라 **체크포인터 = 저장 레이어(시스템 계층), 메모리 = 기억 범위(저장물)**로 관점 자체가 다른 개념

#### 4-2. MCP (Model Context Protocol)

> AI 모델이 외부 도구나 데이터 소스와 상호작용할 수 있도록 표준화된 방법을 제공하는 통신 규약.

**핵심**
- `mcp_server` / `mcp_client` 구조로 나뉨 — 실습 중 포트 충돌 이슈를 겪음
- 실습(alex-rag) 메모: 기존 LangChain 기반 코드에서 LangChain 관련 부분을 걷어내고 LangGraph로 재구성 / 서버를 FastAPI + MCP 두 개로 분리 / 서버를 띄울 때마다 매번 임베딩하던 방식을 파일 DB에 저장하는 방식으로 변경 / FastAPI Cloud로 배포하면서 아무나 접근하지 못하게 API 비용·접근 제어 고려
- 과제 방향: 랭체인 → 랭그래프 → 개인 프로젝트(RAG 챗봇)로 이어지는 마이그레이션. "코딩을 외주 주는 것도 안 되지만, 생각을 외주 주는 건 더 안 된다"는 코멘트가 인상적

---

### Day 5 (07/03) — 카카오 현직자 특강: 대용량 트래픽 처리

> 카카오 선물하기 서버 리더의 특강. 개념 학습보다는 실무 사례 중심이라 노트 요약 형태로 정리한다.

**1장 — 대용량 트래픽 처리 이론**
- 기능 요구사항(기획자가 설계하는 기능)과 비기능 요구사항(성능·확장성·신뢰성·가용성)을 구분
- 성능은 캐시·인덱스·비동기로, 확장성은 수평 확장(샤딩·오토스케일)으로, 가용성은 다중화·헬스체크·로드밸런서로, 신뢰성은 서킷 브레이커·벌크헤드·타임아웃으로 확보
- SLI/SLO/SLA, P50/P90/P99 같은 지표로 "얼마나 느린가"를 정량화한다 — 예를 들어 P99가 10초라는 건, 가장 느린 1%의 요청이 10초 이상 걸린다는 뜻
- 엔지니어링은 결국 한정된 시간·리소스 안에서 트레이드오프를 관리하는 "비용의 문제"

**2장 — 선물하기 사례**
- 모노레포에서 MSA로 전환. 이벤트 트래픽을 지속형(예측 가능 → 서버 증설로 대응)과 순간 폭발형(예: 인기 상품 오픈 → 대기열로 대응)으로 구분해 설계
- 캐시는 부하를 막는 장벽이지만, 캐시 스탬피드·캐시 페네트레이션(존재하지 않는 값 조회)·핫키 같은 무효화 이슈가 발생할 수 있음
- 상품 원장(플랫폼) → 상품 변경 이벤트 → 뷰로 처리하는 이벤트 기반 구조. 주문서처럼 중요한 데이터는 상품 플랫폼에서 직접 처리
- 대기열·문지기(비정상 트래픽 IP 차단) 등으로 순간 폭발형 트래픽을 제어해 연쇄 장애를 방지
- Best Practice보다 **지금 상황의 제약에 맞는 해법을 찾는 것**이 더 중요하다는 게 이 세션의 결론

**QnA에서 기억할 것**
- 신입에게 대용량 트래픽 처리 "경험" 자체를 요구하지는 않는다 — 관련 컨퍼런스(if kakao 등) 학습이나 개인 부하테스트 경험으로 대체 가능
- 포트폴리오는 다양성보다 한 분야를 얼마나 깊이 팠는지가 더 중요
- 경험하지 않은 개념도 학습으로 채울 수 있어야 한다는 전제에서, CS 기본기는 여전히 중요
- 카카오 AI 서비스의 방향성은 에이전트 개발 — RAG 파이프라인·KV 캐시·지식 그래프·벡터DB 구성은 이제 막 시작하는 단계

---

## 🔗 이번 주 개념 흐름

```
[월 06/29] LangGraph 기초
    LCEL 직선의 한계 → Chain/Workflow/Agent → Graph API 6단계
    → State/Reducer/MessagesState → Node/Edge/START/END/Compile
    → Tool Calling/ToolNode → Routing/Command → Parallelization/Send → Loop/ReAct
    ↓ (그래프를 짜는 법은 익혔다. 근데 죽었다 살아나면?)
[화 06/30] LangGraph Persistence
    Functional vs Graph API → ReAct 미니프로젝트(팩트체커)
    → Checkpoint/Checkpointer → Thread(thread_id) → Time Travel
    → Durable Execution/Determinism/멱등성
    ↓ (그래프를 멈췄다 이어가는 법을 익혔다. 사람이 개입하려면?)
[수 07/01] LangGraph 고급
    Human-in-the-loop(내적/외적 멈춤) → Architecture(Planner/Executor)
    ↓ (저장은 하는데, 뭘 저장하는지 구분이 안 된다)
[목 07/02] Checkpointer vs Memory, MCP
    Checkpointer(저장 행위) vs Memory(저장물) → MCP(외부 도구 연동 표준 규약)
    ↓ (지금까지 배운 게 실제 서비스 규모에서는?)
[금 07/03] 카카오 특강
    대용량 트래픽 처리 — 비기능 요구사항, MSA, 캐시/대기열 전략
```

월요일은 "직선(Chain)으로는 안 되는 것"에서 출발해 State·Node·Edge라는 그래프의 최소 문법을 세웠고, 화요일은 그렇게 만든 그래프가 "끊겨도 이어질 수 있는가"라는 실무적 질문으로 이어졌다. Checkpointer·Thread·Time Travel은 결국 Day1의 State를 "시간 축으로 스냅샷 찍어 저장하는 것"이라는 점에서 한 흐름이고, 마지막의 결정성/멱등성 논의는 "영속성을 지원한다고 안전이 자동으로 따라오지 않는다"는, 프레임워크가 해결해주지 않는 개발자의 책임 영역을 짚은 게 인상적이었다. 수·목요일은 그 연장선에서 "사람이 언제 개입하는가(HITL)"와 "무엇을 얼마나 저장하는가(Checkpointer vs Memory)"를 더 세밀하게 구분했고, 금요일 특강은 지금까지 배운 개념들이 실제 대용량 서비스에서는 훨씬 더 큰 비용·트레이드오프 문제로 이어진다는 걸 보여준 자리였다.

---