"""에이전트 그래프의 '전이'를 검사하는 통합 테스트.

단위 테스트(test_study_session.py)가 기록 로직을, 여기서는 그 기록이 그래프·서비스 경로를
타고 나갈 때(라우팅, 응답 조립)를 검사한다. LLM은 전부 가짜라 결정적이고, 임베딩/벡터스토어도
안 탄다(qa_graph를 주입하므로).
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.rag.graph import build_agent_graph
from app.rag.nodes import FINALIZE_CONFIRM_MESSAGE
from app.rag.tools import make_tools
from app.schemas.rag import TopicList, TurnResult
from app.services.rag_service import converse


# ---------------- 가짜 LLM ----------------

class _Structured:
    """with_structured_output(TopicList)이 돌려주는 러너. 큐가 비면 마지막 값을 반복한다."""

    def __init__(self, queue):
        self.queue = list(queue)

    def invoke(self, messages):
        return self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]


class _TurnResults:
    """with_structured_output(TurnResult)이 돌려주는 러너 — study_turn(판정+다음질문 1회 호출)용.

    verdict는 순서대로 소비하고(마지막은 반복), next_question은 프롬프트에 담긴 next_topic을
    그대로 반영해 만든다 — "다음 질문이 실제로 다음 토픽에 대해 나오는지"까지 검증하기 위해서다.
    """

    def __init__(self, verdicts, recall_topics):
        self.verdicts = list(verdicts)
        self.recall_topics = recall_topics  # FakeLLM과 공유 — 질문이 나간 토픽을 순서대로 기록

    def invoke(self, messages):
        next_topic = messages[-1].content.split("\n", 1)[0].removeprefix("next_topic: ").strip()
        next_question = None
        if next_topic != "none":
            self.recall_topics.append(next_topic)
            next_question = f"[인출] {next_topic}?"
        verdict = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
        return TurnResult(verdict=verdict, next_question=next_question)


class FakeLLM:
    """agent 노드(tool-calling) / 세션 첫 질문(invoke) / 구조화 출력(TopicList, TurnResult)
    세 갈래를 한 객체로 흉내낸다. 갈래 구분은 실제로 들어온 프롬프트 모양으로 하므로,
    배선이 어긋나면 여기서 드러난다."""

    def __init__(self, *, topics=(), umbrella="", verdicts=("explained",), tool_plan=None, confirm_topics=None):
        self.topics = list(topics)
        self.umbrella = umbrella
        self.verdicts = list(verdicts)
        self.tool_plan = tool_plan or {}
        self.recall_topics = []
        self.agent_calls = 0
        # confirm_finalize_node("정리할까요?" 답 처리)도 write_daily와 같은
        # with_structured_output(TopicList)을 타므로, 인스턴스당 러너 하나를 공유해 큐로
        # 순서를 준다 — 세션 시작 때 self.topics, 그 다음(confirm 단계) confirm_topics.
        self._topic_list_runner = None
        self.confirm_topics = confirm_topics

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        if schema is TopicList:
            if self._topic_list_runner is None:
                queue = [TopicList(topics=self.topics, umbrella=self.umbrella)]
                if self.confirm_topics is not None:
                    queue.append(TopicList(topics=self.confirm_topics))
                self._topic_list_runner = _Structured(queue)
            return self._topic_list_runner
        if schema is TurnResult:
            return _TurnResults(self.verdicts, self.recall_topics)
        raise AssertionError(f"예상하지 못한 스키마: {schema}")

    def invoke(self, messages):
        last_content = messages[-1].content if messages else ""

        # write_daily가 세션을 열 때 첫 질문(generate_recall_question)만 이 경로를 탄다.
        # 턴2 이후 질문은 with_structured_output(TurnResult)에서 나온다.
        if isinstance(last_content, str) and "Ask about this topic:" in last_content:
            topic = last_content.rsplit("Ask about this topic:", 1)[-1].strip()
            self.recall_topics.append(topic)
            return AIMessage(f"[인출] {topic}?")

        # agent 노드 경로
        self.agent_calls += 1
        history = messages[1:]
        last_human_at = max(i for i, m in enumerate(history) if isinstance(m, HumanMessage))
        already_ran_tool = any(isinstance(m, ToolMessage) for m in history[last_human_at:])
        if already_ran_tool:
            return AIMessage("정리해서 알려줄게요")  # tool 결과를 받은 뒤의 마무리 발화

        text = history[last_human_at].content
        for trigger, (name, args) in self.tool_plan.items():
            if trigger in text:
                return AIMessage("", tool_calls=[{"name": name, "args": dict(args), "id": f"call-{name}"}])
        return AIMessage("무슨 얘기인지 조금 더 알려줄래요?")


class _FakeQAGraph:
    def invoke(self, payload):
        return {"answer": "합성곱은 ...", "sources": ["data/processed/convolutional-networks.md"]}


class _FakeWriterLLM:
    """writer가 노트 본문을 만들 때 쓰는 LLM. 입력을 그대로 되돌려줘서, 사용자가 실제로 한
    말이 노트까지 전달됐는지를 테스트가 확인할 수 있게 한다."""

    def invoke(self, messages):
        return AIMessage("\n".join(str(m.content) for m in messages))


@pytest.fixture
def graph_env(isolated_writer, monkeypatch):
    from app.writer import writer

    monkeypatch.setattr(writer, "default_llm", lambda: _FakeWriterLLM())

    # 가짜 코퍼스 검색 — finalize가 🌱마다 이걸 불러 근거 경로를 붙인다. 실제 벡터스토어를
    # 안 태우면서 "토픽이 검색에 그대로 넘어갔는지"까지 확인할 수 있다.
    def _fake_search(topic):
        return [f"data/processed/{topic}.md"]

    def _build(**kwargs):
        fake = FakeLLM(**kwargs)
        tools = make_tools(llm=fake, qa_graph=_FakeQAGraph())
        return fake, build_agent_graph(llm=fake, tools=tools, search_sources=_fake_search, judge_llm=fake)

    return _build


# ---------------- 인출 세션 전이 ----------------

def test_session_start_asks_for_topic_confirmation_first(graph_env):
    """write_daily는 곧장 질문하지 않고 추출한 주제를 먼저 확인받는다 — 프런트가 체크박스
    카드를 그릴 수 있도록 topics를 구조화된 필드로 돌려줘야 한다."""
    fake, graph = graph_env(topics=["레이어 캐시", "bind-mount"],
                            tool_plan={"오늘": ("write_daily", {})})

    res = converse(graph, "오늘 도커 공부했어", "t1")

    assert res.awaiting_topic_confirm is True
    assert res.topics == ["레이어 캐시", "bind-mount"]
    assert res.tools_used == ["write_daily"]
    assert res.saved_documents == []


def test_session_start_delivers_recall_question_verbatim(graph_env):
    """확인 턴에서 만든 인출 질문이 agent를 거치지 않고 그대로 나가야 한다 —
    agent를 거치면 "힌트 금지" 제약을 모르는 모델이 질문을 다시 써버릴 수 있다."""
    fake, graph = graph_env(topics=["레이어 캐시", "bind-mount"],
                            tool_plan={"오늘": ("write_daily", {})})

    converse(graph, "오늘 도커 공부했어", "t1")
    res = converse(graph, "이 주제로 시작", "t1")

    assert res.answer == "[인출] 레이어 캐시?"      # agent가 다시 쓴 문장이 아니다
    assert fake.agent_calls == 1                    # tool 호출 판단 1회로 끝 — 재진입 없음
    assert res.awaiting_topic_confirm is False
    assert res.saved_documents == []                # 아직 저장된 건 없다


def test_session_start_respects_deselected_topics(graph_env):
    """체크 해제된 주제는 실제로 세션에서 빠져야 한다 — 카드가 장식이 아니라 진짜 필터."""
    fake, graph = graph_env(topics=["레이어 캐시", "bind-mount", "네트워크"],
                            tool_plan={"오늘": ("write_daily", {})})

    converse(graph, "오늘 도커 공부했어", "t1b")
    res = converse(graph, "이 주제로 시작", "t1b", selected_topics=["bind-mount", "네트워크"])

    assert res.answer == "[인출] bind-mount?"        # 제외된 "레이어 캐시"는 묻지 않는다
    state = graph.get_state({"configurable": {"thread_id": "t1b"}}).values
    assert state["pending"] == ["bind-mount", "네트워크"]


def test_session_completes_and_reports_exactly_one_saved_document(graph_env):
    """study 노드가 저장한 데일리노트가 응답(saved_documents)에 실려야 한다.
    마지막 토픽 뒤엔 곧장 저장이 아니라 "정리할까요?" 확인이 먼저 오고, "없다"고 답해야
    저장된다(confirm_topics=[] → 추가 토픽 없음 → finalize)."""
    fake, graph = graph_env(topics=["레이어 캐시", "bind-mount"],
                            umbrella="Docker 배포",
                            verdicts=["explained", "explained"],
                            tool_plan={"오늘": ("write_daily", {})},
                            confirm_topics=[])

    converse(graph, "오늘 도커 공부했어", "t2")
    converse(graph, "이 주제로 시작", "t2")
    mid = converse(graph, "변경 없는 층을 재사용해서 빌드가 빨라져", "t2")
    assert mid.answer == "[인출] bind-mount?"       # 두 번째 토픽으로 넘어간다
    assert mid.saved_documents == []

    confirm = converse(graph, "호스트 디렉터리를 컨테이너에 그대로 붙이는 것", "t2")
    assert confirm.saved_documents == []            # 아직 안 저장됨 — 확인부터 물어봄
    assert confirm.answer == FINALIZE_CONFIRM_MESSAGE

    last = converse(graph, "아니 없어", "t2")

    assert [d.type for d in last.saved_documents] == ["dailynote"]
    assert last.saved_documents[0].file_name.endswith(".md")

    state = graph.get_state({"configurable": {"thread_id": "t2"}}).values
    assert state["pending"] == []
    assert [a["topic"] for a in state["answered"]] == ["레이어 캐시", "bind-mount"]
    assert state["seedlings"] == []
    # umbrella가 write_daily(세션 시작)에서 finalize까지 안 끊기고 그대로 실려왔는지 —
    # 끊기면 finalize_node가 ", ".join(topics)로 조용히 폴백해서 여기서만 드러난다.
    assert state["umbrella"] == "Docker 배포"


def test_finalize_confirmation_with_more_topics_resumes_session(graph_env):
    """"정리할까요?" 확인에 더 공부한 게 있다고 답하면 곧장 저장하지 않고, 그 답에서 새
    토픽을 뽑아 인출연습을 이어간다(write_daily와 같은 토픽 추출 방식 재사용)."""
    fake, graph = graph_env(topics=["레이어 캐시"],
                            verdicts=["explained"],
                            tool_plan={"오늘": ("write_daily", {})},
                            confirm_topics=["쿠버네티스"])

    converse(graph, "오늘 도커 공부했어", "t5")
    converse(graph, "이 주제로 시작", "t5")
    confirm = converse(graph, "변경 없는 층 재사용", "t5")
    assert confirm.answer == FINALIZE_CONFIRM_MESSAGE
    assert confirm.saved_documents == []

    resumed = converse(graph, "쿠버네티스도 좀 봤어", "t5")

    assert resumed.answer == "[인출] 쿠버네티스?"    # 새 토픽으로 인출 질문이 이어짐
    assert resumed.saved_documents == []             # 아직 저장 안 됨

    state = graph.get_state({"configurable": {"thread_id": "t5"}}).values
    assert state["pending"] == ["쿠버네티스"]
    assert state["awaiting_finalize"] is False


def test_all_skip_session_keeps_every_topic_as_seedling(graph_env, isolated_writer):
    """설명 못 한 토픽이 조용히 사라지지 않고 🌱로 노트에 남는지 — 그래프 경로까지 포함해서."""
    fake, graph = graph_env(topics=["레이어 캐시", "bind-mount"],
                            verdicts=["skip", "skip"],
                            tool_plan={"오늘": ("write_daily", {})},
                            confirm_topics=[])

    converse(graph, "오늘 도커 공부했어", "t3")
    converse(graph, "이 주제로 시작", "t3")
    skip1 = converse(graph, "모르겠어", "t3")
    skip2 = converse(graph, "이것도 패스", "t3")  # 마지막 토픽 skip → "정리할까요?" 확인 질문
    last = converse(graph, "아니 없어", "t3")  # 확인에 "없다"고 답해야 실제로 저장됨

    # 문서로 저장되기 전에도, 패스한 그 턴의 응답에 바로 "다시 꺼내볼 것"이 실려야
    # 채팅 화면에서 finalize를 기다리지 않고 인라인으로 보여줄 수 있다.
    assert skip1.recall == ["레이어 캐시"]
    assert skip2.recall == ["bind-mount"]
    assert last.recall == []  # 확인 질문 답변 턴엔 새로 seedling된 게 없음

    state = graph.get_state({"configurable": {"thread_id": "t3"}}).values
    assert [s["topic"] for s in state["seedlings"]] == ["레이어 캐시", "bind-mount"]
    assert state["answered"] == []

    saved = next((isolated_writer / "dailynote").glob("*.md")).read_text(encoding="utf-8")
    assert "다시 꺼내볼 것" in saved
    assert "bind-mount" in saved
    assert "data/processed/bind-mount.md" in saved  # 🌱마다 근거 문서 경로가 붙는다


def test_answered_topic_is_not_asked_again(graph_env):
    """이미 다룬 토픽을 다시 묻지 않는다 — pending에서 빠지는지 그래프 수준에서 확인."""
    fake, graph = graph_env(topics=["레이어 캐시", "bind-mount"],
                            verdicts=["explained", "explained"],
                            tool_plan={"오늘": ("write_daily", {})})

    converse(graph, "오늘 도커 공부했어", "t4")
    converse(graph, "이 주제로 시작", "t4")
    converse(graph, "변경 없는 층 재사용", "t4")
    converse(graph, "호스트 디렉터리 연결", "t4")

    assert fake.recall_topics == ["레이어 캐시", "bind-mount"]  # 각 토픽 정확히 한 번


# ---------------- 저장/비저장 툴의 응답 형태 ----------------

def test_til_tool_reports_saved_document_and_lets_agent_speak(graph_env):
    """저장하는 툴은 saved_documents를 채우고, 마지막 발화는 agent가 한다(인출 세션과 반대)."""
    _, graph = graph_env(tool_plan={"회고": ("write_til", {
        "what": "그래프 배선 고침", "learned": "정적 엣지는 goto로 안 지워진다",
        "troubleshooting": "", "reflection": "테스트가 먼저였어야 했다",
        "actionplan": "", "keywords": ["langgraph"],
    })})

    res = converse(graph, "오늘 회고 남겨줘", "t5")

    assert [d.type for d in res.saved_documents] == ["til"]
    assert res.answer == "정리해서 알려줄게요"


def test_answer_question_never_looks_like_a_saved_document(graph_env):
    """회귀: 예전엔 모든 ToolMessage 본문을 Path(...)로 파싱해서 QA 답변까지 '저장된 문서'가 됐다."""
    _, graph = graph_env(tool_plan={"뭐야": ("answer_question", {"question": "합성곱이 뭐야"})})

    res = converse(graph, "합성곱이 뭐야", "t6")

    assert res.saved_documents == []
    assert res.tools_used == ["answer_question"]
    assert res.sources == ["data/processed/convolutional-networks.md"]


def test_topicless_start_falls_back_to_agent_without_opening_a_session(graph_env):
    """토픽을 못 뽑으면 세션을 열지 않고 agent가 되묻는다 — pending이 비어 있어야 한다."""
    _, graph = graph_env(topics=[], tool_plan={"오늘": ("write_daily", {})})

    res = converse(graph, "오늘 뭐 좀 했어", "t7")

    assert graph.get_state({"configurable": {"thread_id": "t7"}}).values.get("pending") in (None, [])
    assert res.saved_documents == []
    assert res.answer == "정리해서 알려줄게요"  # ToolMessage가 아니라 agent 발화로 마무리
