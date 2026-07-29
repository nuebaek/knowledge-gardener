from typing import Annotated

from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph import MessagesState

# State 정의
class GraphState(TypedDict):
    question: str
    document: list[Document]
    answer: str
    sources: list[str]
    doc_scores: list[float]
    is_relevant: bool
    retry_count: int
    rewritten_question: str | None


def append_records(left, right):
    """add_messages와 같은 append 리듀서 — 누가 언제 넣든 뒤에 붙기만 한다.

    저장 사실을 ToolMessage 문자열 파싱으로 복원하면 저장/비저장 툴을 구분 못 해 오인·누락된다
    (자세한 근거: docs/2026-07-29-study-loop-design.md §7). 그래서 구조화해 상태 채널에 싣는다.
    """
    return (left or []) + (right or [])


class StudySessionState(MessagesState):
    pending: list[str]
    answered: list[dict]
    seedlings: list[dict]
    # 현재 토픽이 stay_on_topic=True로 여러 턴 파고들어질 때, 그 라운드들의 사용자 원문을
    # 이어 붙여 두는 버퍼. 토픽이 최종 기록될 때(apply_verdict) 한 번에 넘기고 ""로 리셋한다.
    # (리듀서 없음 — 매 턴 study_node가 전체 값을 새로 계산해 덮어쓴다.)
    current_explanation: str
    # 이번 턴에 새로 생긴 것만 잘라 쓰기 위해 누적 채널로 둔다(서비스가 turn 전후 길이로 슬라이스).
    saved_documents: Annotated[list[dict], append_records]
    mindmaps: Annotated[list[str], append_records]
