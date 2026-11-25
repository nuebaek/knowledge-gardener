from typing import Literal

from pydantic import BaseModel, Field


class TopicList(BaseModel):
    topics: list[str]
    umbrella: str = ""


class DimScore(BaseModel):
    evidence: str = Field(description="이 점수의 근거가 된 사용자 발화 원문 인용. 없으면 빈 문자열")
    score: Literal[0, 1, 2]


class RubricResult(BaseModel):
    """judge LLM의 관찰 결과. 판정(통과·꼬리질문 여부)은 chain.decide()가 코드로 내린다."""
    skipped: bool = Field(description="모른다/패스 등 설명 시도 자체가 없음")
    hedged: bool = Field(description="최신 발화에 스스로의 불확실성 표지(?, 음.., ~것 같은데, 들었어 등)가 있음")
    hedge_evidence: str = ""
    definition: DimScore
    mechanism: DimScore
    application: DimScore
    connection: DimScore
    follow_up: str | None = Field(None, description="가장 약한 항목을 겨냥한 꼬리질문 1개")
    next_topic_question: str | None = Field(None, description="next_topic 설명을 요청하는 질문")


class TurnResult(BaseModel):
    """study_node가 소비하는 결정. verdict는 태도(확신)만 — 통과 여부는 passed."""
    verdict: Literal["explained", "partial", "skip"]
    stay_on_topic: bool = False
    next_question: str | None = None
    passed: bool = True
    weak: list[str] = []

    @property
    def outcome(self) -> str:
        """기록 결과: 확신 있게 설명했어도 깊이 미달이면 🌱(partial)로 남긴다."""
        return "partial" if self.verdict == "explained" and not self.passed else self.verdict


class ConverseRequest(BaseModel):
    message: str
    thread_id: str
    selected_topics: list[str] | None = None


class SavedDocument(BaseModel):
    type: str
    file_name: str


class ConverseResponse(BaseModel):
    answer: str
    tools_used: list[str]
    saved_documents: list[SavedDocument] = []
    mindmap_plaintext: str | None = None
    sources: list[str] = []
    recall: list[str] = []
    awaiting_topic_confirm: bool = False
    topics: list[str] = []


class ThreadMessage(BaseModel):
    role: str
    content: str


class ThreadHistoryResponse(BaseModel):
    thread_id: str
    messages: list[ThreadMessage]