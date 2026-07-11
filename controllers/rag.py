"""RAG 컨트롤러 — 체인 호출과 응답 변환을 담당."""
from fastapi import HTTPException

from schemas import AskResponse


def ask(rag_chain, question: str) -> AskResponse:
    """체인을 호출해 답변과 출처를 반환. 실패 시 500으로 변환.

    관련 문서를 못 찾은 경우도 LLM이 "자료에 없다"고 정상적으로 답한 것이므로 200으로 반환한다.
    빈 sources가 바로 그 신호이며, 호출 측(프론트엔드)이 이를 보고 UI를 다르게 처리할 수 있다.
    """
    try:
        result = rag_chain.invoke({"question": question})
    except Exception as exc:  # LLM 호출 실패 등
        raise HTTPException(status_code=500, detail=f"RAG 실행 실패: {exc}") from exc

    return AskResponse(answer=result["answer"], sources=result.get("sources", []))
