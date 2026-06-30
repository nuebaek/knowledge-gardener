"""RAG 컨트롤러 — 체인 호출과 응답 변환을 담당."""
from fastapi import HTTPException

from schemas import AskResponse


def ask(rag_chain, question: str) -> AskResponse:
    """체인을 호출해 답변과 출처를 반환. 실패 시 500으로 변환."""
    try:
        result = rag_chain.invoke(question)
    except Exception as exc:  # LLM 호출 실패 등
        raise HTTPException(status_code=500, detail=f"RAG 실행 실패: {exc}") from exc

    sources = result.get("sources", [])
    if not sources:
        # 검색 결과가 0건이면 근거 없이 답한 셈 → 호출 측에 신호
        raise HTTPException(status_code=404, detail="질문과 관련된 문서를 찾지 못했습니다.")

    return AskResponse(answer=result["answer"], sources=sources)
