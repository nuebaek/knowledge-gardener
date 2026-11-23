import pytest
from fastapi import HTTPException

from app.services import rag_service


class _FailingGraph:
    def get_state(self, config):
        return type("S", (), {"values": {}})()

    def invoke(self, graph_input, config):
        raise RuntimeError("upstream 429: api_key=sk-secret-123 at /internal/path.py")


def test_converse_failure_does_not_leak_exception_details():
    """회귀: 500 응답 detail에 예외 원문(키·내부 경로가 섞일 수 있음)이 그대로 노출되던 문제."""
    with pytest.raises(HTTPException) as info:
        rag_service.converse(_FailingGraph(), "안녕", "t-err")

    assert info.value.status_code == 500
    assert "sk-secret" not in info.value.detail
    assert "RuntimeError" not in info.value.detail
