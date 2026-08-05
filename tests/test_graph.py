from app.rag.graph import check_pending

def test_pending_study():
  result = check_pending({"pending": ["RAG"]})
  assert result == "study"

def test_pending_empty_agent():
  result = check_pending({"pending": []})
  assert result == "agent"

def test_pending_key_agent():
  result = check_pending({})
  assert result == "agent"

