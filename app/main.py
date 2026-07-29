import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.main import api_router
from app.core import catalog
from app.core.paths import STATIC_DIR
from app.rag.graph import build_agent_graph

# app/ 안 logger.debug 호출(grade_docs, rewrite_query, study_turn 등)이 실제로 보이게.
# root를 DEBUG로 켜면 chromadb/httpx/langchain 내부 로그까지 쏟아져서, "app" 네임스페이스만
# DEBUG로 올린다 — 우리 로거는 전부 getLogger(__name__)이라 app.* 밑에 걸린다.
# force=True 필수: uvicorn이 이미 root에 핸들러를 달아놔서, 그냥 basicConfig()는
# "root에 핸들러 있으면 아무것도 안 함" 규칙에 걸려 조용히 무시된다.
logging.basicConfig(force=True)
logging.getLogger("app").setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 사람이 data/writer, data/processed에 직접 파일을 추가/삭제해도(앱을 거치지 않고)
    # 재시작할 때마다 카탈로그가 실제 디스크 상태로 맞춰지게 함.
    catalog.sync_from_disk()
    app.state.agent = build_agent_graph()
    yield


app = FastAPI(title="Document RAG", lifespan=lifespan)
app.include_router(api_router)

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
