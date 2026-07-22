from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.main import api_router
from app.core import catalog
from app.core.paths import STATIC_DIR
from app.rag.graph import build_agent_graph


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
