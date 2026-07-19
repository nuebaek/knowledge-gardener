from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.main import api_router
from app.core.paths import STATIC_DIR
from app.rag.graph import build_agent_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = build_agent_graph()
    yield


app = FastAPI(title="Document RAG", lifespan=lifespan)
app.include_router(api_router)

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
