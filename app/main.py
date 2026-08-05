import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import converse, corpus
from app.core import catalog
from app.core.paths import STATIC_DIR
from app.rag.graph import build_agent_graph

# force=True 없으면 uvicorn이 이미 건 root 핸들러 때문에 조용히 무시된다.
logging.basicConfig(force=True)
logging.getLogger("app").setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    catalog.sync_from_disk()
    app.state.agent = build_agent_graph()
    yield


app = FastAPI(title="Document RAG", lifespan=lifespan)
app.include_router(converse.router)
app.include_router(corpus.router)

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
