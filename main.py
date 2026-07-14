from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from graph import build_agent_graph
from routers.converse import router as converse_router
from routers.corpus import router as corpus_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = build_agent_graph()
    yield


app = FastAPI(title="Document RAG", lifespan=lifespan)
app.include_router(converse_router)
app.include_router(corpus_router)

app.mount("/", StaticFiles(directory="static", html=True), name="static")
