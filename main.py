from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from graph import build_rag_graph
from routers.ask import router as ask_router
from routers.corpus import router as corpus_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = build_rag_graph()
    yield


app = FastAPI(title="Document RAG", lifespan=lifespan)
app.include_router(ask_router)
app.include_router(corpus_router)

app.mount("/", StaticFiles(directory="static", html=True), name="static")
