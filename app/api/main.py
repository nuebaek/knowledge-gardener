from fastapi import APIRouter

from app.api.routes import converse, corpus

api_router = APIRouter()
api_router.include_router(converse.router)
api_router.include_router(corpus.router)
