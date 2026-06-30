from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="user query")


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
