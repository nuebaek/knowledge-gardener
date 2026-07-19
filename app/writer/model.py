from datetime import date

from pydantic import BaseModel

class DailynoteEntry(BaseModel):
    schema_version: int = 1
    date: date
    topic: str
    learned: str
    related_concepts: list[str] = []


class WeeklynoteEntry(BaseModel):
    schema_version: int = 1
    date: date
    topics: list[str] = []
    related_concepts: list[str] = []


class TilEntry(BaseModel):
    schema_version: int = 1
    date: date
    what: str
    learned: str
    troubleshooting: str
    reflection: str
    actionplan: str
    keywords: list[str]