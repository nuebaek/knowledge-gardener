from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from app.rag.chain import build_llm

from app.services import corpus_service


MINDMAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a study-recap architect. The input is a set of documents the user selected "
     "themselves — a mix of raw source material (lecture notes, references, unknown internal "
     "structure) and the user's own previously written daily/weekly/retrospective notes. Your "
     "job is NOT to summarize each document — it is to reorganize everything across ALL of them "
     "into one hierarchical topic map that helps the user recap what they learned, top-down. "
     "Match the language of the input.\n\n"
     "Output format: Mind Elixir plaintext format ONLY. No prose, no explanation before or "
     "after — just the outline. Rules:\n"
     "- Each line is one node: `-`, one space, then a short label. A label is a term or a "
     "3-6 word phrase — NEVER a full sentence.\n"
     "- Indentation is exactly 2 spaces per nesting level.\n"
     "- The outline may have more than one top-level (0-indent) node. Do not force everything "
     "under a single root.\n"
     "- Nest a concept under another ONLY when the source material itself shows that relationship. "
     "Never nest based on your own domain knowledge.\n"
     "- Add a node only for something stated in the provided documents. Do not invent facts.\n\n"
     "(Silent step — do not print this as a section.) Before writing, read everything once and "
     "group by real topic first. A document boundary is not a topic boundary."),
    ("human",
     "Selected documents ({doc_count}건, `---`로 구분):\n{documents}"),
])


@lru_cache(maxsize=1)
def _llm():
    return build_llm()


def generate_mindmap_plaintext(documents: list[str]) -> str:
    united_document = "\n\n---\n\n".join(documents)
    messages = MINDMAP_PROMPT.invoke({
        "doc_count": len(documents),
        "documents": united_document,
    }).to_messages()
    return _llm().invoke(messages).content


def visualize_mindmap_text(documents: list[str]):
    if not documents:
        return "해당하는 문서를 찾지 못했어요."

    texts = [corpus_service.get_document(d).content for d in documents]
    return generate_mindmap_plaintext(texts)