from langchain_core.prompts import ChatPromptTemplate
from app.rag.chain import build_llm

from pathlib import Path
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


llm = build_llm()


def generate_mindmap_plaintext(documents: list[str]) -> str:
    united_document = "\n\n---\n\n".join(documents)
    messages = MINDMAP_PROMPT.invoke({
        "doc_count": len(documents),
        "documents": united_document,
    }).to_messages()
    return llm.invoke(messages).content


# def save_mindmap_html(plaintext: str, out_path: str = "mindmap_output_ff3.html") -> str:
#     html = f"""<!DOCTYPE html>
# <html>
# <head>
#   <meta charset="utf-8"><title>mindmap</title>
#   <style>
#     body {{ margin: 0; }}
#     #map {{ width: 100vw; height: 100vh; }}
#   </style>
# </head>
# <body>
#   <div id="map"></div>

#   <script type="module">
#     import MindElixir from "https://cdn.jsdelivr.net/npm/mind-elixir/dist/MindElixir.js";
#     import {{ plaintextToMindElixir }} from "https://cdn.jsdelivr.net/npm/mind-elixir/dist/PlaintextConverter.js";

#     const plaintext = {json.dumps(plaintext)};
#     const data = plaintextToMindElixir(plaintext);

#     const mind = new MindElixir({{ el: "#map", direction: MindElixir.RIGHT }});
#     mind.init(data);
#   </script>
#   <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/mind-elixir/dist/MindElixir.css">
# </body>
# </html>
# """
#     Path(out_path).write_text(html, encoding="utf-8")
#     return str(Path(out_path).resolve())


# mindmap
def visualize_mindmap_img(documents: list[str]):
    if not documents:
        return "해당하는 문서를 찾지 못했어요."

    # 문서를 하나로 뭉쳐준다
    texts = []
    for d in documents:
        texts.append(corpus_service.get_document(d).content)

    # 뭉친 언어 기반 html mindmap 생성한다
    plaintext = generate_mindmap_plaintext(texts)

    # output: 보여준다?
    # save_mindmap_html(plaintext)

    return plaintext