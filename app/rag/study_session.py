def flatten_conversation(messages) -> str:
    return "\n".join(m.content if isinstance(m.content, str) else str(m.content) for m in messages)


def serialize_for_daily_note(state: dict) -> str:
    lines = [f"[{item['topic']}] {item['explanation']}" for item in state["answered"]]
    if state["seedlings"]:
        lines.append("")
        lines.append("설명하지 못한 것 (다시 꺼내볼 것):")
        for item in state["seedlings"]:
            lines.append(f"- {item['topic']}: {item['user_wording']}")
            # source_paths는 finalize에서만 붙는다 — 답이 아니라 근거 위치만 준다.
            for path in item.get("source_paths", []):
                lines.append(f"  다시 볼 근거: {path}")
    return "\n".join(lines)


def new_session(topics: list) -> dict:
    return {"pending": list(topics), "answered": [], "seedlings": []}


def apply_verdict(state: dict, topic: str, verdict: str, wording: str) -> dict:
    answered = list(state["answered"])
    seedlings = list(state["seedlings"])

    if verdict == "explained":
        answered.append({"topic": topic, "explanation": wording})
    else:  # partial, skip 둘 다 🌱로 — 구분은 wording 내용이 한다.
        seedlings.append({"topic": topic, "user_wording": wording})

    return {
        "pending": [t for t in state["pending"] if t != topic],
        "answered": answered,
        "seedlings": seedlings,
    }


def is_complete(state: dict) -> bool:
    return not state["pending"]
