MAX_CONTEXT_MESSAGES = 20


def flatten_conversation(messages) -> str:
    recent = messages[-MAX_CONTEXT_MESSAGES:]
    return "\n".join(f"{m.type}: {m.content if isinstance(m.content, str) else str(m.content)}" for m in recent)


def serialize_for_daily_note(state: dict) -> str:
    lines = [f"[{item['topic']}] {item['explanation']}" for item in state["answered"]]
    if state["seedlings"]:
        lines.append("")
        lines.append("설명하지 못한 것 (다시 꺼내볼 것):")
        for item in state["seedlings"]:
            lines.append(f"- {item['topic']}: {item['user_wording']}")
            for path in item.get("source_paths", []):
                lines.append(f"  다시 볼 근거: {path}")
    return "\n".join(lines)


def new_session(topics: list, umbrella: str = "") -> dict:
    return {"pending": list(topics), "answered": [], "seedlings": [], "umbrella": umbrella}


def apply_verdict(state: dict, topic: str, verdict: str, wording: str) -> dict:
    answered = list(state["answered"])
    seedlings = list(state["seedlings"])

    if verdict == "explained":
        answered.append({"topic": topic, "explanation": wording})
    else:
        seedlings.append({"topic": topic, "user_wording": wording})

    return {
        "pending": [t for t in state["pending"] if t != topic],
        "answered": answered,
        "seedlings": seedlings,
    }


def is_complete(state: dict) -> bool:
    return not state["pending"]
