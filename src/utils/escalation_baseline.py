import re

from src.configuration.red_flags import RED_FLAGS

def baseline_escalation(report: str) -> bool:
    text = report.lower()
    hits = []

    for category, keywords in RED_FLAGS.items():
        for kw in keywords:
            if re.search(rf"\b{kw}\b", text):
                hits.append((category, kw))

    severe_markers = {"hypoton", "somnolent", "tachykard"}
    severe_hit = any(kw in text for kw in severe_markers)

    if len(hits) >= 1:
        return True
    if len(hits) >= 1 and severe_hit:
        return True

    return False

