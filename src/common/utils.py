import re
from typing import Sequence


def normalize_text(text: str) -> str:
    return text.replace("\u0092", "'").replace("\u2019", "'").strip()


def section_between(text: str, start_pattern: str, end_patterns: Sequence[str]) -> str:
    match = re.search(start_pattern, text, flags=re.IGNORECASE)
    if not match:
        return ""
    start = match.end()
    end = len(text)
    for pattern in end_patterns:
        end_match = re.search(pattern, text[start:], flags=re.IGNORECASE)
        if end_match:
            end = min(end, start + end_match.start())
    return normalize_text(text[start:end])
