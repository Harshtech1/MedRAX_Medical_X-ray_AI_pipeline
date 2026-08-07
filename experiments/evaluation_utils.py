import re
from typing import Any, Optional


ANSWER_PATTERN = re.compile(r"\b([A-F])\b", re.IGNORECASE)


def extract_answer_letter(answer: Any) -> Optional[str]:
    """Extract the first standalone A-F answer choice from a model response."""
    if answer is None:
        return None

    match = ANSWER_PATTERN.search(str(answer).strip())
    return match.group(1).upper() if match else None
