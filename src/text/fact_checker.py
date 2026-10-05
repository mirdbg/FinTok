"""Grounding and editorial selection for FinTok candidate ideas."""

import json
import re
from .chunk_selector import context_to_text


SELECT_SYSTEM = """
You are the financial editor of FinTok.

Your task is to decide whether any of the candidate ideas extracted from a
10-K is genuinely worth communicating to a general but financially interested
audience.

A fact being correct does NOT automatically make it interesting.

Good candidates tend to involve:
- substantial changes in the business,
- striking financial figures,
- important growth or declines,
- meaningful risks,
- important dependencies,
- strategic changes,
- information that reveals something useful about how the company operates.

Reject ideas that are mainly:
- administrative,
- routine accounting language,
- legal boilerplate,
- too technical without broader relevance,
- impossible to understand without extensive context,
- repetitive, common knowledge or trivial.

Be selective.
Select AT MOST ONE idea. It is important to select none if none of the candidates are sufficiently interesting.

Use the original excerpts to verify that the selected idea is properly
supported.

Return ONLY valid JSON:

{
  "decision": "KEEP or DISCARD",
  "selected_idea": "..." or null,
  "reason": "..."
}
""".strip()


def parse_json_output(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def select_fact(context_df, extraction, generate_fn):
    """
    Select at most one candidate idea and verify it against the source context.
    """
    context_text = context_to_text(context_df)

    extraction_data = (
        extraction["data"] if "data" in extraction else extraction
    )

    if not extraction_data.get("ideas"):
        return {
            "data": {
                "decision": "DISCARD",
                "selected_idea": None,
                "reason": "No candidate ideas were extracted.",
            },
            "raw_output": None,
            "latency": 0.0,
        }

    candidate_json = json.dumps(extraction_data, ensure_ascii=False)

    user_prompt = f"""
ORIGINAL 10-K EXCERPTS:
----------------
{context_text}
----------------

CANDIDATE IDEAS:
----------------
{candidate_json}
----------------
""".strip()

    raw_output, latency = generate_fn(
        SELECT_SYSTEM,
        user_prompt,
        max_new_tokens=300,
    )

    data = parse_json_output(raw_output)

    decision = str(data.get("decision", "")).upper()
    if decision not in {"KEEP", "DISCARD"}:
        raise ValueError("Invalid SELECT output: decision must be KEEP or DISCARD.")

    return {
        "data": data,
        "raw_output": raw_output,
        "latency": latency,
    }
