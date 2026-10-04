"""Factual idea extraction from consecutive 10-K chunks."""

import json
import re
from .chunk_selector import context_to_text


EXTRACT_SYSTEM = """
You are a financial analyst specialized in SEC 10-K filings.

Read the supplied excerpts from three consecutive chunks of the same 10-K section.

Extract up to 3 distinct, self-contained factual ideas that are explicitly
supported by the text.

Prioritize:
- important quantitative results or changes,
- year-over-year developments,
- business drivers,
- strategic developments,
- customer or supplier dependencies,
- material risks,
- unusual or potentially meaningful facts.

Rules:
- Use ONLY information contained in the supplied excerpts.
- Do not use external knowledge.
- Do not speculate.
- Do not exaggerate.
- Combine information across the three chunks when they clearly belong to
  the same idea.
- Do not create separate ideas just because the same information appears
  across overlapping chunks.
- Return fewer than 3 ideas if necessary.
- If nothing meaningful exists, return an empty list.
- This is factual extraction, NOT social-media copywriting.

Return ONLY valid JSON:

{
  "ideas": [
    {
      "idea": "...",
      "evidence": "...",
      "source_chunks": ["chunk_id"]
    }
  ]
}
""".strip()


def parse_json_output(text: str):
    """
    Parse JSON returned by the model.

    Handles:
    - accidental ```json markdown fences;
    - text before/after the JSON;
    - unescaped control characters generated inside strings.
    """
    text = text.strip()

    # Remove accidental Markdown fences
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    # Keep only the JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            f"No JSON object found in model output:\n{text}"
        )

    text = text[start:end + 1]

    # strict=False tolerates literal control characters
    # occasionally produced by the LLM inside JSON strings
    return json.loads(text, strict=False)


def extract_facts(context_df, generate_fn):
    """
    Extract up to three grounded candidate ideas.

    Parameters
    ----------
    context_df:
        DataFrame containing the consecutive source chunks.
    generate_fn:
        Callable(system_prompt, user_prompt, max_new_tokens) -> (text, latency)
    """
    target = context_df.iloc[0]
    context_text = context_to_text(context_df)

    user_prompt = f"""
COMPANY/TICKER: {target['ticker']}
FISCAL YEAR: {target['fiscal_year']}
10-K ITEM: {target['item']}

CONSECUTIVE 10-K EXCERPTS:
----------------
{context_text}
----------------
""".strip()

    raw_output, latency = generate_fn(
        EXTRACT_SYSTEM,
        user_prompt,
        max_new_tokens=550,
    )

    data = parse_json_output(raw_output)

    if "ideas" not in data or not isinstance(data["ideas"], list):
        raise ValueError("Invalid EXTRACT output: expected an 'ideas' list.")

    return {
        "data": data,
        "raw_output": raw_output,
        "latency": latency,
    }
