"""FinTok script generation and deterministic final JSON construction."""

import json
import re
from .chunk_selector import context_to_text


INTRO = "This is FinTok — your daily dose of financial knowledge."
OUTRO = "That's your FinTok for today. See you in the next one."


WRITE_SYSTEM = """
You are a financial content writer for FinTok, a platform that transforms
official 10-K filings into short, engaging and easy-to-understand financial content.

Turn the selected financial idea into a natural spoken script for a short video.

Use the original 10-K excerpts as the ONLY factual source.

Your goal is to communicate ONE interesting financial story clearly and naturally.

1. HOOK

Write ONE short and engaging question or surprising factual statement based
on the selected idea.

The hook should make the viewer curious and want to keep watching.

IMPORTANT:
- The hook must contain ONLY the hook.
- Do not explain the full story in the hook.
- Maximum 25 words.
- Make it specific to the selected financial story.

You may use formats such as:
- "Did you know that...?"
- "What if I told you that...?"
- "How much...?"
- "Why did...?"
- "One number in [company]'s latest filing stands out..."

Do NOT use misleading clickbait.
Do NOT exaggerate beyond what is supported by the source.

HOOK HIGHLIGHT

Also create a "hook_highlight".

This is a very short visual summary of the hook that may appear as large
on-screen text.

Rules:
- Ideally 3-6 words.
- Preserve the main idea of the hook.
- It may be a question or statement.
- It does NOT need to contain a number.
- Do not introduce new information.

2. EXPLANATION

Explain the selected idea clearly and naturally.

Write 2-3 concise sentences.

IMPORTANT:
- Maximum 70 words.
- Focus on ONE main financial story.
- Every sentence must add NEW information or context.
- Do not repeat the same idea in different words.
- Stop once the main idea has been explained.

The explanation must:
- identify the company naturally;
- preserve the most relevant figures and context;
- explain the fact in language understandable to a general interested audience;
- sound natural when spoken aloud;
- avoid unnecessary financial jargon;
- avoid simply copying regulatory language;
- avoid investment advice;
- avoid unsupported interpretations;
- use ONLY information supported by the supplied 10-K excerpts;
- simplify regulatory language into natural spoken English while preserving
  its factual meaning.

Do not try to summarize everything in the excerpts.

Only include information that directly helps explain the selected story.

EXPLANATION HIGHLIGHT

Also create an "explanation_highlight".

This is ONE short relevant fact from the explanation that may appear as
large on-screen text.

It may be:

- quantitative: a monetary amount, percentage, growth rate, ratio or
  other important numerical fact;

OR

- qualitative: an important business fact, risk, dependency, strategic
  development or limitation.

Rules:
- Ideally 3-7 words.
- Choose the most relevant or interesting fact from the explanation.
- Prefer a quantitative fact when an important number is central to the story.
- Otherwise use a concise qualitative fact.
- It must refer to information already contained in the explanation.
- Never introduce a new fact.

FINTOK INTRO AND OUTRO

The following two sentences will be added automatically later by Python:

"This is FinTok — your daily dose of financial knowledge."

"That's your FinTok for today. See you in the next one."

Therefore:
- DO NOT generate the FinTok intro.
- DO NOT generate the FinTok outro.

OUTPUT

Return ONLY valid JSON with EXACTLY this structure:

{
  "hook": "...",
  "hook_highlight": "...",
  "explanation": "...",
  "explanation_highlight": "..."
}

Do not include markdown.
Do not include ```json.
Do not include any text before or after the JSON.
""".strip()


def parse_json_output(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def generate_script(context_df, selection, generate_fn):
    """Generate hook, explanation and visual highlights for a KEEP decision."""
    selection_data = selection["data"] if "data" in selection else selection

    if str(selection_data.get("decision", "")).upper() != "KEEP":
        return None

    context_text = context_to_text(context_df)
    selected_story = json.dumps(selection_data, ensure_ascii=False)

    user_prompt = f"""
SELECTED FINANCIAL STORY:
----------------
{selected_story}
----------------

SOURCE 10-K EXCERPTS:
----------------
{context_text}
----------------

Write the hook, explanation and their visual highlights.

Remember:
- hook: maximum 25 words;
- explanation: maximum 70 words;
- use only information supported by the source.
""".strip()

    raw_output, latency = generate_fn(
        WRITE_SYSTEM,
        user_prompt,
        max_new_tokens=350,
    )

    data = parse_json_output(raw_output)

    required = {
        "hook",
        "hook_highlight",
        "explanation",
        "explanation_highlight",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"Invalid WRITE output; missing fields: {sorted(missing)}")

    return {
        "data": data,
        "raw_output": raw_output,
        "latency": latency,
    }


def build_video_json(context_df, script):
    """
    Build the final four-scene FinTok object deterministically.
    No LLM call is performed here.
    """
    if script is None:
        return None

    write_data = script["data"] if "data" in script else script
    target = context_df.iloc[0]
    source_chunk_ids = context_df["chunk_id"].tolist()

    video_data = {
        "company": str(target["ticker"]),
        "year": int(target["fiscal_year"]),
        "language": "en",
        "scenes": [
            {
                "scene_id": 1,
                "scene_type": "hook",
                "speaker": "ANDREA",
                "text": write_data["hook"],
                "key_figure": write_data["hook_highlight"],
                "source_chunk_ids": source_chunk_ids,
                "verified": True,
                "audio_path": None,
            },
            {
                "scene_id": 2,
                "scene_type": "intro",
                "speaker": "MIRIAM",
                "text": INTRO,
                "key_figure": None,
                "source_chunk_ids": [],
                "verified": True,
                "audio_path": None,
            },
            {
                "scene_id": 3,
                "scene_type": "explanation",
                "speaker": "MIRIAM",
                "text": write_data["explanation"],
                "key_figure": write_data["explanation_highlight"],
                "source_chunk_ids": source_chunk_ids,
                "verified": True,
                "audio_path": None,
            },
            {
                "scene_id": 4,
                "scene_type": "outro",
                "speaker": "ANDREA",
                "text": OUTRO,
                "key_figure": None,
                "source_chunk_ids": [],
                "verified": True,
                "audio_path": None,
            },
        ],
    }

    # Deterministic structural validation.
    assert len(video_data["scenes"]) == 4
    assert [s["scene_type"] for s in video_data["scenes"]] == [
        "hook", "intro", "explanation", "outro"
    ]
    assert video_data["scenes"][1]["text"] == INTRO
    assert video_data["scenes"][3]["text"] == OUTRO
    assert video_data["scenes"][1]["key_figure"] is None
    assert video_data["scenes"][3]["key_figure"] is None
    assert video_data["scenes"][1]["source_chunk_ids"] == []
    assert video_data["scenes"][3]["source_chunk_ids"] == []

    available_ids = set(context_df["chunk_id"])
    for scene in video_data["scenes"]:
        assert scene["speaker"] in {"MIRIAM", "ANDREA"}
        assert scene["audio_path"] is None
        for chunk_id in scene["source_chunk_ids"]:
            assert chunk_id in available_ids

    return video_data
