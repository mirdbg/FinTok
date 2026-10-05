"""Generate FinTok scene audio with F5-TTS.

Input:
    JSON produced by the text pipeline. Each scene must contain:
    - scene_id
    - text
    - speaker

The `speaker` field in the JSON is the source of truth. This module does not
assign speakers based on scene number.

Output:
    - One WAV file per scene.
    - A copy of the JSON with `audio_path` populated.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf
from f5_tts.api import F5TTS


# ---------------------------------------------------------------------------
# Voice configuration
# ---------------------------------------------------------------------------
# Keys are lowercase because speaker names from the JSON are normalized
# internally with .strip().lower().
#
# IMPORTANT:
# ref_text should match exactly what is spoken in the corresponding WAV.
# If ref_text is empty, F5-TTS will handle/transcribe the reference internally.
SPEAKERS = {
    "andrea": {
        "audio": "data/audio_samples/03_EN_ANDREA_SHORT_FAST.wav",
        "ref_text": (
            "Welcome back. Stock prices are moving and investors are asking "
            "the same question. What happens next? Let's get started."
        ),
    },
    "miriam": {
        "audio": "data/audio_samples/06_EN_MIRIAM_SHORT_FAST.wav",
        "ref_text": "",
    },
}


def _repo_root() -> Path:
    """Infer the FinTok repository root from src/audio/audio_generator.py."""
    return Path(__file__).resolve().parents[2]


def _validate_payload(payload: dict[str, Any]) -> None:
    """Validate the minimum JSON structure required by the audio pipeline."""
    scenes = payload.get("scenes")

    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Input JSON must contain a non-empty `scenes` list.")

    for scene in scenes:
        scene_id = scene.get("scene_id")

        if scene_id is None:
            raise ValueError("Every scene must contain `scene_id`.")

        if not str(scene.get("text", "")).strip():
            raise ValueError(f"Scene {scene_id} has no `text`.")

        if not str(scene.get("speaker", "")).strip():
            raise ValueError(f"Scene {scene_id} has no `speaker` assigned.")

        speaker_key = str(scene["speaker"]).strip().lower()
        if speaker_key not in SPEAKERS:
            raise ValueError(
                f"Unknown speaker in scene {scene_id}: {scene['speaker']!r}. "
                f"Available speakers: {', '.join(SPEAKERS)}"
            )


def _normalise_reference(input_path: Path, output_path: Path) -> Path:
    """Convert a speaker reference WAV to mono, 24 kHz."""
    if not input_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "24000",
            "-vn",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return output_path


def generate_audio_from_json(
    input_json: str | Path,
    output_dir: str | Path | None = None,
    output_json: str | Path | None = None,
    speed: float = 1.0,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Generate one audio file per scene.

    Speaker selection comes exclusively from `scene["speaker"]`.

    Example:
        "speaker": "ANDREA" -> SPEAKERS["andrea"]
        "speaker": "MIRIAM" -> SPEAKERS["miriam"]

    The original speaker string is preserved in the output JSON.
    """
    # Avoid invalid values inherited by subprocesses in some Colab runtimes.
    os.environ["PYTHONHASHSEED"] = "0"

    root = Path(repo_root).resolve() if repo_root else _repo_root()
    input_json = Path(input_json).resolve()

    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    with input_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    _validate_payload(payload)

    company = str(payload.get("company", "UNKNOWN"))
    year = str(payload.get("year", "UNKNOWN"))

    if output_dir is None:
        output_dir = root / "data" / "audio_output" / f"{company}_{year}"
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = root / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    if output_json is None:
        output_json = output_dir / "script_with_audio.json"
    else:
        output_json = Path(output_json)
        if not output_json.is_absolute():
            output_json = root / output_json

    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Only prepare the speakers that actually appear in this JSON.
    used_speakers = {
        str(scene["speaker"]).strip().lower()
        for scene in payload["scenes"]
    }

    temp_dir = Path(tempfile.mkdtemp(prefix="fintok_refs_"))
    normalized_refs: dict[str, Path] = {}

    for speaker_key in used_speakers:
        source_path = root / SPEAKERS[speaker_key]["audio"]
        normalized_refs[speaker_key] = _normalise_reference(
            source_path,
            temp_dir / f"{speaker_key}_reference.wav",
        )

    print("Loading F5-TTS...")
    model = F5TTS()

    for scene in payload["scenes"]:
        scene_id = int(scene["scene_id"])
        scene_type = str(scene.get("scene_type", "scene")).strip().lower()
        text = str(scene["text"]).strip()

        # JSON is the source of truth.
        speaker_original = str(scene["speaker"]).strip()
        speaker_key = speaker_original.lower()

        safe_type = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in scene_type
        )

        filename = (
            f"scene_{scene_id:02d}_{safe_type}_{speaker_key}.wav"
        )
        audio_path = output_dir / filename

        print(
            f"[{scene_id}] {scene_type} | "
            f"speaker={speaker_original} | "
            f"{text[:90]}"
        )

        wav, sr, _ = model.infer(
            ref_file=str(normalized_refs[speaker_key]),
            ref_text=SPEAKERS[speaker_key]["ref_text"],
            gen_text=text,
            speed=speed,
        )

        sf.write(audio_path, wav, sr)

        try:
            stored_path = audio_path.resolve().relative_to(root).as_posix()
        except ValueError:
            stored_path = audio_path.resolve().as_posix()

        # Keep scene["speaker"] exactly as it came from the text pipeline.
        scene["audio_path"] = stored_path

    with Path(output_json).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Generated {len(payload['scenes'])} audio files")
    print(f"✓ Audio directory: {output_dir}")
    print(f"✓ Updated JSON: {output_json}")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one FinTok WAV per JSON scene using the "
            "speaker specified in each scene."
        )
    )
    parser.add_argument("input_json", help="Path to the JSON produced by src/text.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for generated WAV files.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for the JSON with audio_path populated.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="F5-TTS generation speed. Default: 1.0.",
    )

    args = parser.parse_args()

    generate_audio_from_json(
        input_json=args.input_json,
        output_dir=args.output_dir,
        output_json=args.output_json,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
