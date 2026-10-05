"""
FinTok AI · src/video/avatar_echomimic.py

Proveedor de avatar animado: envía el retrato y el audio de una escena al
Space público de Hugging Face "fffiloni/EchoMimic" (modelo EchoMimic, lip-sync
guiado por audio) y devuelve un vídeo de la cara hablando.

- Caché: si ya se generó ese mismo avatar + audio + parámetros, no se vuelve a
  llamar al Space (ahorra cuota de GPU y hace la demo instantánea).
- Si el Space falla (caído, sin cuota, cola), la escena queda sin vídeo y el
  montador usa el avatar de 3 bocas como respaldo.

Uso:
    from avatar_echomimic import attach_talking_heads
    spec = attach_talking_heads(spec, avatar_dir="assets/avatar", token=HF_TOKEN)
    render_video(spec, ...)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil

from PIL import Image
from scipy.io import wavfile

SPACE_ID = "fffiloni/EchoMimic"
API_NAME = "/generate_video"
DEFAULTS = {"width": 512, "height": 512, "seed": 420, "cfg": 2.5, "steps": 30, "fps": 24}


def _flatten_avatar(png_path: str, out_path: str, bg=(225, 233, 245)) -> str:
    """EchoMimic necesita una imagen sin transparencia: ponemos el avatar sobre fondo liso
    (el mismo color que la burbuja del montador, para que no se note el borde)."""
    im = Image.open(png_path).convert("RGBA")
    base = Image.new("RGBA", im.size, bg + (255,))
    base.alpha_composite(im)
    base.convert("RGB").resize((DEFAULTS["width"], DEFAULTS["height"]), Image.LANCZOS).save(out_path)
    return out_path


def _audio_seconds(path: str) -> float:
    sr, data = wavfile.read(path)
    return len(data) / sr


def _cache_key(*files: str, params: dict) -> str:
    h = hashlib.sha1()
    for f in files:
        with open(f, "rb") as fh:
            h.update(fh.read())
    h.update(json.dumps(params, sort_keys=True).encode())
    return h.hexdigest()[:16]


def generate_talking_head(image_path: str, audio_path: str, cache_dir: str = "outputs/avatar_videos",
                          token: str | None = None, client=None, **params) -> str:
    """Devuelve la ruta a un MP4 con la cara hablando. Lanza excepción si el Space falla."""
    os.makedirs(cache_dir, exist_ok=True)
    p = {**DEFAULTS, **params}
    p["length"] = int(math.ceil(_audio_seconds(audio_path) * p["fps"])) + p["fps"]  # nº de fotogramas + margen

    flat = os.path.join(cache_dir, "avatar_flat.png")
    _flatten_avatar(image_path, flat)
    out = os.path.join(cache_dir, f"{_cache_key(flat, audio_path, params=p)}.mp4")
    if os.path.exists(out):
        return out  # ya generado antes

    from gradio_client import Client, handle_file  # import aquí: solo hace falta si se usa este proveedor
    client = client or Client(SPACE_ID, token=token)
    result = client.predict(
        uploaded_img=handle_file(flat),
        uploaded_audio=handle_file(audio_path),
        width=p["width"], height=p["height"], length=p["length"], seed=p["seed"],
        cfg=p["cfg"], steps=p["steps"], fps=p["fps"],
        api_name=API_NAME,
    )
    video = result.get("video") if isinstance(result, dict) else result
    shutil.copy(video, out)
    return out


def attach_talking_heads(spec: dict, avatar_dir: str = "assets/avatar", token: str | None = None,
                         default_speaker: str = "andrea", verbose: bool = True, **params) -> dict:
    """Añade 'avatar_video_path' a cada escena con audio. Si una escena falla, se deja sin vídeo
    y el montador usará el avatar de 3 bocas para esa escena."""
    client = None
    for s in spec["scenes"]:
        if not s.get("audio_path") or not s.get("verified", True):
            continue
        speaker = s.get("speaker") or default_speaker
        img = os.path.join(avatar_dir, f"{speaker}_closed.png")
        try:
            if client is None:
                from gradio_client import Client
                client = Client(SPACE_ID, token=token)
            s["avatar_video_path"] = generate_talking_head(img, s["audio_path"], token=token, client=client, **params)
            if verbose:
                print(f"✓ escena {s.get('scene_id')}: {s['avatar_video_path']}")
        except Exception as e:  # Space caído, sin cuota, cola, etc.
            s.pop("avatar_video_path", None)
            if verbose:
                print(f"✗ escena {s.get('scene_id')}: EchoMimic no disponible ({type(e).__name__}: {e}). "
                      f"Se usará el avatar de respaldo.")
    return spec
