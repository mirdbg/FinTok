"""
FinTok AI · src/video/compositor.py

Montador de vídeo: recibe el JSON de escenas (guion + audio) y los avatares,
y genera un vídeo vertical 1080x1920 listo para redes sociales.

Por cada fotograma:
  fondo (con zoom lento) → cabecera + cifra clave → avatar en burbuja
  (boca según el volumen de la voz, rebote y balanceo) → subtítulos palabra
  a palabra → fuente + aviso legal.

Uso:
    from compositor import render_video
    render_video("scenes.json", avatar_dir="assets/avatar", out_path="outputs/video.mp4")
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.io import wavfile

# ───────────────────────────── estilo ─────────────────────────────


@dataclass
class Style:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    bg_top: tuple = (12, 20, 45)          # degradado de fondo si no hay imagen
    bg_bottom: tuple = (20, 60, 90)
    accent: tuple = (0, 214, 143)         # verde "finanzas"
    text: tuple = (255, 255, 255)
    muted: tuple = (190, 200, 215)
    bubble_d: int = 560                   # diámetro de la burbuja del avatar
    bubble_y: int = 1120                  # centro vertical de la burbuja
    bubble_bg: tuple = (225, 233, 245)
    mouth_thresholds: tuple = (0.12, 0.40)  # volumen para boca a medias / abierta
    disclaimer: str = "Generado con IA · No constituye recomendación de inversión"
    font_paths: list = field(default_factory=lambda: [
        "assets/fonts/Montserrat-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ])


def _font(style: Style, size: int) -> ImageFont.FreeTypeFont:
    for p in style.font_paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


# ───────────────────────────── audio ─────────────────────────────


def _read_wav(path: str, target_sr: int = 24000) -> np.ndarray:
    """Lee un WAV, lo pasa a mono float32 [-1, 1] y lo remuestrea a target_sr."""
    sr, data = wavfile.read(path)
    data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    peak = np.abs(data).max()
    if peak > 1.0:  # enteros (int16/int32) → [-1, 1]
        data /= 32768.0 if peak <= 32768 else 2147483648.0
    if sr != target_sr:
        n = int(len(data) * target_sr / sr)
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)
    return data


def _envelope(audio: np.ndarray, sr: int, fps: int) -> np.ndarray:
    """Volumen (RMS) por fotograma, normalizado a [0, 1] y suavizado."""
    hop = sr // fps
    n = max(1, len(audio) // hop)
    rms = np.array([np.sqrt(np.mean(audio[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
    ref = np.percentile(rms, 95) + 1e-9
    level = np.clip(rms / ref, 0, 1)
    smooth = np.copy(level)
    for i in range(1, len(smooth)):  # ataque rápido, caída suave
        a = 0.6 if level[i] > smooth[i - 1] else 0.35
        smooth[i] = a * level[i] + (1 - a) * smooth[i - 1]
    return smooth


# ───────────────────────────── escenas ─────────────────────────────


def _word_timings(scene: dict, duration: float) -> list[dict]:
    """Usa word_timestamps si existen; si no, reparte el tiempo según la longitud de cada palabra."""
    if scene.get("word_timestamps"):
        return scene["word_timestamps"]
    words = scene["text"].split()
    weights = np.array([len(w) + 2 for w in words], dtype=float)
    ends = np.cumsum(weights) / weights.sum() * duration * 0.97
    starts = np.concatenate([[0], ends[:-1]])
    return [{"word": w, "start": float(s), "end": float(e)} for w, s, e in zip(words, starts, ends)]


def _load_background(path: str | None, style: Style) -> Image.Image:
    w, h = style.width, style.height
    if path and os.path.exists(path):
        img = Image.open(path).convert("RGB")
        scale = max(w / img.width, h / img.height) * 1.15  # margen para el zoom
        return img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    grad = np.linspace(0, 1, h)[:, None, None]
    arr = (1 - grad) * np.array(style.bg_top) + grad * np.array(style.bg_bottom)
    return Image.fromarray(np.repeat(arr, w, axis=1).astype(np.uint8))


def _ken_burns(bg: Image.Image, progress: float, style: Style) -> Image.Image:
    """Zoom lento hacia dentro durante la escena."""
    w, h = style.width, style.height
    if bg.size == (w, h):
        return bg.copy()
    z = 1.0 + 0.08 * progress
    cw, ch = int(w * 1.1 / z), int(h * 1.1 / z)
    left, top = (bg.width - cw) // 2, (bg.height - ch) // 2
    return bg.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


# ───────────────────────────── avatar ─────────────────────────────


def _load_avatar(avatar_dir: str, speaker: str) -> dict:
    states = {}
    for state in ("closed", "half", "open"):
        p = os.path.join(avatar_dir, f"{speaker}_{state}.png")
        if os.path.exists(p):
            states[state] = Image.open(p).convert("RGBA")
    if "closed" not in states:
        raise FileNotFoundError(f"Falta {speaker}_closed.png en {avatar_dir}")
    states.setdefault("open", states["closed"])
    states.setdefault("half", states["open"])
    return states


def _bubble(avatar: Image.Image, level: float, t: float, appear: float, style: Style) -> Image.Image:
    """Avatar dentro de un círculo, con rebote al hablar y balanceo suave."""
    d = style.bubble_d
    inner = Image.new("RGBA", (d, d), style.bubble_bg + (255,))
    # rebote (sube y se estira con la voz) + balanceo continuo
    sx = 1.0 - 0.02 * level
    sy = 1.0 + 0.04 * level
    size = int(d * 1.05)
    av = avatar.resize((int(size * sx), int(size * sy)), Image.BILINEAR)
    av = av.rotate(2.5 * math.sin(t * 1.6), resample=Image.BICUBIC, expand=False)
    dy = int(-18 * level + 4 * math.sin(t * 2.2))
    inner.alpha_composite(av, ((d - av.width) // 2, d - av.height + int(d * 0.08) + dy))
    mask = Image.new("L", (d, d), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, d - 1, d - 1), fill=255)
    inner.putalpha(mask)

    ring = 14
    out = Image.new("RGBA", (d + 2 * ring, d + 2 * ring), (0, 0, 0, 0))
    ImageDraw.Draw(out).ellipse((0, 0, out.width - 1, out.height - 1), fill=style.accent + (255,))
    out.alpha_composite(inner, (ring, ring))
    # entrada con rebote al inicio del vídeo
    if appear < 1:
        s = max(0.01, 1 + 0.15 * math.sin(appear * math.pi) - (1 - appear) ** 2)
        out = out.resize((max(1, int(out.width * s)), max(1, int(out.height * s))), Image.BILINEAR)
    return out


# ───────────────────────────── textos ─────────────────────────────


def _draw_centered(draw, text, y, font, fill, width, stroke=0, stroke_fill=(0, 0, 0)):
    tw = draw.textlength(text, font=font)
    draw.text(((width - tw) / 2, y), text, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)


def _subtitle_group(timings: list[dict], t: float, group: int = 4):
    """Devuelve el grupo de palabras a mostrar y el índice de la que se está diciendo."""
    idx = 0
    for i, w in enumerate(timings):
        if t >= w["start"]:
            idx = i
    g0 = (idx // group) * group
    return [w["word"] for w in timings[g0:g0 + group]], idx - g0


def _draw_texts(frame, scene, spec, t_scene, duration, timings, fonts, style):
    d = ImageDraw.Draw(frame)
    W = style.width
    # cabecera
    header = f"{spec.get('company', '')} · {spec.get('source_label', '10-K ' + str(spec.get('year', '')))}"
    _draw_centered(d, header.upper(), 120, fonts["small"], style.muted, W)
    d.rounded_rectangle((W - 150, 100, W - 50, 150), 14, fill=style.accent)
    d.text((W - 128, 106), "IA", font=fonts["small"], fill=(10, 20, 40))
    # cifra clave con efecto "pop" al empezar la escena
    if scene.get("key_figure"):
        p = min(1.0, t_scene / 0.35)
        scale = 1 + 0.25 * math.sin(p * math.pi) if p < 1 else 1.0
        f = _font(style, int(170 * scale))
        _draw_centered(d, scene["key_figure"], 260 - int(30 * (scale - 1)), f, style.accent, W,
                       stroke=6, stroke_fill=(10, 20, 40))
    # subtítulos palabra a palabra
    words, active = _subtitle_group(timings, t_scene)
    font = fonts["sub"]
    y = 1500
    total = sum(d.textlength(w + " ", font=font) for w in words)
    if total > W - 120:  # si no cabe, reducir el tamaño de letra
        font = _font(style, int(font.size * (W - 120) / total))
        total = sum(d.textlength(w + " ", font=font) for w in words)
    x = (W - total) / 2
    for i, w in enumerate(words):
        color = style.accent if i == active else style.text
        d.text((x, y), w, font=font, fill=color, stroke_width=5, stroke_fill=(0, 0, 0))
        x += d.textlength(w + " ", font=font)
    # fuente y aviso legal
    src = scene.get("source")
    if src:
        _draw_centered(d, f"Fuente: {src}", 1740, fonts["tiny"], style.muted, W)
    _draw_centered(d, style.disclaimer, 1790, fonts["tiny"], style.muted, W)


# ───────────────────────────── render ─────────────────────────────


def render_video(scenes_json: str | dict, avatar_dir: str = "assets/avatar",
                 out_path: str = "outputs/video.mp4", style: Style | None = None,
                 sr: int = 24000) -> str:
    style = style or Style()
    spec = scenes_json if isinstance(scenes_json, dict) else json.load(open(scenes_json, encoding="utf-8"))
    scenes = [s for s in spec["scenes"] if s.get("verified", True)]  # las no verificadas no se publican
    if not scenes:
        raise ValueError("No hay escenas verificadas que renderizar.")

    # 1) audio completo + duración real de cada escena (manda el audio)
    tracks, durations = [], []
    for s in scenes:
        p = s.get("audio_path")
        if p and os.path.exists(p):
            a = _read_wav(p, sr)
        else:
            a = np.zeros(int(sr * float(s.get("duration_s", 4.0))), dtype=np.float32)
        a = np.concatenate([a, np.zeros(int(sr * 0.25), dtype=np.float32)])  # pequeña pausa entre escenas
        tracks.append(a)
        durations.append(len(a) / sr)
    audio = np.concatenate(tracks)
    env = _envelope(audio, sr, style.fps)

    fonts = {"small": _font(style, 40), "sub": _font(style, 72), "tiny": _font(style, 30)}
    avatars = {}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = tempfile.mkdtemp()
    silent, wav = os.path.join(tmp, "video.mp4"), os.path.join(tmp, "audio.wav")
    wavfile.write(wav, sr, (np.clip(audio, -1, 1) * 32767).astype(np.int16))

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{style.width}x{style.height}", "-r", str(style.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", silent],
        stdin=subprocess.PIPE)

    frame_idx, t_global = 0, 0.0
    lo, hi = style.mouth_thresholds
    for s, dur in zip(scenes, durations):
        speaker = s.get("speaker", "andrea")
        if speaker not in avatars:
            avatars[speaker] = _load_avatar(avatar_dir, speaker)
        av = avatars[speaker]
        bg = _load_background(s.get("background_path"), style)
        timings = _word_timings(s, dur)
        n = int(round(dur * style.fps))
        for i in range(n):
            t_scene = i / style.fps
            t = frame_idx / style.fps
            level = float(env[min(frame_idx, len(env) - 1)])
            mouth = av["open"] if level > hi else av["half"] if level > lo else av["closed"]

            frame = _ken_burns(bg, t_scene / max(dur, 1e-6), style).convert("RGBA")
            shade = Image.new("RGBA", frame.size, (0, 0, 0, 90))  # oscurecer para que se lean los textos
            frame.alpha_composite(shade)

            bub = _bubble(mouth, level, t, min(1.0, t / 0.45), style)
            frame.alpha_composite(bub, ((style.width - bub.width) // 2, style.bubble_y - bub.height // 2))
            _draw_texts(frame, s, spec, t_scene, dur, timings, fonts, style)

            ffmpeg.stdin.write(frame.convert("RGB").tobytes())
            frame_idx += 1
    ffmpeg.stdin.close()
    ffmpeg.wait()

    # 2) unir vídeo + audio
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", wav,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", out_path], check=True)
    return out_path
