"""Render the English narration with edge-tts, one audio file per segment.

Free, no API key, no paid service. Produces segment durations so the visual
scenes can be timed to the voice rather than the other way round.

    python narrate.py --out E:\\AIMS-sprint\\video\\audio
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import wave

import edge_tts

VOICE = "en-GB-RyanNeural"      # natural British male, clear on numbers
RATE = "-4%"                    # a touch slower: the script is dense with figures
PITCH = "+0Hz"

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "script.md")


def segments(path: str) -> list[tuple[str, str]]:
    """Return (title, narration) for each '## n — title' block."""
    text = open(path, encoding="utf-8").read()
    out = []
    for block in re.split(r"\n## ", text)[1:]:
        head, _, body = block.partition("\n")
        title = re.sub(r"\s*\(~\d+\s*s\)\s*$", "", head).strip()
        body = body.strip()
        if body:
            out.append((title, " ".join(body.split())))
    return out


async def speak(text: str, path: str):
    comm = edge_tts.Communicate(text, VOICE, rate=RATE, pitch=PITCH)
    await comm.save(path)


def duration_mp3(path: str) -> float:
    """Decode with the bundled ffmpeg to measure the real duration."""
    import subprocess
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    wav = path + ".probe.wav"
    subprocess.run([exe, "-y", "-loglevel", "error", "-i", path, wav], check=True)
    with wave.open(wav) as w:
        d = w.getnframes() / w.getframerate()
    os.remove(wav)
    return d


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    segs = segments(SCRIPT)
    print(f"{len(segs)} segmentos")
    meta = []
    for i, (title, text) in enumerate(segs, 1):
        path = os.path.join(a.out, f"seg{i:02d}.mp3")
        await speak(text, path)
        d = duration_mp3(path)
        meta.append({"index": i, "title": title, "file": os.path.basename(path),
                     "seconds": round(d, 3), "words": len(text.split()), "text": text})
        print(f"  {i:>2}. {title:<34} {d:6.1f}s  {len(text.split()):>3} palabras")
    total = sum(m["seconds"] for m in meta)
    print(f"\nTOTAL narracion: {total:.1f}s  ({total/60:.2f} min)")
    json.dump({"voice": VOICE, "rate": RATE, "total_seconds": round(total, 2),
               "segments": meta},
              open(os.path.join(a.out, "narration.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
