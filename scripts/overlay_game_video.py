"""Overlay the agent's "thoughts" panel onto a real-game capture from scripts/play_game.py.

    python -m uv run python scripts/overlay_game_video.py videos/game_eval/episode_00.mp4 \
        [--log videos/game_eval/episode_00.jsonl] [-o out.mp4] [--title "Zero-shot | trained in sim | real Cuphead"]

The panel is drawn by the sim renderer itself (``Renderer._draw_panel``: reward, input glyphs,
value gauge, per-head probability bars, caption bar), at the same place and scale as in sim
videos, so sim and game footage look alike. The capture is scaled to 1280x720 and resampled
to 60 fps. Sync: video time ``t`` is wall time ``video_start_epoch + t`` (the capture's first
frame, see ``cuphead_rl.game.capture``). Each frame shows the latest decision whose ``wall`` time
is at or before it. Before the first decision the panel shows the title only, and after the last
one it holds the final values.

Output: H.264 (libx264, CRF 20, preset medium), yuv420p, 1280x720 @ 60 fps.
"""

from __future__ import annotations

import argparse
import bisect
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuphead_rl import spec  # noqa: E402
from cuphead_rl.sim import render  # noqa: E402  (sets SDL_VIDEODRIVER=dummy before pygame)

import pygame  # noqa: E402

W, H, FPS = render.W, render.H, 60
DEFAULT_TITLE = "Zero-shot | trained in sim | real Cuphead"


def load_log(path: Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    if not lines or "meta" not in lines[0]:
        raise ValueError(f"{path}: first line must be {{'meta': ...}} (written by scripts/play_game.py)")
    return lines[0]["meta"], lines[1:]


def overlay_video(video: Path, log: Path, out: Path, title: str, offset: float = 0.0) -> dict:
    meta, steps = load_log(log)
    start = meta.get("video_start_epoch")
    if start is None:
        raise ValueError(f"{log}: no video_start_epoch, cannot sync")
    walls = [s["wall"] - start + offset for s in steps]  # decision times in video seconds

    renderer = render.Renderer(character=meta.get("character", "mugman"))
    reader = imageio_ffmpeg.read_frames(
        str(video), pix_fmt="rgb24", output_params=["-vf", f"fps={FPS},scale={W}:{H}:flags=lanczos"])
    src_meta = next(reader)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(out), (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=8,
        output_params=["-crf", "20", "-preset", "medium"])
    writer.send(None)
    n = 0
    try:
        for raw in reader:
            t = n / FPS
            i = bisect.bisect_right(walls, t) - 1
            frame = pygame.image.frombuffer(raw, (W, H), "RGB")
            renderer.screen.blit(frame, (0, 0))
            if i >= 0:
                s = steps[i]
                ov = {"title": title, "value": s.get("value")}
                if s.get("probs"):
                    ov["probs"] = s["probs"]
                ov["phase"] = ("I", "II", "III")[int(s.get("phase", 0))]
                ov["boss hp"] = f"{100 * float(s.get('boss_hp_frac', 1.0)):.0f}%"
                renderer._draw_panel(spec.decode_action(s["action"]), float(s.get("return", 0.0)), ov)
            else:
                renderer._draw_panel(spec.Action(), 0.0, {"title": title})
            writer.send(np.ascontiguousarray(pygame.surfarray.array3d(renderer.screen).transpose(1, 0, 2)))
            n += 1
    finally:
        writer.close()
        reader.close()
        renderer.close()
    # Frame n of the output is source time n / FPS, so the source's audio track lines up as is.
    tmp = out.with_suffix(".mux.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(out), "-i", str(video),
                    "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    str(tmp)], check=True, stdin=subprocess.DEVNULL)
    tmp.replace(out)
    return {"out": str(out), "frames": n, "seconds": round(n / FPS, 2), "decisions": len(steps),
            "source_size": src_meta.get("source_size"), "first_decision_at": round(walls[0], 3) if walls else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--log", type=Path, help="default: the video's .jsonl sibling")
    ap.add_argument("-o", "--out", type=Path, help="default: <video>_overlay.mp4")
    ap.add_argument("--title", default=DEFAULT_TITLE)
    ap.add_argument("--offset", type=float, default=0.0, help="extra seconds added to decision times (manual sync tweak)")
    args = ap.parse_args()
    log = args.log or args.video.with_suffix(".jsonl")
    out = args.out or args.video.with_name(args.video.stem + "_overlay.mp4")
    print(json.dumps(overlay_video(args.video, log, out, args.title, args.offset), indent=2))


if __name__ == "__main__":
    main()
