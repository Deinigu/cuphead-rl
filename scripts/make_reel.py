"""Highlight reel of training-run milestone videos, for the YouTube edit.

    python -m uv run python scripts/make_reel.py runs/ppo_curriculum runs/ppo_baseline \
        -o videos/reel.mp4 [--max-clip-seconds 20] [--only-milestones] [--names "Curriculum,Plain PPO"]

    python -m uv run python scripts/make_reel.py --side-by-side runs/ppo_baseline runs/ppo_curriculum \
        -o videos/versus.mp4 [--steps 2e6,5e6,10e6]

Reads ``<run>/videos/index.jsonl`` (one line per milestone video, written by
``cuphead_rl.train.callbacks``). For every run, in the order given, and every indexed video
in step order: a title card in the sim's 1930s look (cream paper, ink serif, film grain) with
the run, the step count, the milestone in plain words and the eval numbers, then the clip
with a small corner label. Clips longer than ``--max-clip-seconds`` keep their END when the
episode was a win (so the knockout is shown), else their beginning.

``--side-by-side A B`` instead pairs the two runs' videos at comparable steps (nearest step
count within ``--step-tolerance``, or the ones nearest to ``--steps``) and shows them next to
each other at 640x360 with labels.

Output: H.264 (libx264, CRF 20), 1280x720 @ 60 fps, through imageio-ffmpeg. Missing or
unreadable clips (e.g. a video that training is still writing) are skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import imageio_ffmpeg
import numpy as np

W, H, FPS = 1280, 720, 60


# ------------------------------------------------------------------------------ index
@dataclass
class Clip:
    run: str  # display name
    path: Path
    steps: int
    tags: list[str]
    eval: dict
    episode: dict

    @property
    def won(self) -> bool:
        return bool(self.episode.get("won"))


def pretty_run(name: str) -> str:
    """``ppo_curriculum_hit8`` -> ``PPO curriculum hit8``."""
    words = name.replace("-", "_").split("_")
    return " ".join(w.upper() if w.lower() in ("ppo", "rl", "sim") else w for w in words if w)


def load_run(run_dir: Path, name: str | None = None, only_milestones: bool = False) -> list[Clip]:
    index = run_dir / "videos" / "index.jsonl"
    if not index.exists():
        warnings.warn(f"{run_dir}: no videos/index.jsonl, skipping the run")
        return []
    clips = []
    for n, line in enumerate(index.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            warnings.warn(f"{index}:{n}: unreadable line, skipped")  # e.g. being written right now
            continue
        tags = list(e.get("tags", []))
        if only_milestones and not [t for t in tags if t != "periodic"]:
            continue
        clips.append(Clip(name or pretty_run(run_dir.name), run_dir / "videos" / e["file"], int(e["steps"]),
                          tags, e.get("eval", {}), e.get("episode", {})))
    return sorted(clips, key=lambda c: c.steps)


# ------------------------------------------------------------------------------ wording
def steps_text(steps: int) -> str:
    if steps < 1000:
        return "untrained" if steps < 100 else f"{steps} steps"
    if steps < 1_000_000:
        return f"{steps / 1e3:.0f}k steps"
    return f"{steps / 1e6:.2f}M steps"


def tag_text(tag: str) -> str:
    """Milestone tag (callbacks._milestones) -> words for the audience."""
    fixed = {
        "untrained": "The untrained agent",
        "periodic": "Progress check",
        "first_phase2": "First time reaching phase 2",
        "first_phase3": "First time reaching phase 3",
        "first_win": "First win!",
    }
    if tag in fixed:
        return fixed[tag]
    kind, _, num = tag.rpartition("_")
    if num.isdigit():
        if kind == "damage":
            return f"Boss damage {num} %"
        if kind == "winrate":
            return f"Win rate {num} %"
    return tag.replace("_", " ").capitalize()


def headline_tags(tags: list[str]) -> list[str]:
    """Plain-word milestones, most exciting first; 'Progress check' only if nothing else."""
    order = ["first_win", "winrate", "first_phase3", "first_phase2", "damage", "untrained"]

    def rank(t: str) -> int:
        return next((i for i, p in enumerate(order) if t.startswith(p)), len(order))

    real = sorted((t for t in tags if t != "periodic"), key=rank)
    return [tag_text(t) for t in (real or tags or ["periodic"])]


def eval_text(ev: dict) -> str:
    parts = []
    if "win_rate" in ev:
        parts.append(f"Win rate {100 * ev['win_rate']:.0f} %")
    if "boss_damage" in ev:
        parts.append(f"Boss damage {100 * ev['boss_damage']:.0f} %")
    if "mean_phase" in ev:
        parts.append(f"Mean phase {ev['mean_phase'] + 1:.1f}")
    n = ev.get("episodes")
    return "   ·   ".join(parts) + (f"   (over {n} test fights)" if n else "")


def episode_text(ep: dict) -> str:
    if not ep:
        return ""
    t = ep.get("time")
    tt = f" in {t:.1f} s" if t is not None else ""
    if ep.get("won"):
        return f"This fight: KNOCKOUT{tt}"
    phase = int(ep.get("phase", 0)) + 1
    hp = ep.get("boss_hp_frac")
    hp_t = f", Goopy at {100 * hp:.0f} % HP" if hp is not None else ""
    return f"This fight: lost in phase {phase}{hp_t}"


# ------------------------------------------------------------------------------ drawing
class Painter:
    """Title cards and labels in the sim renderer's look (its colours, fonts and grain)."""

    def __init__(self) -> None:
        import os

        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        from cuphead_rl.sim import render as R

        self.pg, self.R = pygame, R
        self.r = R.Renderer()  # fonts, vignette and grain frames
        self.paper = self._paper()

    def _paper(self):
        pg, R = self.pg, self.R
        s = pg.Surface((W, H))
        s.fill(R.PAPER)
        rng = np.random.default_rng(7)
        # Soft blotches like aged paper.
        for _ in range(40):
            x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
            rad = int(rng.integers(60, 220))
            blot = pg.Surface((2 * rad, 2 * rad), pg.SRCALPHA)
            pg.draw.circle(blot, (*R.SEPIA, int(rng.integers(6, 16))), (rad, rad), rad)
            s.blit(blot, (x - rad, y - rad))
        return s

    def text(self, s: str, size: int, color=None, bold: bool = True):
        return self.r.font(size, bold).render(s, True, color or self.R.INK)

    def fit(self, s: str, max_w: int, size: int, color=None, bold: bool = True):
        while size > 12:
            surf = self.text(s, size, color, bold)
            if surf.get_width() <= max_w:
                return surf
            size -= 2
        return self.text(s, size, color, bold)

    def blit_center(self, surf, dst, y: int) -> None:
        dst.blit(surf, (W // 2 - surf.get_width() // 2, y))

    def card(self, lines: list[tuple[str, int, str]]) -> object:
        """A title card. ``lines`` = (text, size, colour key: ink/red/sepia), stacked."""
        pg, R = self.pg, self.R
        s = self.paper.copy()
        frame = pg.Rect(48, 40, W - 96, H - 80)
        pg.draw.rect(s, R.INK, frame, 6, border_radius=26)
        pg.draw.rect(s, R.INK, frame.inflate(-26, -26), 2, border_radius=18)
        for cx, cy in (frame.topleft, frame.topright, frame.bottomleft, frame.bottomright):
            pg.draw.circle(s, R.PAPER, (cx, cy), 22)
            pg.draw.circle(s, R.INK, (cx, cy), 22, 4)
            pg.draw.circle(s, R.RED, (cx, cy), 9)
        colors = {"ink": R.INK, "red": R.RED, "sepia": (120, 90, 60)}
        surfs = [self.fit(t, frame.w - 140, size, colors[c]) if t else None for t, size, c in lines]
        gap = 14
        total = sum((x.get_height() if x else 20) for x in surfs) + gap * (len(surfs) - 1)
        y = H // 2 - total // 2
        for x in surfs:
            if x is None:
                y += 20 + gap
                continue
            self.blit_center(x, s, y)
            y += x.get_height() + gap
        s.blit(self.r.vignette, (0, 0))
        return s

    def frames_of(self, surf, seconds: float) -> Iterator[bytes]:
        """``seconds`` of a card with flickering film grain and a short fade in/out."""
        pg = self.pg
        n = max(1, round(seconds * FPS))
        fade = min(12, n // 4)
        base = surf.copy()
        for i in range(n):
            f = base.copy()
            f.blit(self.r.grain[(i // 2) % len(self.r.grain)], (0, 0))
            # Gate weave: the whole card shifts by a pixel now and then, like a projector.
            dy = (1 if (i // 5) % 7 == 3 else 0)
            out = pg.Surface((W, H))
            out.fill(self.R.PAPER)
            out.blit(f, (0, dy))
            a = pg.surfarray.array3d(out).swapaxes(0, 1)
            k = min(1.0, (i + 1) / fade, (n - i) / fade) if fade else 1.0
            if k < 1.0:
                a = (a * k).astype(np.uint8)
            yield np.ascontiguousarray(a).tobytes()

    def label(self, text: str, size: int = 18, max_w: int = 476) -> np.ndarray:
        """RGBA array of a small cream tag with an ink border (narrow enough to stay clear
        of the boss HP counter under the clip's health bar)."""
        pg, R = self.pg, self.R
        t = self.fit(text, max_w, size)
        w, h = t.get_width() + 24, t.get_height() + 10
        s = pg.Surface((w, h), pg.SRCALPHA)
        pg.draw.rect(s, (*R.CREAM, 235), (0, 0, w, h), border_radius=8)
        pg.draw.rect(s, (*R.INK, 255), (0, 0, w, h), 2, border_radius=8)
        s.blit(t, (12, 5))
        rgb = pg.surfarray.array3d(s).swapaxes(0, 1)
        alpha = pg.surfarray.array_alpha(s).swapaxes(0, 1)
        return np.dstack([rgb, alpha])


def overlay(frame: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    """Alpha-blend ``rgba`` onto ``frame`` (H, W, 3) at (x, y), in place, clipped."""
    h, w = rgba.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = src[..., 3:4].astype(np.float32) / 255.0
    dst = frame[y0:y1, x0:x1]
    dst[:] = (src[..., :3] * a + dst * (1.0 - a)).astype(np.uint8)


# ------------------------------------------------------------------------------ video io
def probe(path: Path) -> float | None:
    """Duration in seconds, or None if the file is missing/unreadable."""
    if not path.exists():
        warnings.warn(f"missing clip {path}, skipped")
        return None
    try:
        gen = imageio_ffmpeg.read_frames(str(path))
        meta = next(gen)
        gen.close()
    except Exception as exc:  # noqa: BLE001 - ffmpeg reports unreadable files many ways
        warnings.warn(f"unreadable clip {path} ({str(exc).splitlines()[0]}), skipped")
        return None
    dur = meta.get("duration")
    if not dur or not math.isfinite(dur):
        warnings.warn(f"clip {path} has no duration, skipped")
        return None
    return float(dur)


def clip_window(duration: float, max_seconds: float | None, keep_end: bool) -> tuple[float, float]:
    """(start, length) of the part to show."""
    if max_seconds is None or duration <= max_seconds:
        return 0.0, duration
    return (duration - max_seconds if keep_end else 0.0), max_seconds


def read_clip(path: Path, start: float, length: float, size: tuple[int, int]) -> Iterator[np.ndarray]:
    """Exactly ``round(length * 60)`` frames of ``path`` from ``start``, scaled to ``size``
    (ffmpeg's own trim can be a frame short; a short clip holds its last frame)."""
    w, h = size
    n = round(length * FPS)
    gen = imageio_ffmpeg.read_frames(
        str(path), input_params=["-ss", f"{start:.3f}"],
        output_params=["-t", f"{length + 0.25:.3f}", "-vf", f"scale={w}:{h}:flags=area,fps={FPS}"])
    log = logging.getLogger("imageio_ffmpeg")
    level = log.level
    log.setLevel(logging.ERROR)  # "frame size for reading differs": we scale on purpose
    last = None
    try:
        next(gen)  # metadata
        for _, raw in zip(range(n), gen):
            last = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
            n -= 1
            yield last.copy()
    finally:
        gen.close()
        log.setLevel(level)
    for _ in range(n if last is not None else 0):
        yield last.copy()


class Writer:
    def __init__(self, path: Path, crf: int = 20, preset: str = "medium") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.gen = imageio_ffmpeg.write_frames(
            str(path), (W, H), fps=FPS, codec="libx264", quality=None, pix_fmt_out="yuv420p",
            macro_block_size=8, output_params=["-crf", str(crf), "-preset", preset])
        self.gen.send(None)
        self.frames = 0

    def write(self, frame: bytes | np.ndarray) -> None:
        self.gen.send(frame if isinstance(frame, bytes) else np.ascontiguousarray(frame).tobytes())
        self.frames += 1

    def close(self) -> None:
        self.gen.close()


# ------------------------------------------------------------------------------ reels
def title_lines(c: Clip) -> list[tuple[str, int, str]]:
    tags = headline_tags(c.tags)
    lines = [(c.run, 34, "sepia"), (steps_text(c.steps), 76, "red"), ("", 0, "ink")]
    lines += [(t, 44 if i == 0 else 32, "ink") for i, t in enumerate(tags[:3])]
    lines += [("", 0, "ink"), (eval_text(c.eval), 24, "sepia")]
    ep = episode_text(c.episode)
    if ep:
        lines.append((ep, 24, "sepia"))
    return lines


def make_reel(runs: list[Path], out: Path, max_clip: float | None = 20.0, only_milestones: bool = False,
              names: list[str] | None = None, card_seconds: float = 2.5, crf: int = 20,
              preset: str = "medium") -> dict:
    painter = Painter()
    writer = Writer(out, crf, preset)
    shown, skipped = 0, 0
    try:
        for i, run in enumerate(runs):
            name = names[i] if names and i < len(names) else None
            for c in load_run(run, name, only_milestones):
                dur = probe(c.path)
                if dur is None:
                    skipped += 1
                    continue
                for fr in painter.frames_of(painter.card(title_lines(c)), card_seconds):
                    writer.write(fr)
                start, length = clip_window(dur, max_clip, keep_end=c.won)
                words = [c.run, steps_text(c.steps)]
                if "untrained" not in c.tags:
                    words.append(headline_tags(c.tags)[0])
                tag = painter.label("  ·  ".join(words))
                for fr in read_clip(c.path, start, length, (W, H)):
                    overlay(fr, tag, 24, 62)
                    writer.write(fr)
                shown += 1
    finally:
        writer.close()
    return {"clips": shown, "skipped": skipped, "frames": writer.frames, "seconds": writer.frames / FPS}


def pair_clips(a: list[Clip], b: list[Clip], targets: list[float] | None, tolerance: float) -> list[tuple[Clip, Clip]]:
    """Clips of A and B at comparable step counts."""
    pairs = []
    if targets:
        for t in targets:
            ca = min(a, key=lambda c: abs(c.steps - t), default=None)
            cb = min(b, key=lambda c: abs(c.steps - t), default=None)
            if ca and cb:
                pairs.append((ca, cb))
    else:
        for ca in a:
            cb = min(b, key=lambda c: abs(c.steps - ca.steps), default=None)
            if cb and abs(cb.steps - ca.steps) <= tolerance * max(ca.steps, cb.steps, 1):
                pairs.append((ca, cb))
    seen, out = set(), []
    for ca, cb in pairs:  # drop repeats (two targets landing on the same videos)
        if (ca.path, cb.path) not in seen:
            seen.add((ca.path, cb.path))
            out.append((ca, cb))
    return out


def make_versus(run_a: Path, run_b: Path, out: Path, max_clip: float | None = 20.0,
                targets: list[float] | None = None, tolerance: float = 0.1, names: list[str] | None = None,
                only_milestones: bool = False, card_seconds: float = 2.5, crf: int = 20,
                preset: str = "medium") -> dict:
    painter = Painter()
    pg, R = painter.pg, painter.R
    a = load_run(run_a, names[0] if names else None, only_milestones)
    b = load_run(run_b, names[1] if names and len(names) > 1 else None, only_milestones)
    writer = Writer(out, crf, preset)
    shown, skipped = 0, 0
    half = (640, 360)
    top = (H - half[1]) // 2
    try:
        for ca, cb in pair_clips(a, b, targets, tolerance):
            durs = [probe(ca.path), probe(cb.path)]
            if None in durs:
                skipped += 1
                continue
            steps = steps_text(round((ca.steps + cb.steps) / 2))
            card = painter.card([("Side by side", 34, "sepia"), (f"around {steps}", 70, "red"), ("", 0, "ink"),
                                 (f"{ca.run}  vs  {cb.run}", 40, "ink"), ("", 0, "ink"),
                                 (f"{ca.run}: {eval_text(ca.eval)}", 22, "sepia"),
                                 (f"{cb.run}: {eval_text(cb.eval)}", 22, "sepia")])
            for fr in painter.frames_of(card, card_seconds):
                writer.write(fr)
            # Background with both captions; the clips play in the two slots.
            bg = painter.paper.copy()
            for k, c in enumerate((ca, cb)):
                x0 = k * 640
                head = painter.fit(f"{c.run}  ·  {steps_text(c.steps)}", 600, 30)
                bg.blit(head, (x0 + 320 - head.get_width() // 2, top - 16 - head.get_height()))
                sub = painter.fit(headline_tags(c.tags)[0] + "   ·   " + episode_text(c.episode).replace("This fight: ", ""),
                                  600, 20, (120, 90, 60))
                bg.blit(sub, (x0 + 320 - sub.get_width() // 2, top + half[1] + 14))
                pg.draw.rect(bg, R.INK, (x0 + 1, top - 3, 638, half[1] + 6), 3)
            bg.blit(painter.r.vignette, (0, 0))
            bg_arr = pg.surfarray.array3d(bg).swapaxes(0, 1).copy()
            wins = [clip_window(d, max_clip, keep_end=c.won) for d, c in zip(durs, (ca, cb))]
            n = round(max(length for _, length in wins) * FPS)
            gens = [read_clip(c.path, s, length, half) for c, (s, length) in zip((ca, cb), wins)]
            last: list[np.ndarray | None] = [None, None]
            for _ in range(n):
                fr = bg_arr.copy()
                for k, g in enumerate(gens):
                    nxt = next(g, None)
                    if nxt is not None:
                        last[k] = nxt
                    if last[k] is not None:  # the shorter clip holds its final frame
                        fr[top:top + half[1], k * 640:(k + 1) * 640] = last[k]
                writer.write(fr)
            for g in gens:
                g.close()
            shown += 1
    finally:
        writer.close()
    return {"pairs": shown, "skipped": skipped, "frames": writer.frames, "seconds": writer.frames / FPS}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", type=Path, help="run directories, in reel order")
    ap.add_argument("-o", "--out", type=Path, default=Path("videos/reel.mp4"))
    ap.add_argument("--max-clip-seconds", type=float, default=20.0, help="0 = full clips")
    ap.add_argument("--only-milestones", action="store_true", help="skip purely periodic videos")
    ap.add_argument("--names", help="comma-separated display names, one per run")
    ap.add_argument("--side-by-side", nargs=2, type=Path, metavar=("RUN_A", "RUN_B"))
    ap.add_argument("--steps", help="side-by-side: comma-separated step counts to compare, e.g. 2e6,5e6,10e6")
    ap.add_argument("--step-tolerance", type=float, default=0.1,
                    help="side-by-side without --steps: max relative step difference of a pair")
    ap.add_argument("--card-seconds", type=float, default=2.5)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--preset", default="medium", help="x264 preset (ultrafast for drafts)")
    args = ap.parse_args(argv)
    max_clip = args.max_clip_seconds or None
    names = [n.strip() for n in args.names.split(",")] if args.names else None
    warnings.simplefilter("always", UserWarning)
    warnings.formatwarning = lambda msg, *a, **k: f"warning: {msg}\n"
    if args.side_by_side:
        targets = [float(s) for s in args.steps.split(",")] if args.steps else None
        res = make_versus(*args.side_by_side, args.out, max_clip, targets, args.step_tolerance, names,
                          args.only_milestones, args.card_seconds, args.crf, args.preset)
    elif args.runs:
        res = make_reel(args.runs, args.out, max_clip, args.only_milestones, names, args.card_seconds,
                        args.crf, args.preset)
    else:
        ap.error("give run directories, or --side-by-side RUN_A RUN_B")
    print(f"wrote {args.out}: {res}")
    if not res.get("frames"):
        sys.exit("nothing to show")


if __name__ == "__main__":
    main()
