"""Turn the bridge's live hitbox timelines into per-tick tracks for the simulator.

    python -m uv run python scripts/extract_live_tracks.py [--json data/live_hitboxes.json]

Writes ``src/cuphead_rl/sim/live_tracks.py``. Each track is a tuple of frames (one per 60 Hz
tick from the start of the move); positions are relative to the attack direction so the
sim can mirror them:

* punch frames: ``(windup, body, fists)`` with ``body = (dx_fwd, cy, r)`` and fists
  ``(dx_fwd, cy, half_w, half_h, round)``; ``cy`` is the absolute spec y (slimes punch
  on the ground).
* tomb smash frames: ``(windup, dx, cy, r)``; ``cy`` absolute for a tomb at rest.

The timelines stop a few ticks into the end animation; the remaining recovery back to the
rest pose is appended as a short linear blend (``RETRACT_TICKS``, estimated).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "cuphead_rl" / "sim" / "live_tracks.py"
RETRACT_TICKS = 12  # est: rest of punch_end not covered by the timeline


def _r(v: float) -> float:
    return round(float(v), 1)


def punch_track(frames: list[dict], rest_r: float) -> tuple[list, int]:
    x0 = frames[0]["body"][0]
    # Attack direction: where the furthest hitbox ends up relative to the start.
    far = max((f for fr in frames for f in ([fr["body"]] + list(fr["fist"] or []))),
              key=lambda b: abs(b[0] - x0))
    d = 1.0 if far[0] >= x0 else -1.0
    out = []
    for fr in frames:
        bx, by, bhw, _ = fr["body"]
        fists = tuple((_r((fx - x0) * d), _r(fy), _r(fhw), _r(fhh), abs(fhw - fhh) < 1e-6)
                      for fx, fy, fhw, fhh in (fr["fist"] or []) if fhw > 1.0)
        out.append((fr["state"] == "PUNCH_WINDUP" and fr["t"] < _first_punch(frames),
                    (_r((bx - x0) * d), _r(by), _r(bhw)), fists))
    # Blend the last frame back to the rest pose.
    last = out[-1]
    rest_cy = frames[0]["body"][1]
    for i in range(1, RETRACT_TICKS + 1):
        k = 1.0 - i / RETRACT_TICKS
        bdx, bcy, br = last[1]
        body = (_r(bdx * k), _r(rest_cy + (bcy - rest_cy) * k), _r(rest_r + (br - rest_r) * k))
        fists = tuple((_r(f[0] * k), f[1], _r(f[2] * k), _r(f[3] * k), f[4]) for f in last[2]) if k > 0.3 else ()
        out.append((False, body, fists))
    return out, _first_punch(frames)


def _first_punch(frames: list[dict]) -> int:
    return next(fr["t"] for fr in frames if fr["state"] == "PUNCH")


def tomb_track(frames: list[dict]) -> tuple[list, int]:
    x0 = frames[0]["body"][0]
    windup = next(fr["t"] for fr in frames if fr["state"] == "TOMB_SLAM")
    out = [(fr["t"] < windup, _r(fr["body"][0] - x0), _r(fr["body"][1]), _r(fr["body"][2])) for fr in frames]
    return out, windup


def render(data: dict) -> str:
    """Source of live_tracks.py for a parsed live_hitboxes.json."""
    tl = data["timelines"]
    small, small_w = punch_track(tl["small.punch"], 80.0)
    big, big_w = punch_track(tl["big.punch"], 160.0)
    tomb, tomb_w = tomb_track(tl["tomb.slam"])
    lines = [
        '"""Per-tick hitbox tracks measured live in Cuphead (generated: do not edit).',
        "",
        "Source: data/live_hitboxes.json timelines, via scripts/extract_live_tracks.py.",
        '"""',
        "",
        "# fmt: off",
        f"SMALL_PUNCH_WINDUP_TICKS = {small_w}",
        f"BIG_PUNCH_WINDUP_TICKS = {big_w}",
        f"TOMB_SMASH_WINDUP_TICKS = {tomb_w}",
        "",
        "# (windup, (dx_fwd, cy, r), ((dx_fwd, cy, half_w, half_h, round), ...)) per tick",
        f"SMALL_PUNCH = {tuple(small)!r}",
        "",
        f"BIG_PUNCH = {tuple(big)!r}",
        "",
        "# (windup, dx, cy, r) per tick",
        f"TOMB_SMASH = {tuple(tomb)!r}",
        "# fmt: on",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "data" / "live_hitboxes.json"))
    args = ap.parse_args()
    OUT.write_text(render(json.loads(Path(args.json).read_text())))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
