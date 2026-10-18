"""Record per-tick game traces for sim-vs-game comparison into data/traces/*.jsonl.

    python -m uv run python tools/record_trace.py [--seconds 60]

Each file: line 1 = {"meta": ...}; then one JSON object per fixed tick (lockstep, 1 tick/step):
{"tick", "action": [move, jump, dash, shoot, ex, aim], "state": <spec.GameState fields as dicts>}.
Policies are deterministic functions of the state, so a simulator can replay either the
recorded actions open-loop or the same policy closed-loop.

  stand_shoot   stand at the spawn point and shoot straight ahead (debug no-hit, so the trace
                covers phase 1 -> 2 -> 3; standing shots cannot hurt the tombstone)
  dodge_shoot   stand left, shoot, jump when Goopy is within 250 u or a fist is out, aim up
                under the tombstone; normal damage (episode ends when the player dies)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cuphead_rl.game.client import BridgeClient
from cuphead_rl.spec import Action

OUT = Path(__file__).resolve().parents[1] / "data" / "traces"
KEEP_PLAYER = ("x", "y", "vx", "vy", "grounded", "dashing", "can_dash", "facing", "hp", "invuln", "super_meter")
KEEP_BOSS = ("phase", "state", "x", "y", "half_w", "half_h", "vx", "vy", "state_time", "facing", "hp",
             "max_hp", "phase_hp_frac", "vulnerable", "entity", "anim")


def r(v):
    return round(v, 2) if isinstance(v, float) else v


def compact(m: dict) -> dict:
    return {
        "time": r(m["time"]),
        "player": {k: r(m["player"][k]) for k in KEEP_PLAYER},
        "boss": {k: r(m["boss"][k]) for k in KEEP_BOSS},
        "hazards": [{k: r(h[k]) for k in ("kind", "x", "y", "half_w", "half_h", "vx", "vy", "parryable")} for h in m["hazards"]],
        "damage_dealt": r(m["damage_dealt"]), "won": m["won"], "lost": m["lost"], "parries": m["parries"],
    }


def stand_shoot(m: dict, i: int) -> Action:
    return Action(shoot=True)


def dodge_shoot(m: dict, i: int) -> Action:
    p, b = m["player"], m["boss"]
    dx = b["x"] - p["x"]
    fist = any(h["kind"] == 1 for h in m["hazards"])
    if b["phase"] == 2:  # tombstone: stay under it and aim up, dodge sideways during the slam windup
        if b["state"] in (7, 8):
            return Action(move=-1 if dx > 0 else 1, shoot=True)
        return Action(move=0 if abs(dx) < 40 else (1 if dx > 0 else -1), shoot=True, aim=1 if abs(dx) < 40 else 0)
    near = abs(dx) < 250 + b["half_w"] or fist
    hold_x = -520.0
    move = 0 if abs(p["x"] - hold_x) < 15 else (1 if p["x"] < hold_x else -1)
    return Action(move=move, jump=near and p["grounded"] and i % 2 == 0, shoot=True)


def record(c: BridgeClient, name: str, policy, seconds: float, seed: int, no_hit: bool) -> Path:
    c.configure(lockstep=True, frames=1, time_scale=1.0, stream_every=0, debug=False)
    m = c.reset(seed=seed)
    c.invincible(no_hit)
    m = c.last_state
    path = OUT / f"{name}.jsonl"
    with path.open("w") as f:
        f.write(json.dumps({"meta": {
            "policy": name, "seed": seed, "no_hit": no_hit, "ticks_per_line": 1, "fixed_dt": 1 / 60,
            "coords": "spec (x = unity x, y = unity y + 250)", "difficulty": "Regular",
            "note": "seeded lockstep reset; runs are frame-identical for the first several seconds only (see docs/GAME_BRIDGE.md)",
        }}) + "\n")
        for i in range(int(seconds * 60)):
            a = policy(m, i)
            m = c.step_lockstep(a, 1)
            if not m.get("in_fight"):
                break
            f.write(json.dumps({"tick": i, "action": [a.move, int(a.jump), int(a.dash), int(a.shoot), int(a.ex), a.aim],
                                "state": compact(m)}, separators=(",", ":")) + "\n")
            if m["won"] or m["lost"]:
                break
    c.invincible(False)
    print(f"{path.name}: {i + 1} ticks, {path.stat().st_size / 1e6:.2f} MB, final phase {m.get('boss', {}).get('phase')}, "
          f"won={m.get('won')} lost={m.get('lost')}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    c = BridgeClient(timeout=30)
    c.connect(wait=90)
    try:
        record(c, "stand_shoot", stand_shoot, args.seconds, args.seed, no_hit=True)
        record(c, "dodge_shoot", dodge_shoot, args.seconds, args.seed, no_hit=False)
    finally:
        c.configure(lockstep=False, time_scale=1.0, stream_every=0)
        c.close()


if __name__ == "__main__":
    main()
