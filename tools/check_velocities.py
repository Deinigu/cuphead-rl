"""Check that the bridge's velocities are per-tick, whatever the lockstep ``frames``.

    python -m uv run python tools/check_velocities.py [--out data/check_velocities.json]

Needs the game running with the plugin. Plays one tick-aligned input script twice from the same
seeded lockstep reset (no-hit): with frames=1 and with frames=4. The script idles, holds jump for
20 ticks, then idles through small Goopy's first hop. The start of a seeded fight is deterministic
for its first second or so, so both runs see the same motion. At every tick both runs report, the
velocities must agree. With per-tick velocities they do, while 4-tick averages lag behind. The jump
vy is also compared with the sim's (``SimConfig.player.jump_speed`` = 1223.1 while jump is held).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuphead_rl.game.client import BridgeClient  # noqa: E402
from cuphead_rl.spec import Action  # noqa: E402

IDLE_TICKS, JUMP_TICKS, TOTAL_TICKS = 32, 20, 88  # multiples of 4


def script(tick: int) -> Action:
    return Action(jump=IDLE_TICKS <= tick < IDLE_TICKS + JUMP_TICKS)


def rollout(c: BridgeClient, frames: int, seed: int) -> dict[int, dict]:
    c.configure(lockstep=True, frames=frames, time_scale=1.0, stream_every=0, debug=False)
    c.reset(seed=seed, character="mugman")
    c.invincible(True)
    out = {}
    for k in range(0, TOTAL_TICKS, frames):
        m = c.step_lockstep(script(k), frames)
        p, b = m["player"], m["boss"]
        out[k + frames] = {"p": (p["x"], p["y"], p["vx"], p["vy"]), "b": (b["x"], b["y"], b["vx"], b["vy"], b["state"])}
    c.invincible(False)
    return out


def sim_jump_vy() -> list[float]:
    from cuphead_rl.sim.env import GoopyEnv

    env = GoopyEnv(frame_skip=1)
    env.reset(seed=0)
    vys = []
    for k in range(TOTAL_TICKS):
        a = script(k)
        env.step(np.array([1, int(a.jump), 0, 0, 0, 1]))
        vys.append(round(float(env.state.player.vy), 1))
    return vys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    c = BridgeClient(timeout=60)
    c.connect(wait=120)
    try:
        r1 = rollout(c, 1, args.seed)
        r4 = rollout(c, 4, args.seed)
    finally:
        c.close()
    sim = sim_jump_vy()

    # Compare each entity only while its positions agree in both runs (the boss sometimes picks a
    # different hop in the two runs, the known long-rollout chaos; the player's motion is scripted).
    rows, worst, matched = [], {"p": 0.0, "b": 0.0}, {"p": 0, "b": 0}
    print(" tick | player y  vy f1     f4    sim | boss y  vy f1     f4  (state)")
    for t in sorted(r4):
        a, b = r1[t], r4[t]
        row = {"tick": t, "sim_vy": sim[t - 1]}
        for e, n in (("p", "player"), ("b", "boss")):
            if a[e][:2] == b[e][:2]:
                matched[e] += 1
                worst[e] = max(worst[e], abs(a[e][2] - b[e][2]), abs(a[e][3] - b[e][3]))
                row.update({f"{n}_y": a[e][1], f"{n}_vy_f1": a[e][3], f"{n}_vy_f4": b[e][3]})
        rows.append(row)
        f = lambda k: f"{row[k]:8.1f}" if k in row else "      --"  # noqa: E731
        print(f" {t:4d} | {f('player_y')} {f('player_vy_f1')} {f('player_vy_f4')} {sim[t - 1]:7.1f} | "
              f"{f('boss_y')} {f('boss_vy_f1')} {f('boss_vy_f4')}  ({a['b'][4]}/{b['b'][4]})")
    ok = matched["p"] >= 10 and matched["b"] >= 10 and worst["p"] < 1.0 and worst["b"] < 1.0
    print(f"\nmatched ticks: player {matched['p']}, boss {matched['b']}; max |v(frames=4) - v(frames=1)|: "
          f"player {worst['p']:.1f}, boss {worst['b']:.1f} u/s -> {'PASS' if ok else 'FAIL'}")
    held = [r["player_vy_f1"] for r in rows if IDLE_TICKS < r["tick"] <= IDLE_TICKS + 8 and r.get("player_y", 0) > 0]
    if held:
        print(f"player vy while jump held (frames=1): {min(held):.1f}..{max(held):.1f}  (sim {SIM_JUMP_SPEED})")
    if args.out:
        args.out.write_text(json.dumps({"seed": args.seed, "pass": ok, "matched_ticks": matched,
                                        "max_diff": worst, "rows": rows}, indent=1))
        print(f"-> {args.out}")


SIM_JUMP_SPEED = 1223.1

if __name__ == "__main__":
    main()
