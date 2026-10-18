"""Sample live world-space hitboxes from the running game into data/live_hitboxes.json.

    python -m uv run python tools/sample_hitboxes.py [--seconds-per-phase 25]

Lockstep, 1 tick per step, player made invincible (debug) so the fight can be observed for
long enough. Phases 2/3 are reached with the bridge's `damage` command. All coordinates are spec
coordinates (x unchanged, y above Level.Ground = -250). Per boss animation clip the script
aggregates every enabled collider; it also records a full punch and a full slam timeline, the
player's hurtbox (standing / ducking / airborne) and peashooter bullet heights.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from cuphead_rl.game.client import BridgeClient
from cuphead_rl.spec import Action, BossMove

OUT = Path(__file__).resolve().parents[1] / "data" / "live_hitboxes.json"


def r(v: float) -> float:
    return round(float(v), 1)


class Agg:
    """min/max aggregation of collider geometry keyed by (entity, anim, collider)."""

    def __init__(self) -> None:
        self.d: dict = defaultdict(lambda: {"n": 0})

    def add(self, key: str, col: dict) -> None:
        e = self.d[key]
        e["n"] += 1
        dx = (col["x"] - col["tf_x"]) * (-1 if col["ent_sx"] > 0 else 1)  # + = in front of the boss
        vals = {
            "half_w": col["half_w"],
            "half_h": col["half_h"],
            "center_dx_forward": dx,
            "center_dy_from_entity": col["y"] - col["ent_y"],
            "center_y": col["y"],
            "bottom_y": col["y"] - col["half_h"],
            "top_y": col["y"] + col["half_h"],
        }
        for k, v in vals.items():
            lo, hi = e.get(k, (v, v))
            e[k] = (min(lo, v), max(hi, v))
        e["type"] = col["type"]
        if col["radius"]:
            e["radius"] = col["radius"]

    def out(self) -> dict:
        res = {}
        for k, e in sorted(self.d.items()):
            res[k] = {kk: ([r(v[0]), r(v[1])] if isinstance(v, tuple) else v) for kk, v in e.items()}
        return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds-per-phase", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    ticks_per_phase = int(args.seconds_per_phase * 60)

    c = BridgeClient(timeout=20)
    c.connect(wait=90)
    c.configure(lockstep=False, stream_every=0, debug=True)
    c.reset(seed=args.seed)
    c.configure(lockstep=True, frames=1, debug=True)
    c.wait_for("state")
    c.invincible(True)
    try:
        step = lambda a=Action(), y=None: c.step_lockstep(a, 1, move_y=y)  # noqa: E731

        out: dict = {"_meta": {
            "units": "spec coordinates: x = unity x (arena centre 0), y = unity y + 250 (Level.Ground); half sizes are AABB half extents",
            "source": "live Collider2D.bounds via CupheadRLBridge debug state, lockstep 1 tick/sample",
            "seed": args.seed,
        }}

        # ---------------------------------------------------------------- player hurtbox
        m = step()
        for _ in range(5):
            m = step()
        p = m["player"]
        hb = lambda p: {"center_dy_from_feet": r(p["hb_y"] - p["y"]), "half_w": r(p["hb_half_w"]), "half_h": r(p["hb_half_h"])}  # noqa: E731
        out["player.hurtbox.standing"] = hb(p)
        for _ in range(8):
            m = step(y=-1)
        out["player.hurtbox.ducking"] = hb(m["player"]) | {"ducking_flag": m["player"]["ducking"]}
        m = step()
        air = []
        for i in range(30):
            m = step(Action(jump=i < 12))
            if not m["player"]["grounded"]:
                air.append(hb(m["player"]))
        out["player.hurtbox.airborne"] = air[len(air) // 2] if air else None

        # ---------------------------------------------------------------- bullets
        ys_stand, ys_duck = set(), set()
        for i in range(60):
            m = step(Action(shoot=True))
            for b in m.get("bullets", []):
                if b["type"] == "BasicProjectile" or "Peashot" in b["type"] or b["type"].endswith("Projectile"):
                    ys_stand.add(r(b["y"]))
        for i in range(60):
            m = step(Action(shoot=True), y=-1)
            for b in m.get("bullets", []):
                ys_duck.add(r(b["y"]))
        sample = m.get("bullets", [])[:1]
        out["peashooter.bullet_y.standing"] = sorted(ys_stand)
        out["peashooter.bullet_y.ducking"] = sorted(ys_duck)
        out["peashooter.bullet_half_size"] = [r(sample[0]["half_w"]), r(sample[0]["half_h"])] if sample else None

        # ---------------------------------------------------------------- boss phases
        agg = Agg()
        timelines: dict = {}

        def observe(label: str, ticks: int, policy) -> None:
            punch, slam, recording = [], [], None
            for i in range(ticks):
                m = step(policy(i))
                if not m.get("in_fight") or m["won"]:
                    break
                b = m["boss"]
                for col in m.get("colliders", []):
                    if col["enabled"]:
                        agg.add(f"{label}/{col['name']}/{b['anim']}", col)
                fist = [h for h in m["hazards"] if h["kind"] == 1]
                if b["state"] in (BossMove.PUNCH_WINDUP, BossMove.PUNCH) and f"{label}.punch" not in timelines:
                    recording = "punch"
                    punch.append({"t": len(punch), "state": BossMove(b["state"]).name, "anim": b["anim"],
                                  "body": [r(b["x"]), r(b["y"]), r(b["half_w"]), r(b["half_h"])],
                                  "fist": [[r(f["x"]), r(f["y"]), r(f["half_w"]), r(f["half_h"])] for f in fist]})
                elif recording == "punch":
                    timelines[f"{label}.punch"] = punch
                    recording = None
                if b["state"] in (BossMove.TOMB_SLAM_WINDUP, BossMove.TOMB_SLAM) and "tomb.slam" not in timelines:
                    recording = "slam"
                    slam.append({"t": len(slam), "state": BossMove(b["state"]).name, "anim": b["anim"],
                                 "body": [r(b["x"]), r(b["y"]), r(b["half_w"]), r(b["half_h"])],
                                 "tomb_hazard": [[r(h["x"]), r(h["y"]), r(h["half_w"]), r(h["half_h"])] for h in m["hazards"] if h["kind"] == 2]})
                elif recording == "slam":
                    timelines["tomb.slam"] = slam
                    recording = None

        idle = lambda i: Action()  # noqa: E731  (no shooting: keep the boss in the phase being sampled)
        observe("small", ticks_per_phase, idle)
        c.damage(290)
        for _ in range(1200):
            m = step()
            if m["boss"]["entity"] == "big":
                break
        observe("big", ticks_per_phase, idle)
        c.damage(545)
        for _ in range(1500):
            m = step()
            if m["boss"]["entity"] == "tomb" and m["boss"]["state"] == BossMove.TOMB_MOVE:
                break

        observe("tomb", ticks_per_phase, idle)
        # does a standing peashooter hit the tombstone? walk back and forth under it shooting, 10 s
        hp0 = c.last_state["boss"]["hp"]
        observe("tomb_shot", 600, lambda i: Action(shoot=True, move=1 if (i // 30) % 2 == 0 else -1))
        m = c.last_state
        tomb_hits_standing = hp0 - (m["boss"]["hp"] if m.get("in_fight") else 0.0)

        out["boss.colliders_by_anim"] = agg.out()
        out["timelines"] = timelines
        out["tomb.damage_from_standing_shots"] = r(tomb_hits_standing)
        out["tomb.note"] = (
            "Single CircleCollider2D r=100 at offset (0,170) from the tombstone transform (rest y_spec 170 => "
            "circle spans y 240..440 unless animated). See boss.colliders_by_anim['tomb/...'] for the live values "
            "and tomb.damage_from_standing_shots for whether ground-level peashots connect."
        )
    finally:
        # never leave the debug no-hit flag on for whoever connects next
        c.invincible(False)
        c.configure(lockstep=False, stream_every=0, debug=False)
        c.close()
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT, f"({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
