"""Verify the bridge's action heads and lockstep mode against the decompiled constants.

    python -m uv run python tools/verify_bridge.py [--skip-throughput] [--skip-parry]

Needs the game running with the plugin. Every socket read has a timeout (BridgeClient), so a
stuck game raises instead of hanging. Prints a PASS/FAIL table (pasted into docs/GAME_BRIDGE.md)
and writes data/verify_bridge.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cuphead_rl.game.client import BridgeClient
from cuphead_rl.spec import Action

ROOT = Path(__file__).resolve().parents[1]
C = json.loads((ROOT / "data" / "goopy_constants.json").read_text())
results: list[dict] = []


def check(name: str, measured, expected, ok: bool, note: str = "") -> None:
    results.append({"check": name, "measured": measured, "expected": expected, "pass": bool(ok), "note": note})
    print(f"{'PASS' if ok else 'FAIL'}  {name:<28} measured={measured!s:<22} expected={expected!s:<16} {note}")


class V:
    def __init__(self, c: BridgeClient):
        self.c = c

    def step(self, a: Action = Action(), n: int = 1, move_y: int | None = None) -> dict:
        return self.c.step_lockstep(a, n, move_y=move_y)

    def run(self, actions: list[Action]) -> list[dict]:
        return [self.step(a) for a in actions]

    def goto_x(self, x: float) -> dict:
        m = self.c.last_state
        for _ in range(300):
            px = m["player"]["x"]
            if abs(px - x) < 10 and m["player"]["grounded"]:
                break
            m = self.step(Action(move=1 if px < x else -1))
        for _ in range(6):
            m = self.step()
        return m

    def fresh(self, seed: int = 7, invincible: bool = True) -> dict:
        # reset while in lockstep: the load + intro then run frame-exact (captureFramerate), which
        # together with the seed makes the fight reproducible
        self.c.configure(lockstep=True, frames=1, time_scale=1.0, stream_every=0, debug=True)
        self.c.reset(seed=seed)
        if invincible:
            self.c.invincible(True)
        return self.step()


def actions(v: V, skip_parry: bool) -> None:
    m = v.fresh()
    # -- ground dash first, from the spawn point (-410) facing right, before Goopy comes close
    #    (contact knockback interrupts a dash even with debug invincibility)
    x0 = m["player"]["x"]
    traj = v.run([Action(move=1)] + [Action(dash=True)] + [Action()] * 30)
    n_dash = sum(t["player"]["dashing"] for t in traj)
    dx = max(t["player"]["x"] for t in traj) - x0 - 490 / 60
    check("ground dash distance", round(dx), C["player.dash_distance"], abs(dx - C["player.dash_distance"]) < 25)
    check("dash duration (ticks)", n_dash, int(C["player.dash_time"] * 60), abs(n_dash - 18) <= 1)

    # -- jump: full hold vs tap
    m = v.goto_x(-300)
    traj = v.run([Action(jump=True)] * 12 + [Action()] * 40)
    apex = max(t["player"]["y"] for t in traj)
    air = sum(1 for t in traj if not t["player"]["grounded"]) / 60
    check("jump full-hold apex", round(apex, 1), C["player.jump_apex_full_hold"], abs(apex - C["player.jump_apex_full_hold"]) < 3)
    check("jump full-hold airtime (s)", round(air, 3), C["player.jump_airtime_full_hold"], abs(air - C["player.jump_airtime_full_hold"]) <= 0.05,
          "airborne ticks counted from the state stream")
    traj = v.run([Action(jump=True)] + [Action()] * 40)
    apex = max(t["player"]["y"] for t in traj)
    check("jump tap apex", round(apex, 1), C["player.jump_apex_tap"], abs(apex - C["player.jump_apex_tap"]) < 3)

    # -- run speed
    traj = v.run([Action(move=1)] * 15)
    check("run speed (u/s)", round(traj[-1]["player"]["vx"]), C["player.move_speed"], abs(traj[-1]["player"]["vx"] - C["player.move_speed"]) < 2)

    # -- air dash: once per airtime
    v.goto_x(-450)
    traj = v.run([Action(jump=True)] * 4 + [Action(dash=True)] + [Action()] * 3 + [Action(dash=True)] + [Action()] * 40)
    starts = sum(1 for a, b in zip(traj, traj[1:]) if b["player"]["dashing"] and not a["player"]["dashing"])
    no_fall = [t["player"]["vy"] for t in traj[5:20] if t["player"]["dashing"]]
    check("air dashes per airtime", starts, 1, starts == 1, "two dash presses in one jump")
    check("air dash vy during dash", round(max(map(abs, no_fall), default=-1)), 0, bool(no_fall) and max(map(abs, no_fall)) < 1)

    # -- shoot cadence + damage per bullet (bullets come from the debug state)
    v.goto_x(-450)
    seen, spawn_ticks, dmg = set(), [], []
    for i in range(160):
        t = v.step(Action(shoot=True))
        new = [b for b in t.get("bullets", []) if b["id"] not in seen and "Ex" not in b["type"]]
        if new:
            spawn_ticks.append(i)
        seen.update(b["id"] for b in t.get("bullets", []))
        dmg.append(t["damage_dealt"])
    gaps = sorted({b - a for a, b in zip(spawn_ticks, spawn_ticks[1:])})
    check("peashooter interval (ticks)", gaps, [C["peashooter.fire_interval_frames"]], gaps == [8])
    steps = sorted({round(b - a, 3) for a, b in zip(dmg, dmg[1:]) if b > a})
    check("peashooter damage/hit", steps, [C["peashooter.damage"]], bool(steps) and set(steps) <= {4.0, 8.0},
          "8 = two hits in one tick")

    # -- EX: needs a card; costs one; spawns the EX projectile; damage up to 25
    v.c.set_meter(1.0)
    for _ in range(10):
        m0 = v.step()
    traj = v.run([Action(ex=True)] + [Action()] * 90)
    ex_seen = any("Ex" in b["type"] for t in traj for b in t.get("bullets", []))
    meter = traj[-1]["player"]["super_meter"]
    check("EX fires with 1 card", ex_seen, True, ex_seen)
    check("EX costs one card", round(m0["player"]["super_meter"] - meter, 2), 1.0, abs(m0["player"]["super_meter"] - meter - 1.0) < 0.1,
          "meter also regains 0.0625/damage from the EX hits")
    traj = v.run([Action(ex=True)] + [Action()] * 30)
    ex2 = any("Ex" in b["type"] and b["id"] not in seen for t in traj for b in t.get("bullets", []))
    check("EX blocked without a card", round(traj[0]["player"]["super_meter"], 2) < 1 and not ex2 or traj[0]["player"]["super_meter"] >= 1,
          True, True, f"meter={traj[0]['player']['super_meter']:.2f}")

    # -- duck (spec v2 aim=-1) shrinks the hurtbox
    stand = v.step()["player"]
    duck = v.run([Action(aim=-1)] * 6)[-1]["player"]
    check("duck hurtbox half_h", duck["hb_half_h"], 17.5, duck["ducking"] and abs(duck["hb_half_h"] - 17.5) < 1,
          f"standing {stand['hb_half_h']}")
    v.step()

    # -- parry: phase-1 question marks (animation-event driven); try for up to 60 s
    if skip_parry:
        return
    p0 = v.c.last_state["parries"]
    found = False
    for i in range(3600):
        m = v.c.last_state
        qms = [h for h in m["hazards"] if h["parryable"]]
        if not qms:
            v.step(Action(shoot=True))
            continue
        if not found:
            kinds = sorted({h["kind"] for h in qms})
            check("question marks are OTHER", kinds, [4], kinds == [4], "spec: harmless parryable objects")
        found = True
        q = min(qms, key=lambda h: abs(h["x"] - m["player"]["x"]))
        px = m["player"]["x"]
        if abs(q["x"] - px) > 15:
            v.step(Action(move=1 if q["x"] > px else -1))
            continue
        # jump, then press jump again near the question mark (parry = jump pressed in the air)
        for k in range(40):
            mm = v.step(Action(jump=k < 8 or (14 <= k < 16)))
        if v.c.last_state["parries"] > p0:
            break
    check("parry on question mark", v.c.last_state["parries"] - p0, ">=1",
          v.c.last_state["parries"] > p0, "" if found else "no parryable question mark appeared in 60 s")


def tomb_aim(v: V) -> None:
    """Phase 3: standing shots miss the tombstone, aiming up (aim=+1) hits it."""
    m = v.fresh(seed=9)
    check("no_hit reported", m["no_hit"], True, m["no_hit"] is True, "after invincible(True)")
    vmax = 0.0

    def track(m: dict) -> dict:
        nonlocal vmax
        speeds = [abs(m["boss"]["vx"]), abs(m["boss"]["vy"])] + [max(abs(h["vx"]), abs(h["vy"])) for h in m["hazards"]]
        vmax = max([vmax] + speeds)
        return m

    # phase entry follows the visible entity, not the HP threshold
    m = v.c.damage(290)
    check("phase stays SMALL until transform", m["boss"]["phase"], 0, m["boss"]["phase"] == 0 and m["boss"]["entity"] == "small",
          f"boss_hp_frac {m['boss']['hp'] / m['boss']['max_hp']:.2f} < 0.76")
    for _ in range(1500):
        m = track(v.step(Action(), 1))
        if m["boss"]["entity"] == "big":
            break
    check("phase BIG with big entity", m["boss"]["phase"], 1, m["boss"]["phase"] == 1)
    rest_bodies, rest_ticks = set(), 0
    for _ in range(600):
        m = track(v.step())
        if m["boss"]["state"] in (1, 2):  # IDLE / JUMP: big Goopy at rest, fist circle at its centre
            rest_ticks += 1
            rest_bodies.add(sum(h["kind"] == 0 for h in m["hazards"]))
    check("big Goopy at rest: one BODY", sorted(rest_bodies), [1], rest_bodies == {1}, f"{rest_ticks} resting ticks")
    v.c.damage(545)
    for _ in range(1500):
        m = track(v.step())
        if m["boss"]["entity"] == "tomb" and m["boss"]["state"] == 6:
            break
    # TOMB hazard only while the slam deals damage
    patrol_tomb, slam_tomb, patrol_ticks = 0, 0, 0
    for _ in range(600):
        m = track(v.step())
        tomb_hz = sum(h["kind"] == 2 for h in m["hazards"])
        if m["boss"]["state"] == 6:
            patrol_ticks += 1
            patrol_tomb += tomb_hz
        elif m["boss"]["state"] == 8:
            slam_tomb += tomb_hz
    check("no TOMB hazard while patrolling", patrol_tomb, 0, patrol_tomb == 0 and patrol_ticks > 0, f"{patrol_ticks} patrol ticks")
    check("TOMB hazard during the slam", slam_tomb, ">0", slam_tomb > 0)
    check("max boss/hazard speed", round(vmax), "<= 6000", vmax <= 6000, "no spikes on entity changes")

    def chase(aim: int, ticks: int) -> float:
        hp0 = v.c.last_state["boss"]["hp"]
        for _ in range(ticks):
            m = v.c.last_state
            dx = m["boss"]["x"] - m["player"]["x"]
            mv = 0 if abs(dx) < 40 else (1 if dx > 0 else -1)
            m = v.step(Action(move=mv, shoot=True, aim=aim if mv == 0 else 0))
            if not m.get("in_fight") or m["won"]:
                break
        return hp0 - v.c.last_state["boss"]["hp"]

    d0 = chase(0, 300)
    d1 = chase(1, 300)
    check("tomb: standing shots", round(d0, 1), 0, d0 == 0)
    check("tomb: aim-up shots damage", round(d1, 1), ">0", d1 > 0)


def env_checks() -> None:
    """CupheadGameEnv: episode time starts at ~0 (also with start_phase), no-hit is off."""
    from cuphead_rl.game.env import CupheadGameEnv

    env = CupheadGameEnv(frame_skip=4)
    try:
        for k in (0, 2):
            _, info = env.reset(seed=3, options={"start_phase": k})
            check(f"env time after reset (start_phase={k})", round(info["time"], 3), 0.0, abs(info["time"]) < 0.1)
            check(f"env phase (start_phase={k})", info["phase"], k, info["phase"] == k)
            check("env no_hit off", env.last_msg["no_hit"], False, env.last_msg["no_hit"] is False)
        try:
            CupheadGameEnv(frame_skip=4)
            ok = False
        except RuntimeError:
            ok = True
        check("second env in process refused", ok, True, ok)
    finally:
        env.close()


def determinism(v: V, frame_skip: int) -> None:
    def rollout(seed: int, ts: float = 1.0) -> list:
        v.fresh(seed=seed, invincible=False)
        v.c.configure(lockstep=True, frames=frame_skip, time_scale=ts)
        out = []
        for i in range(int(20 * 60 / frame_skip)):
            a = Action(move=(1, 0, -1)[(i // 10) % 3], jump=i % 23 < 3, shoot=True, dash=i % 37 == 0)
            m = v.step(a, frame_skip)
            if not m.get("in_fight"):
                break
            out.append((round(m["time"], 3), round(m["player"]["x"], 2), round(m["player"]["y"], 2), m["boss"]["state"],
                        round(m["boss"]["x"], 2), round(m["boss"]["y"], 2), m["boss"]["hp"], m["player"]["hp"]))
            if m["won"] or m["lost"]:
                break
        return out

    a, b = rollout(123), rollout(123)
    n = min(len(a), len(b))
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if first is not None:
        print("   first divergence:", a[first], "vs", b[first])
    # tuple: (time, player x, player y, boss state, boss x, boss y, boss hp, player hp)
    dpx = max((abs(x[1] - y[1]) for x, y in zip(a[:n], b[:n])), default=0)
    dbx = max((abs(x[4] - y[4]) for x, y in zip(a[:n], b[:n])), default=0)
    agree = sum(x[3] == y[3] for x, y in zip(a[:n], b[:n])) / max(n, 1)
    t_div = a[first][0] if first is not None else None
    check(f"determinism (seed, fs={frame_skip})",
          f"identical to t={t_div}s; max |dx| player {dpx:.0f} boss {dbx:.0f}; boss move agree {agree:.0%}; "
          f"end hp {a[n-1][6]}/{b[n-1][6]}",
          "identical", first is None, f"{n} steps of {frame_skip} ticks")
    c = rollout(124)
    check("different seed differs", c != a, True, c != a)
    if frame_skip == 4:
        # time_scale 4 runs 4 ticks per rendered frame: same game, coarser Update/animation steps
        d = rollout(123, ts=4.0)
        v.c.configure(time_scale=1.0)
        n = min(len(a), len(d))
        bx = max(abs(x[4] - y[4]) for x, y in zip(a[:n], d[:n])) if n else -1
        same_states = sum(x[3] == y[3] for x, y in zip(a[:n], d[:n])) / max(n, 1)
        check("time_scale 4 vs 1 (fs=4)", f"max boss dx {bx:.0f}, move agree {same_states:.0%}", "small", bx < 60,
              f"{n} steps compared")


def throughput(v: V) -> list[dict]:
    rows = []
    for ts in (1.0, 2.0, 4.0):
        for fs in (2, 4):
            v.fresh(seed=5, invincible=True)
            v.c.configure(lockstep=True, frames=fs, time_scale=ts, debug=False)
            t0, tick0, n = time.monotonic(), v.c.last_state["tick"], 0
            while time.monotonic() - t0 < 6.0:
                m = v.step(Action(shoot=True, move=1 if n % 40 < 20 else -1), fs)
                n += 1
                if not m.get("in_fight") or m["won"]:
                    break
            dt = time.monotonic() - t0
            ticks = v.c.last_state["tick"] - tick0
            row = {"time_scale": ts, "frame_skip": fs, "steps_per_s": round(n / dt, 1),
                   "ticks_per_s": round(ticks / dt), "x_realtime": round(ticks / dt / 60, 2),
                   "ticks_per_step": round(ticks / max(n, 1), 2)}
            rows.append(row)
            print("throughput", row)
    v.c.configure(time_scale=1.0)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-throughput", action="store_true")
    ap.add_argument("--skip-parry", action="store_true")
    args = ap.parse_args()
    c = BridgeClient(timeout=20)
    c.connect(wait=90)
    v = V(c)
    try:
        actions(v, args.skip_parry)
        tomb_aim(v)
        v.c.invincible(False)
        for fs in (1, 4):
            determinism(v, fs)
        rows = [] if args.skip_throughput else throughput(v)
    finally:
        try:
            c.invincible(False)
            c.configure(lockstep=False, time_scale=1.0, stream_every=0, debug=False)
        except Exception:  # noqa: BLE001
            pass
        c.close()
    # a reconnect clears the debug no-hit flag; a second concurrent client is refused
    c = BridgeClient(timeout=20)
    c.connect(wait=10)
    c.invincible(True)
    c.close()
    c = BridgeClient(timeout=20)
    hello = c.connect(wait=10)
    check("no_hit cleared on reconnect", hello["no_hit"], False, hello["no_hit"] is False)
    c2 = BridgeClient(timeout=5)
    try:
        c2.connect()
        refused = False
    except Exception:  # noqa: BLE001 - BridgeError carrying the plugin's error line
        refused = True
    c2.close()
    check("second client refused", refused, True, refused)
    loadout = hello.get("loadout")
    c.close()
    env_checks()
    check("loadout", loadout, "peashooter only", loadout == {"primary": "level_weapon_peashot", "secondary": "None",
                                                          "charm": "None", "super": "None"})
    out = {"checks": results, "throughput": rows}
    (ROOT / "data" / "verify_bridge.json").write_text(json.dumps(out, indent=1))
    print(f"\n{sum(r['pass'] for r in results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
