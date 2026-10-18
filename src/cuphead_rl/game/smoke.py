"""Smoke test for the real-game bridge.

    python -m uv run python -m cuphead_rl.game.smoke [--record videos/game_smoke.mp4]

Needs Cuphead running with BepInEx + CupheadRLBridge (docs/GAME_BRIDGE.md). It connects,
(re)loads the Goopy fight, prints live states and the achieved update rate, runs a scripted
walk/jump/shoot/dash policy in real time, benchmarks lockstep stepping, and checks reset.
"""

from __future__ import annotations

import argparse
import time

from cuphead_rl import spec
from cuphead_rl.game.capture import WindowRecorder
from cuphead_rl.game.client import BridgeClient, to_game_state


def fmt(msg: dict) -> str:
    if not msg.get("in_fight"):
        return f"[not in fight: {msg.get('scene')}]"
    s = to_game_state(msg)
    p, b = s.player, s.boss
    return (
        f"tick={msg['tick']:5d} t={s.time:6.2f} | P x={p.x:6.0f} y={p.y:5.0f} v=({p.vx:6.0f},{p.vy:6.0f}) "
        f"g={int(p.grounded)} d={int(p.dashing)} hp={p.hp} inv={p.invuln:.1f} sm={p.super_meter:.2f} | "
        f"B {b.phase.name}/{b.state.name:<16} x={b.x:6.0f} y={b.y:5.0f} hw={b.half_w:4.0f} "
        f"hp={b.hp:6.1f} anim={msg['boss']['anim']} | hz={len(s.hazards)} dmg={s.damage_dealt:.0f}"
    )


def scripted(t: float, meter: float) -> spec.Action:
    """A fixed walk / jump / shoot / dash routine, ~10 s long, keyed on elapsed seconds."""
    phase = t % 10.0
    if phase < 1.5:
        return spec.Action(move=1, shoot=True)
    if phase < 1.9:
        return spec.Action(move=1, jump=True, shoot=True)
    if phase < 3.5:
        return spec.Action(move=-1, shoot=True)
    if phase < 3.6:
        return spec.Action(move=-1, dash=True)
    if phase < 4.5:
        return spec.Action(shoot=True)
    if phase < 4.9:
        return spec.Action(jump=True, shoot=True)
    if phase < 5.0:
        return spec.Action(move=1, dash=True)
    if phase < 7.0:
        return spec.Action(move=1 if int(phase * 2) % 2 else -1, jump=phase % 1.0 < 0.3, shoot=True)
    if phase < 7.1 and meter >= 1.0:
        return spec.Action(ex=True)
    return spec.Action(shoot=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47800)
    ap.add_argument("--monitor", type=float, default=3.0, help="seconds of passive state printing")
    ap.add_argument("--policy", type=float, default=12.0, help="seconds of scripted policy (real time)")
    ap.add_argument("--lockstep-steps", type=int, default=600)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--record", default=None, help="MP4 path for a capture of the scripted policy")
    ap.add_argument("--print-every", type=float, default=0.25)
    args = ap.parse_args()

    c = BridgeClient(args.host, args.port)
    hello = c.connect(wait=180.0)
    print("hello:", hello)
    print("config:", c.configure(lockstep=False, stream_every=1, time_scale=1.0, control=True))

    t0 = time.monotonic()
    first = c.reset()
    print(f"reset -> fight started in {time.monotonic() - t0:.1f}s")
    print(fmt(first))

    # ---- passive monitoring: update rate
    n, t_start, last_print = 0, time.monotonic(), 0.0
    tick0 = first["tick"]
    while time.monotonic() - t_start < args.monitor:
        msg = c.wait_for("state")
        n += 1
        if time.monotonic() - last_print >= args.print_every:
            last_print = time.monotonic()
            print(fmt(msg))
    dt = time.monotonic() - t_start
    ticks = c.last_state["tick"] - tick0
    print(f"free-run update rate: {n / dt:.1f} states/s, {ticks / dt:.1f} game ticks/s")

    # ---- scripted policy in real time (optionally recorded)
    rec = WindowRecorder(args.record).start() if args.record else None
    try:
        t_start, last_print = time.monotonic(), 0.0
        while time.monotonic() - t_start < args.policy:
            msg = c.last_state
            meter = msg["player"]["super_meter"] if msg and msg.get("in_fight") else 0.0
            c.act(scripted(time.monotonic() - t_start, meter))
            msg = c.wait_for("state")
            if time.monotonic() - last_print >= args.print_every:
                last_print = time.monotonic()
                print(fmt(msg))
            if msg.get("in_fight") and (msg["won"] or msg["lost"]):
                print("episode ended:", "won" if msg["won"] else "lost")
                break
    finally:
        if rec:
            print("recorded", rec.stop())
    c.act(spec.Action())

    # ---- lockstep benchmark
    if args.lockstep_steps > 0:
        print("reset ->", f"{c.reset()['tick']}")
        c.configure(lockstep=True, frames=args.frame_skip, stream_every=0)
        c.wait_for("state")  # lockstep emits a state, then waits for an action
        t_start = time.monotonic()
        tick_start = c.last_state["tick"]
        msg = c.last_state
        for i in range(args.lockstep_steps):
            meter = msg["player"]["super_meter"] if msg.get("in_fight") else 0.0
            msg = c.step_lockstep(scripted(i * args.frame_skip / spec.GAME_FPS, meter), args.frame_skip)
            if i % 50 == 0:
                print(fmt(msg))
            if not msg.get("in_fight") or msg["won"] or msg["lost"]:
                break
        dt = time.monotonic() - t_start
        steps = i + 1
        ticks = msg.get("tick", tick_start) - tick_start
        print(
            f"lockstep: {steps / dt:.1f} steps/s, {ticks / dt:.1f} ticks/s "
            f"({ticks / dt / spec.GAME_FPS:.2f}x real time), {ticks / steps:.2f} ticks/step"
        )
        c.configure(lockstep=False, stream_every=1)

    # ---- reset check
    t0 = time.monotonic()
    st = to_game_state(c.reset())
    print(
        f"reset check: {time.monotonic() - t0:.1f}s, player hp={st.player.hp}, boss hp={st.boss.hp:.0f}/"
        f"{st.boss.max_hp:.0f}, t={st.time:.2f}"
    )
    c.configure(lockstep=False, stream_every=0)
    c.close()


if __name__ == "__main__":
    main()
