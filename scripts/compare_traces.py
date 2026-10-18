"""Sim-vs-game gap report.

    python -m uv run python scripts/compare_traces.py [--traces data/traces] [--seeds 20]
                                                      [--out docs/SIM_VS_GAME.md]

Reads the bridge's real-game traces (``data/traces/*.jsonl``: one line of metadata, then
one line per 60 Hz tick with the action and the spec GameState) and replays the same
recorded actions in the simulator from several seeds. The same statistics are computed
from both (hop timing and height per form, punch timing, transformation lengths, tomb
patrol and slam timing, damage rate, phase durations) and written as a markdown table,
plus an HP-vs-time figure, sorted by the size of the gap.

The traces are in ``data/traces/`` (recorded with ``tools/record_trace.py``).
Hand-written notes below the marker line of the report are kept on regeneration.

Phases are taken from the visible entity (``boss.entity`` in the trace: small / big /
tomb), which is what spec.BossPhase means; older traces reported the HP bracket instead.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from cuphead_rl.spec import BossMove, BossPhase
from cuphead_rl.sim.config import SimConfig
from cuphead_rl.sim.goopy import GROW
from cuphead_rl.sim.player import FrameInput
from cuphead_rl.sim.world import World

ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 60.0
NOTES_MARK = "<!-- hand-written notes below this line are kept when the report is regenerated -->"
# Depend on where the player stands / what it hits: only comparable when the recorded
# policy does not react to the boss (an open-loop replay of a reactive policy diverges).
POLICY_STATS = ("dps.", "final_hp", "episode_s", "phase1.", "phase2.", "transform.small_to_big_hp", "transform.death_hp")

Record = dict  # tick, t, entity, state, x, y, hp, hazards (kinds)


# ------------------------------------------------------------------------------ loading
def load_trace(path: Path) -> tuple[dict, list[Record], list[list[int]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    meta = json.loads(lines[0]).get("meta", {})
    recs, actions = [], []
    for ln in lines[1:]:
        row = json.loads(ln)
        s, b = row["state"], row["state"]["boss"]
        entity = b.get("entity") or ("small", "big", "tomb")[b["phase"]]
        recs.append({"tick": row["tick"], "entity": entity, "state": b["state"], "x": b["x"], "y": b["y"],
                     "hp": b["hp"], "php": s["player"]["hp"]})
        actions.append(row["action"])
    return meta, recs, actions


def sim_rollout(actions: list[list[int]], seed: int, no_hit: bool, cfg: SimConfig) -> list[Record]:
    """Replay recorded per-tick actions ([move, jump, dash, shoot, ex, aim]) in the sim."""
    w = World(cfg, seed=seed)
    recs: list[Record] = []
    jump_prev = False
    for tick, a in enumerate(actions):
        move, jump, dash, shoot, ex = (int(v) for v in a[:5])
        aim = int(a[5]) if len(a) > 5 else 0
        if no_hit:
            w.player.invuln = 1e9
        # Each trace row holds the state AFTER that row's action (tools/record_trace.py).
        w.step(FrameInput(move, bool(jump), bool(jump) and not jump_prev, bool(dash), bool(shoot), bool(ex), aim))
        jump_prev = bool(jump)
        b = w.boss
        if b.phase == BossPhase.SMALL or (b.state == BossMove.TRANSFORM and b.transform == GROW):
            entity = "small"  # the trace keeps the small entity until the transformation ends
        else:
            entity = "big" if b.phase == BossPhase.BIG else "tomb"
        x, y, *_ = b.body()
        recs.append({"tick": tick, "entity": entity, "state": int(b.state), "x": x, "y": y, "hp": b.hp,
                     "php": w.player.hp})
        if w.done:
            break
    return recs


# ------------------------------------------------------------------------------ statistics
def segments(recs: list[Record], pred) -> list[tuple[int, int]]:
    """[start, end) index ranges where pred(rec) holds."""
    out, start = [], None
    for i, r in enumerate(recs):
        if pred(r) and start is None:
            start = i
        elif not pred(r) and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(recs)))
    return out


def first(recs: list[Record], pred) -> int | None:
    return next((i for i, r in enumerate(recs) if pred(r)), None)


def stats(recs: list[Record]) -> tuple[dict[str, float], dict[str, int]]:
    """Statistic -> value, and statistic -> number of events it averages (1 if absent)."""
    out: dict[str, float] = {}
    n: dict[str, int] = {}
    S = {m: int(m) for m in BossMove}

    def put(key: str, vals: list[float], agg=np.mean) -> None:
        out[key], n[key] = float(agg(vals)), len(vals)

    for ent in ("small", "big"):
        jumps = segments(recs, lambda r, e=ent: r["entity"] == e and r["state"] == S[BossMove.JUMP])
        jumps = [(a, b) for a, b in jumps if b < len(recs)]  # complete hops only
        if jumps:
            rest = np.median([recs[b]["y"] for _, b in jumps])
            put(f"{ent}.hop_airtime_s", [(b - a) * DT for a, b in jumps])
            out[f"{ent}.hop_apex_max"] = float(np.max([max(r["y"] for r in recs[a:b]) - rest for a, b in jumps]))
            out[f"{ent}.hop_apex_min"] = float(np.min([max(r["y"] for r in recs[a:b]) - rest for a, b in jumps]))
            put(f"{ent}.hop_dx", [abs(recs[b]["x"] - recs[a]["x"]) for a, b in jumps])
            # Ground time between consecutive hops with only IDLE in between.
            ground = [(c - b) * DT for (a, b), (c, d) in zip(jumps, jumps[1:])
                      if all(r["state"] == S[BossMove.IDLE] for r in recs[b:c])]
            if ground:
                put(f"{ent}.ground_time_s", ground, np.median)
                out[f"{ent}.hop_period_s"] = out[f"{ent}.hop_airtime_s"] + out[f"{ent}.ground_time_s"]
                n[f"{ent}.hop_period_s"] = len(ground)
        windups = segments(recs, lambda r, e=ent: r["entity"] == e and r["state"] == S[BossMove.PUNCH_WINDUP])
        punches = segments(recs, lambda r, e=ent: r["entity"] == e and r["state"] == S[BossMove.PUNCH])
        # Only windups that end in the punch (skips ones cut short by a phase change and
        # one-tick relabels at the punch end). The first punch of big Goopy has a longer
        # blink windup, so it is reported apart.
        windups = [(a, b) for a, b in windups if b < len(recs) and recs[b]["state"] == S[BossMove.PUNCH]]
        if windups:
            out[f"{ent}.punch_windup_first_s"] = (windups[0][1] - windups[0][0]) * DT
        if len(windups) > 1:
            put(f"{ent}.punch_windup_later_s", [(b - a) * DT for a, b in windups[1:]])
        if punches:
            complete = [(a, b) for a, b in punches if b < len(recs)]
            if complete:
                put(f"{ent}.punch_s", [(b - a) * DT for a, b in complete])
                # Punch start -> next take-off: independent of where each backend puts the
                # PUNCH/IDLE boundary inside the Punch_End anim.
                after = []
                for a, b in complete:
                    nxt = first(recs[b:], lambda r: r["state"] == S[BossMove.JUMP])
                    if nxt is not None:
                        after.append((b - a + nxt) * DT)
                if after:
                    put(f"{ent}.punch_to_jump_s", after)
    i_jump = first(recs, lambda r: r["state"] == S[BossMove.JUMP])
    if i_jump is not None:
        out["first_takeoff_s"] = i_jump * DT  # checks the intro and the row alignment
    # Phases and transformations.
    i_tr = first(recs, lambda r: r["entity"] == "small" and r["state"] == S[BossMove.TRANSFORM])
    i_big = first(recs, lambda r: r["entity"] == "big")
    i_die = first(recs, lambda r: r["entity"] == "big" and r["state"] == S[BossMove.TRANSFORM])
    i_tomb = first(recs, lambda r: r["entity"] == "tomb")
    i_land = first(recs, lambda r: r["entity"] == "tomb" and r["state"] == S[BossMove.TOMB_MOVE])
    if i_tr is not None:
        out["phase1.until_transform_s"] = i_tr * DT
        out["phase1.hp_at_transform"] = recs[i_tr]["hp"]
    if i_tr is not None and i_big is not None:
        out["transform.small_to_big_s"] = (i_big - i_tr) * DT
        out["transform.small_to_big_hp_lost"] = recs[i_tr]["hp"] - recs[i_big]["hp"]
    if i_big is not None and i_die is not None:
        out["phase2.duration_s"] = (i_die - i_big) * DT
    if i_die is not None and i_tomb is not None:
        out["transform.death_to_tomb_s"] = (i_tomb - i_die) * DT
        out["transform.death_hp_lost"] = recs[i_die]["hp"] - recs[i_tomb]["hp"]
    if i_tomb is not None and i_land is not None:
        out["transform.tomb_drop_s"] = (i_land - i_tomb) * DT
    # Tombstone.
    if i_land is not None:
        # Wall-to-wall leg time (turn pause included): arrivals at alternating walls,
        # where the resting body sits at x = +-max_x.
        walls = [(i_land + a, recs[i_land + a]["x"] > 0) for a, _ in segments(
            recs[i_land:], lambda r: r["state"] == S[BossMove.TOMB_MOVE] and abs(r["x"]) >= 499.5)]
        legs = [(j - i) * DT for (i, si), (j, sj) in zip(walls, walls[1:]) if si != sj]
        if legs:
            put("tomb.wall_to_wall_s", legs, np.median)
        smashes = [i_land + a for a, _ in segments(recs[i_land:], lambda r: r["state"] == S[BossMove.TOMB_SLAM_WINDUP])]
        if smashes:
            out["tomb.land_to_first_smash_s"] = (smashes[0] - i_land) * DT
        if len(smashes) > 1:
            put("tomb.smash_period_s", list(np.diff(smashes) * DT))
        slams = segments(recs[i_land:], lambda r: r["state"] in (S[BossMove.TOMB_SLAM_WINDUP], S[BossMove.TOMB_SLAM]))
        complete = [(a, b) for a, b in slams if i_land + b < len(recs)]
        if complete:
            put("tomb.smash_s", [(b - a) * DT for a, b in complete])
    # Damage rate per form under the recorded policy.
    for ent in ("small", "big", "tomb"):
        idx = [i for i, r in enumerate(recs) if r["entity"] == ent]
        if len(idx) > 60:
            out[f"dps.{ent}"] = (recs[idx[0]]["hp"] - recs[idx[-1]]["hp"]) / ((idx[-1] - idx[0]) * DT)
    out["episode_s"] = len(recs) * DT
    out["final_hp"] = recs[-1]["hp"]
    return out, n


# ------------------------------------------------------------------------------ report
def compare(trace_path: Path, seeds: int, cfg: SimConfig) -> tuple[dict, dict, dict, list, list]:
    meta, game, actions = load_trace(trace_path)
    g, gn = stats(game)
    runs = [sim_rollout(actions, seed, bool(meta.get("no_hit")), cfg) for seed in range(seeds)]
    per_seed = [stats(r)[0] for r in runs]
    keys = sorted(set(g) | set().union(*per_seed))
    mean = {k: float(np.mean([s[k] for s in per_seed if k in s])) for k in keys if any(k in s for s in per_seed)}
    std = {k: float(np.std([s[k] for s in per_seed if k in s])) for k in keys if any(k in s for s in per_seed)}
    return meta, g, {"mean": mean, "std": std, "n": gn}, game, runs


def gap(g: float, s: float) -> float:
    return abs(s - g) / max(abs(g), 1e-6)


def write_figure(path: Path, traces: list[tuple[str, list, list]], cfg: SimConfig) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(traces), figsize=(6 * len(traces), 3.6), squeeze=False)
    for ax, (name, game, runs) in zip(axes[0], traces):
        for r in runs:
            ax.plot([q["tick"] * DT for q in r], [q["hp"] for q in r], color="#c4302b", alpha=0.25, lw=1)
        ax.plot([q["tick"] * DT for q in game], [q["hp"] for q in game], color="#1f3b57", lw=2, label="game")
        ax.plot([], [], color="#c4302b", label="sim (seeds)")
        for frac in (cfg.transition.big_trigger_frac, cfg.transition.tomb_trigger_frac):
            ax.axhline(cfg.total_hp * frac, color="#999", lw=0.8, ls="--")
        ax.set_title(name)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("boss HP")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(ROOT / "data" / "traces"))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "docs" / "SIM_VS_GAME.md"))
    ap.add_argument("--figure", default=str(ROOT / "docs" / "sim_vs_game_hp.png"))
    args = ap.parse_args()
    cfg = SimConfig()
    lines = ["# Sim vs game", "",
             "Generated by `scripts/compare_traces.py`: the recorded real-game action sequence of each "
             f"trace is replayed in the simulator from {args.seeds} seeds (mean ± std); the same "
             "statistics are computed from both. Gap = |sim − game| / |game|; n = number of game "
             "events behind the game value (n = 1: a single sample, read the sim std before the gap). "
             "Rows sorted by gap.", ""]
    figs, all_gaps = [], []
    for path in sorted(Path(args.traces).glob("*.jsonl")):
        meta, g, sim, game, runs = compare(path, args.seeds, cfg)
        figs.append((path.stem, game, runs))
        open_loop = not meta.get("no_hit")
        lines += [f"## {path.stem}", "",
                  f"policy `{meta.get('policy')}`, {len(game)} ticks, no_hit={meta.get('no_hit')}, "
                  f"seed {meta.get('seed')}", ""]
        if open_loop:
            lines += ["The recorded policy reacts to the boss and the player can be hit, so the "
                      "open-loop replay drifts from the game run: damage/phase statistics are listed "
                      "but left out of the largest gaps.", ""]
        lines += ["| statistic | game | n | sim | gap |", "|---|---|---|---|---|"]
        rows = []
        for k in sorted(set(g) | set(sim["mean"])):
            gv, sv, sd = g.get(k), sim["mean"].get(k), sim["std"].get(k, 0.0)
            gp = gap(gv, sv) if gv is not None and sv is not None else math.inf
            rows.append((gp, k, gv, sv, sd))
        rows.sort(key=lambda r: -r[0] if math.isfinite(r[0]) else 1.0)  # no game value: last
        for gp, k, gv, sv, sd in rows:
            gs = "—" if gv is None else f"{gv:.3g}"
            ss = "—" if sv is None else f"{sv:.3g} ± {sd:.2g}"
            ps = "—" if not math.isfinite(gp) else f"{100 * gp:.0f} %"
            ns = "—" if gv is None else sim["n"].get(k, 1)
            lines.append(f"| {k} | {gs} | {ns} | {ss} | {ps} |")
            if math.isfinite(gp) and not (open_loop and k.startswith(POLICY_STATS)):
                all_gaps.append((gp, path.stem, k, gv, sv))
        lines.append("")
    write_figure(Path(args.figure), figs, cfg)
    lines += [f"![HP vs time]({Path(args.figure).name})", ""]
    all_gaps.sort(reverse=True)
    lines += ["## Largest gaps", ""]
    for gp, tr, k, gv, sv in all_gaps[:10]:
        lines.append(f"* `{tr}` {k}: game {gv:.3g}, sim {sv:.3g} ({100 * gp:.0f} %)")
    out = Path(args.out)
    old = out.read_text(encoding="utf-8") if out.exists() else ""
    notes = old[old.index(NOTES_MARK):] if NOTES_MARK in old else NOTES_MARK + "\n"
    out.write_text("\n".join(lines) + "\n\n" + notes, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
