"""Per-episode results and the likely cause of every hit, from scripts/play_game.py logs, or from
the same policy in the simulator for comparison.

    python -m uv run python scripts/game_hits.py videos/<dir> [videos/<dir2> ...] [--out hits.json]
    python -m uv run python scripts/game_hits.py --sim runs/<name>/final.zip --episodes 200 [--out hits.json]

A hit is a decision step whose ``player_hp`` dropped. Its cause is the hazard whose box comes
closest to the player's hurtbox (standing box, 50 x 105, centre 62 above the feet, from the sim's
calibration) in the states just before and just after that step; the hit itself happened at some
tick in between (frame_skip ticks), so a gap of a few units is still a contact. The boss move is
reported before -> after the step. Game logs need ``state`` (play_game.py ``--record`` or ``--log``
writes it per step). ``--sim`` plays the checkpoint deterministically in the nominal simulator
(seeds from 10000, as in evaluation) and logs the same compact state.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HURT_HALF_W, HURT_HALF_H, HURT_CY = 25.0, 52.5, 62.0
PHASES = ("I", "II", "III")


def gap(player: list, hz: list) -> float:
    """Distance between the player's hurtbox and a hazard box (0 = overlapping)."""
    px, py = player[0], player[1] + HURT_CY
    _, x, y, hw, hh = hz
    gx = max(0.0, abs(x - px) - hw - HURT_HALF_W)
    gy = max(0.0, abs(y - py) - hh - HURT_HALF_H)
    return (gx * gx + gy * gy) ** 0.5


def find_hits(steps: list[dict], start_hp: int) -> list[dict]:
    hits, hp = [], start_hp
    for i, e in enumerate(steps):
        if e["player_hp"] >= hp:
            continue
        pre = steps[i - 1]["state"] if i else e["state"]
        post = e["state"]
        cands = [(gap(s["p"], h), h[0], tag) for tag, s in (("pre", pre), ("post", post)) for h in s["hz"]]
        g, kind, tag = min(cands) if cands else (float("nan"), "?", "")
        hits.append({
            "t": e["t"], "phase": e["phase"], "hp_after": e["player_hp"],
            "boss_move": f"{pre['b'][0]}->{post['b'][0]}", "cause": kind, "gap": round(g, 1), "when": tag,
            "player": {"x": post["p"][0], "y": post["p"][1], "airborne": post["p"][1] > 1.0, "dashing": bool(post["p"][3])},
            "boss_x": post["b"][1],
        })
        hp = e["player_hp"]
    return hits


def episode_hits(path: Path) -> dict:
    lines = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    meta, steps = lines[0]["meta"], lines[1:]
    start_hp = meta["result"]["hits_taken"] + (steps[-1]["player_hp"] if steps else 0)
    return {"source": str(path.parent), "episode": meta["episode"], "seed": meta["seed"], **meta["result"],
            "hits": find_hits(steps, start_hp)}


def sim_episodes(checkpoint: Path, episodes: int, seed: int = 10_000) -> list[dict]:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    from play_game import compact_state, find_config

    from cuphead_rl.train.agent import AgentPolicy, load_model
    from cuphead_rl.train.config import ExperimentConfig
    from cuphead_rl.train.envs import make_base_env

    cfg = ExperimentConfig.load(find_config(checkpoint))
    policy = AgentPolicy(load_model(checkpoint), frame_stack=cfg.frame_stack, deterministic=True, introspect=False)
    env = make_base_env(cfg, nominal=True)
    out = []
    try:
        for i in range(episodes):
            obs, info = env.reset(seed=seed + i)
            policy.reset()
            start_hp, steps, done = info["player_hp"], [], False
            while not done:
                t = float(info["time"])
                obs, _, term, trunc, info = env.step(policy(obs))
                done = term or trunc
                steps.append({"t": round(t, 4), "phase": int(info["phase"]), "player_hp": int(info["player_hp"]),
                              "state": compact_state(env.unwrapped.state)})
            out.append({"source": "sim", "episode": i, "seed": seed + i, "won": bool(info["won"]),
                        "phase": int(info["phase"]), "boss_damage": 1.0 - float(info["boss_hp_frac"]),
                        "hits_taken": int(start_hp - info["player_hp"]), "time": float(info["time"]),
                        "hits": find_hits(steps, start_hp)})
    finally:
        env.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logdirs", type=Path, nargs="*")
    ap.add_argument("--sim", type=Path, metavar="CHECKPOINT", help="analyse this checkpoint in the simulator instead")
    ap.add_argument("--episodes", type=int, default=200, help="episodes with --sim")
    ap.add_argument("--quiet", action="store_true", help="only the per-cause summary")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.sim:
        eps = sim_episodes(args.sim, args.episodes)
    else:
        eps = [episode_hits(p) for d in args.logdirs for p in sorted(d.glob("episode_*.jsonl"))]
    causes = collections.Counter()
    if not args.quiet:
        print("ep  seed   won  phase  boss_dmg  hits  time")
    for e in eps:
        for h in e["hits"]:
            causes[(PHASES[h["phase"]], h["cause"], h["boss_move"])] += 1
        if args.quiet:
            continue
        print(f"{e['episode']:>2}  {e['seed']}  {'Y' if e['won'] else '-':>3}  {PHASES[e['phase']]:>5}  "
              f"{100 * e['boss_damage']:7.1f}%  {e['hits_taken']:>4}  {e['time']:5.1f}s")
        for h in e["hits"]:
            print(f"      hit t={h['t']:5.2f}s phase {PHASES[h['phase']]:<3} {h['cause']:<10} "
                  f"(gap {h['gap']:5.1f}, {h['when']})  boss {h['boss_move']:<26} "
                  f"player x={h['player']['x']:6.1f} {'air' if h['player']['airborne'] else 'ground'}"
                  f"{' dash' if h['player']['dashing'] else ''}")
    n_hits = sum(causes.values())
    wins = sum(e["won"] for e in eps)
    print(f"\n{len(eps)} episodes, {wins} wins, {n_hits} hits ({n_hits / max(len(eps), 1):.2f} per episode)")
    print("causes (phase, hazard, boss move):")
    for k, n in causes.most_common():
        print(f"  {n:>3}  {k}")
    if args.out:
        args.out.write_text(json.dumps({"episodes": eps, "causes": [[*k, n] for k, n in causes.most_common()]}, indent=1))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
