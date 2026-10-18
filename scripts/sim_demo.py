"""Render sample videos of the simulator and measure baseline win rates.

    python -m uv run python scripts/sim_demo.py [--episodes 50] [--out videos]

Writes <out>/sim_random.mp4 and <out>/sim_heuristic.mp4 and prints random vs heuristic
win rates plus the env throughput (steps/s at frame_skip=4, no rendering).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cuphead_rl import spec
from cuphead_rl.sim import GoopyEnv
from cuphead_rl.sim.policies import heuristic_policy, random_policy
from cuphead_rl.sim.video import record_episode


def evaluate(policy, episodes: int, seed0: int = 0) -> dict:
    """Win rate, phase reached, episode length and time spent in each (completed) phase."""
    env = GoopyEnv()
    wins, phases, times, hp_left, truncs = 0, [], [], [], 0
    phase_times: dict[int, list[float]] = {0: [], 1: [], 2: []}
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        phase, t_start = 0, 0.0
        while True:
            obs, _, term, trunc, info = env.step(policy(obs))
            if info["phase"] != phase:
                phase_times[phase].append(info["time"] - t_start)
                phase, t_start = info["phase"], info["time"]
            if term or trunc:
                break
        if info["won"]:
            phase_times[2].append(info["time"] - t_start)
        wins += info["won"]
        truncs += trunc
        phases.append(info["phase"])
        times.append(info["time"])
        hp_left.append(info["boss_hp_frac"])
    return {"win_rate": wins / episodes, "mean_phase": float(np.mean(phases)),
            "mean_time": float(np.mean(times)), "max_time": float(np.max(times)), "truncated": truncs,
            "mean_boss_hp_frac": float(np.mean(hp_left)),
            "phase_time": {k: (float(np.mean(v)) if v else None) for k, v in phase_times.items()}}


def benchmark(steps: int = 30000) -> float:
    env = GoopyEnv(frame_skip=4)
    rng = np.random.default_rng(0)
    actions = [env.action_space.sample() for _ in range(1000)]
    env.reset(seed=0)
    t0 = time.perf_counter()
    for i in range(steps):
        _, _, term, trunc, _ = env.step(actions[i % 1000])
        if term or trunc:
            env.reset(seed=int(rng.integers(1 << 30)))
    return steps / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", default="videos")
    args = ap.parse_args()
    out = Path(args.out)

    print(f"env throughput: {benchmark():,.0f} steps/s (frame_skip=4, no render)")
    for name, pol in (("random", random_policy), ("heuristic", heuristic_policy)):
        r = evaluate(pol, args.episodes)
        print(f"{name:9s} over {args.episodes} episodes: win rate {r['win_rate']:.0%}, "
              f"mean phase reached {r['mean_phase']:.2f}, mean boss HP left {r['mean_boss_hp_frac']:.0%}, "
              f"mean episode {r['mean_time']:.1f}s (max {r['max_time']:.1f}s, {r['truncated']} truncated at "
              f"{spec.MAX_EPISODE_SECONDS:.0f}s), mean time per completed phase "
              + ", ".join(f"P{k + 1} {v:.1f}s" if v is not None else f"P{k + 1} -" for k, v in r["phase_time"].items()))

    rng = np.random.default_rng(0)
    env = GoopyEnv()
    stats = record_episode(env, lambda o: random_policy(o, rng), out / "sim_random.mp4", seed=0,
                           overlay_fn=lambda: {"policy": "random"})
    print("random video:", stats, _size(out / "sim_random.mp4", stats))
    # Pick a seed the heuristic wins so the sample video shows the whole fight.
    seed = next((s for s in range(50) if _wins(s)), 0)
    stats = record_episode(env, heuristic_policy, out / "sim_heuristic.mp4", seed=seed,
                           overlay_fn=lambda: {"policy": "heuristic", "seed": seed})
    print("heuristic video:", stats, _size(out / "sim_heuristic.mp4", stats))


def _size(path: Path, stats: dict) -> str:
    mb = path.stat().st_size / 1e6
    seconds = stats["frames"] / 60.0 + 1.0
    return f"[{mb:.1f} MB, {8 * mb / seconds:.1f} Mbps]"


def _wins(seed: int) -> bool:
    env = GoopyEnv()
    obs, _ = env.reset(seed=seed)
    while True:
        obs, _, term, trunc, info = env.step(heuristic_policy(obs))
        if term or trunc:
            return bool(info["won"])


if __name__ == "__main__":
    main()
