"""Re-record milestone videos that were made before videos replayed the milestone episode.

Before commit 1e98f68 every milestone video replayed eval seed 10000, so e.g. a "first_win"
video could show a loss. For each such entry in ``videos/index.jsonl`` this re-evaluates the
checkpoint saved at that eval, picks the winning (or furthest-reaching) episode and records it
again under the same file name. Entries already carrying ``win_seed``/``best_seed`` are skipped.

    python scripts/rerender_milestones.py runs/ppo_curriculum [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cuphead_rl.train.agent import AgentPolicy, load_model
from cuphead_rl.train.config import ExperimentConfig
from cuphead_rl.train.evaluate import evaluate, record_video


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.run_dir / "config.yaml")
    index_path = args.run_dir / "videos" / "index.jsonl"
    entries = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    for e in entries:
        if "best_seed" in e["eval"]:
            continue
        ckpt = args.run_dir / "checkpoints" / f"step_{e['steps']:010d}.zip"
        if not ckpt.exists():
            print(f"skip {e['file']}: no checkpoint {ckpt.name}")
            continue
        model = load_model(ckpt)
        stats = evaluate(AgentPolicy(model, frame_stack=cfg.frame_stack, introspect=False), cfg,
                         e["eval"]["episodes"], start_phase=0)
        seed = stats["win_seed"] if stats["win_seed"] is not None else stats["best_seed"]
        print(f"{e['file']}: eval win={stats['win_rate']:.2f} (logged {e['eval']['win_rate']:.2f}) -> seed {seed}")
        if args.dry_run:
            continue
        title = f"{cfg.name} | {e['steps'] / 1e6:.2f}M steps | {', '.join(e['tags'])}"
        vstats = record_video(AgentPolicy(model, frame_stack=cfg.frame_stack), cfg, args.run_dir / "videos" / e["file"],
                              seed=seed, title=title, start_phase=0)
        e["eval"].update(best_seed=stats["best_seed"], win_seed=stats["win_seed"])
        e["episode"] = vstats
        e["rerendered"] = True
        print(f"  video: won={vstats['won']} phase={vstats['phase']} time={vstats['time']:.1f}s")
    if not args.dry_run:
        index_path.write_text("".join(json.dumps(e, default=str) + "\n" for e in entries))


if __name__ == "__main__":
    main()
