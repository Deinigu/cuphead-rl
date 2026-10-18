"""Learning-curve figures for the retrospective / video.

    python -m cuphead_rl.train.plots runs/ppo_baseline runs/ppo_curriculum -o docs/figures/curves.png [--dark]

One panel per measure (never a dual axis); experiments keep a fixed colour slot in the
order given on the command line, are direct-labelled at the line end and listed in a legend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (fixed order, never cycled) + surfaces/ink per mode.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#33322f",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]),
}
PANELS = (("win_rate", "Win rate", True), ("boss_damage", "Boss HP removed", True), ("mean_phase", "Mean phase reached", False))


def load_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plot_runs(run_dirs: list[Path], out: Path, dark: bool = False, labels: list[str] | None = None) -> None:
    th = THEMES["dark" if dark else "light"]
    if len(run_dirs) > len(th["series"]):
        raise ValueError("more than 8 experiments: facet them into several figures instead")
    runs = [(labels[i] if labels else d.name, load_metrics(d), th["series"][i]) for i, d in enumerate(run_dirs)]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(19.2, 6.0), dpi=100)
    fig.patch.set_facecolor(th["surface"])
    for ax, (key, title, is_frac) in zip(axes, PANELS):
        ax.set_facecolor(th["surface"])
        ends = []
        for name, rows, color in runs:
            xs = [r["steps"] / 1e6 for r in rows]
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, color=color, linewidth=2, label=name, solid_capstyle="round")
            if xs:
                ends.append([ys[-1], xs[-1], name])
        # Direct labels at line ends, pushed apart vertically so they never overlap.
        span = 1.0 if is_frac else 2.0
        ends.sort(key=lambda e: e[0])
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.07 * span)
        if ends and ends[-1][0] > span:  # pushed past the top: pack downward from the top instead
            ends[-1][0] = span
            for i in range(len(ends) - 2, -1, -1):
                ends[i][0] = min(ends[i][0], ends[i + 1][0] - 0.07 * span)
        for y, x, name in ends:
            ax.annotate(name, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=11, color=th["ink2"], annotation_clip=False)
        ax.set_title(title, color=th["ink"], loc="left", fontsize=15, pad=10)
        ax.set_xlabel("Environment steps (millions)", color=th["ink2"])
        if is_frac:
            ax.set_ylim(-0.02, 1.02)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        else:
            ax.set_ylim(-0.05, 2.05)
            ax.set_yticks([0, 1, 2], ["Phase 1", "Phase 2", "Phase 3"])
        ax.grid(True, color=th["grid"], linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(th["grid"])
        ax.tick_params(colors=th["ink2"])
    if len(runs) > 1:
        # One legend row under all panels, so it never covers a curve.
        handles, names = axes[0].get_legend_handles_labels()
        leg = fig.legend(handles, names, loc="lower center", ncol=len(runs), frameon=False)
        for t in leg.get_texts():
            t.set_color(th["ink"])
        fig.tight_layout(rect=(0, 0.07, 1, 1))
    else:
        fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=th["surface"])
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--dark", action="store_true")
    a = ap.parse_args()
    plot_runs(a.runs, a.out, dark=a.dark, labels=a.labels)


if __name__ == "__main__":
    main()
