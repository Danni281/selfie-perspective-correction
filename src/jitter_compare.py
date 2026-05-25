"""Phase 6 ablation: load smoother landmark logs, compute frame-to-frame
landmark jitter, write a comparison plot + summary JSON.

Usage:
    python src/jitter_compare.py none ema kalman oneeuro

Each name maps to results/<name>_landmarks.npy produced by process_clip.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def frame_jitter(traj: np.ndarray) -> np.ndarray:
    """L2 frame-to-frame landmark displacement averaged across landmarks.

    traj: (T, N, D)  ->  jitter: (T-1,) in pixels.
    NaN frames (face lost) propagate as NaN so they can be masked.
    """
    diff = np.diff(traj, axis=0)
    per_landmark = np.linalg.norm(diff, axis=2)  # (T-1, N)
    return np.nanmean(per_landmark, axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 6 jitter comparison")
    p.add_argument("variants", nargs="+",
                   help="smoother names (must match results/<name>_landmarks.npy)")
    p.add_argument("--out-prefix", default="phase6_jitter",
                   help="output basename in results/")
    args = p.parse_args()

    summary: dict[str, dict[str, float]] = {}
    traces: dict[str, np.ndarray] = {}

    for name in args.variants:
        path = RESULTS_DIR / f"{name}_landmarks.npy"
        if not path.exists():
            print(f"[skip] {path} missing", file=sys.stderr)
            continue
        traj = np.load(path)
        j = frame_jitter(traj)
        traces[name] = j
        summary[name] = {
            "mean_jitter_px": float(np.nanmean(j)),
            "median_jitter_px": float(np.nanmedian(j)),
            "p95_jitter_px": float(np.nanpercentile(j, 95)),
            "n_frames": int(traj.shape[0]),
        }

    if not traces:
        sys.exit("no inputs found")

    print(f"{'variant':>12} {'mean':>8} {'median':>8} {'p95':>8}  (px)")
    for name, s in summary.items():
        print(f"{name:>12} {s['mean_jitter_px']:>8.3f} "
              f"{s['median_jitter_px']:>8.3f} {s['p95_jitter_px']:>8.3f}")

    # Trace plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5),
                                   gridspec_kw={"width_ratios": [3, 1]})
    colors = {"none": "#888888", "ema": "#1f77b4",
              "kalman": "#2ca02c", "oneeuro": "#d62728"}
    for name, j in traces.items():
        ax1.plot(j, label=name, color=colors.get(name), linewidth=1.2,
                 alpha=0.9)
    ax1.set_xlabel("frame index")
    ax1.set_ylabel("mean landmark L2 jitter (px)")
    ax1.set_title("Per-frame landmark jitter (lower = more stable)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    # Bar summary
    names = list(summary.keys())
    means = [summary[n]["mean_jitter_px"] for n in names]
    bar_colors = [colors.get(n, "#999") for n in names]
    ax2.bar(names, means, color=bar_colors)
    ax2.set_ylabel("mean jitter (px)")
    ax2.set_title("Mean jitter")
    for i, m in enumerate(means):
        ax2.text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    png = RESULTS_DIR / f"{args.out_prefix}.png"
    fig.savefig(png, dpi=140)
    print(f"\n[saved] {png}")

    js = RESULTS_DIR / f"{args.out_prefix}.json"
    js.write_text(json.dumps(summary, indent=2))
    print(f"[saved] {js}")


if __name__ == "__main__":
    main()
