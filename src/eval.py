"""Phase 8: end-to-end evaluation script.

Takes a recorded clip and reports, per variant:
  - average pipeline FPS and average per-stage latency (mediapipe / warp / total)
  - landmark ratios before vs after correction (mean across detected frames)
  - frame-to-frame landmark L2 delta (mean across landmarks)
        - raw         : MediaPipe with no smoothing
        - per-frame   : raw correction applied per-frame (still no smoothing)
        - smoothed    : one row per available smoother (ema, kalman, oneeuro)

Dumps results to a JSON file and renders two matplotlib plots:
  results/<prefix>_latency.png   bar chart of per-stage timing per variant
  results/<prefix>_jitter.png    bar chart of frame-to-frame landmark delta

Usage:
    python src/eval.py clips/raw_*.mp4 --out-prefix eval_default
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from main import landmarks_to_xyz, make_face_mesh
from metrics import compute_metrics
from smoothing import make_smoother
import warp as warp_mod

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def run_variant(clip_path: Path, *, smoother_name: str,
                correct: bool, warp_kwargs: dict) -> dict:
    """One full pass over `clip_path` with the given smoother + correction
    setting. Returns per-frame timings, landmarks, and metrics."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open {clip_path}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fm = make_face_mesh(refine=True)
    smoother = make_smoother(smoother_name) if smoother_name != "none" else None

    mp_ms_list: list[float] = []
    warp_ms_list: list[float] = []
    total_ms_list: list[float] = []
    landmark_traj = np.full((n_frames_hint, 478, 3), np.nan, dtype=np.float32)
    pre_ratios: list[float] = []  # nose_w/face_w before correction
    post_ratios: list[float] = []  # ... after correction
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()

        t_mp = time.perf_counter()
        res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_ms = (time.perf_counter() - t_mp) * 1000

        pts_xyz = None
        if res.multi_face_landmarks:
            pts_xyz = landmarks_to_xyz(res.multi_face_landmarks[0],
                                       frame.shape[1], frame.shape[0])
            if smoother is not None:
                pts_xyz = smoother.update(pts_xyz, frame_idx / fps_in)
            landmark_traj[frame_idx] = pts_xyz
        elif smoother is not None:
            smoother.reset()

        warp_ms = 0.0
        if correct and pts_xyz is not None and pts_xyz.shape[0] >= 478:
            mr = compute_metrics(pts_xyz[:, :2])
            if mr is not None:
                pre_ratios.append(mr.nose_w_over_face_w)
            t_w = time.perf_counter()
            corrected, _ = warp_mod.correct(frame, pts_xyz, **warp_kwargs)
            warp_ms = (time.perf_counter() - t_w) * 1000
            # post-correction landmark detection for the ratio compare
            res2 = fm.process(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
            if res2.multi_face_landmarks:
                pts2 = landmarks_to_xyz(res2.multi_face_landmarks[0],
                                        corrected.shape[1], corrected.shape[0])
                mc = compute_metrics(pts2[:, :2])
                if mc is not None:
                    post_ratios.append(mc.nose_w_over_face_w)
        elif pts_xyz is not None:
            mr = compute_metrics(pts_xyz[:, :2])
            if mr is not None:
                pre_ratios.append(mr.nose_w_over_face_w)

        total_ms = (time.perf_counter() - t0) * 1000
        mp_ms_list.append(mp_ms)
        warp_ms_list.append(warp_ms)
        total_ms_list.append(total_ms)
        frame_idx += 1
    cap.release()
    fm.close()

    landmark_traj = landmark_traj[:frame_idx]
    # mean frame-to-frame landmark L2
    if landmark_traj.shape[0] >= 2:
        diff = np.diff(landmark_traj, axis=0)
        per_lm = np.linalg.norm(diff, axis=2)  # (T-1, N)
        jitter_per_frame = np.nanmean(per_lm, axis=1)
        jitter_mean = float(np.nanmean(jitter_per_frame))
    else:
        jitter_mean = float("nan")

    return {
        "n_frames": int(frame_idx),
        "mean_mp_ms": float(np.mean(mp_ms_list)),
        "mean_warp_ms": float(np.mean(warp_ms_list)),
        "mean_total_ms": float(np.mean(total_ms_list)),
        "p95_total_ms": float(np.percentile(total_ms_list, 95)),
        "fps_mean": 1000.0 / float(np.mean(total_ms_list)),
        "pre_ratio_mean": float(np.mean(pre_ratios)) if pre_ratios else None,
        "post_ratio_mean": float(np.mean(post_ratios)) if post_ratios else None,
        "frame_to_frame_jitter_mean_px": jitter_mean,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 8 end-to-end evaluation")
    p.add_argument("clip", help="path to a recorded raw mp4 to evaluate on")
    p.add_argument("--out-prefix", default="eval")
    p.add_argument("--mode", default="dense")
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--feather", type=float, default=30.0)
    p.add_argument("--depth-downsample", type=int, default=2)
    p.add_argument("--process-downsample", type=int, default=1)
    args = p.parse_args()

    clip = Path(args.clip)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    warp_kwargs = dict(mode=args.mode, alpha=args.alpha,
                       feather=args.feather,
                       depth_downsample=args.depth_downsample,
                       process_downsample=args.process_downsample)

    plan = [
        ("raw",          dict(smoother_name="none",    correct=False)),
        ("per_frame",    dict(smoother_name="none",    correct=True)),
        ("ema",          dict(smoother_name="ema",     correct=True)),
        ("kalman",       dict(smoother_name="kalman",  correct=True)),
        ("oneeuro",      dict(smoother_name="oneeuro", correct=True)),
    ]
    results: dict[str, dict] = {}
    print(f"[eval] clip={clip}  mode={args.mode} alpha={args.alpha} "
          f"ds={args.depth_downsample} ps={args.process_downsample}")
    for name, kwargs in plan:
        print(f"  running variant: {name}")
        r = run_variant(clip, warp_kwargs=warp_kwargs, **kwargs)
        results[name] = r
        print(f"    -> fps={r['fps_mean']:5.1f}  "
              f"mp={r['mean_mp_ms']:4.1f}ms  warp={r['mean_warp_ms']:4.1f}ms  "
              f"jitter={r['frame_to_frame_jitter_mean_px']:.3f}px")

    json_path = RESULTS_DIR / f"{args.out_prefix}.json"
    json_path.write_text(json.dumps({
        "clip": str(clip),
        "config": vars(args),
        "variants": results,
    }, indent=2))
    print(f"\n[json] {json_path}")

    # --- Plot 1: per-stage latency ---
    names = list(results.keys())
    mp_t = [results[n]["mean_mp_ms"] for n in names]
    warp_t = [results[n]["mean_warp_ms"] for n in names]
    other_t = [results[n]["mean_total_ms"] - results[n]["mean_mp_ms"]
               - results[n]["mean_warp_ms"] for n in names]
    fps = [results[n]["fps_mean"] for n in names]

    fig, (ax_lat, ax_jit) = plt.subplots(1, 2, figsize=(12, 4.5),
                                          gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(names))
    ax_lat.bar(x, mp_t, label="MediaPipe", color="#1f77b4")
    ax_lat.bar(x, warp_t, bottom=mp_t, label="warp", color="#d62728")
    ax_lat.bar(x, other_t, bottom=[a + b for a, b in zip(mp_t, warp_t)],
               label="other", color="#cccccc")
    ax_lat.set_xticks(x); ax_lat.set_xticklabels(names, rotation=0)
    ax_lat.set_ylabel("mean per-frame latency (ms)")
    ax_lat.set_title("Per-stage latency by variant")
    ax_lat.grid(axis="y", alpha=0.3); ax_lat.legend(loc="upper left")
    for i, f in enumerate(fps):
        ax_lat.text(i, mp_t[i] + warp_t[i] + other_t[i],
                    f"{f:.1f} FPS", ha="center", va="bottom", fontsize=9)

    jit = [results[n]["frame_to_frame_jitter_mean_px"] for n in names]
    colors = ["#888", "#999", "#1f77b4", "#2ca02c", "#d62728"]
    ax_jit.bar(names, jit, color=colors)
    ax_jit.set_ylabel("mean landmark L2 (px)")
    ax_jit.set_title("Frame-to-frame landmark jitter")
    ax_jit.grid(axis="y", alpha=0.3)
    for i, v in enumerate(jit):
        ax_jit.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    png = RESULTS_DIR / f"{args.out_prefix}.png"
    fig.savefig(png, dpi=140)
    print(f"[png]  {png}")

    print("\n=== summary ===")
    print(f"{'variant':>10} {'fps':>6} {'mp(ms)':>8} {'warp(ms)':>9} "
          f"{'jitter(px)':>11} {'pre_nose':>9} {'post_nose':>10}")
    for n in names:
        r = results[n]
        pre = f"{r['pre_ratio_mean']:.3f}" if r['pre_ratio_mean'] else "  --"
        post = f"{r['post_ratio_mean']:.3f}" if r['post_ratio_mean'] else "  --"
        print(f"{n:>10} {r['fps_mean']:>6.1f} {r['mean_mp_ms']:>8.2f} "
              f"{r['mean_warp_ms']:>9.2f} "
              f"{r['frame_to_frame_jitter_mean_px']:>11.3f} "
              f"{pre:>9} {post:>10}")


if __name__ == "__main__":
    main()
