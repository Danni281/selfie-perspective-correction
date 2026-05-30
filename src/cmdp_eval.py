"""Phase 8.5: CMDP evaluation.

Caltech Multi-Distance Portraits (Burgos-Artizzu et al. 2014) provides 51
subjects photographed at 7 distances (2, 3, 4, 6, 8, 12, 16 ft). The 16 ft
image of each subject is the closest thing to a perspective-distortion-
free ground truth available; closer distances exhibit progressively more
distortion.

This script:
  1. For each (subject, distance) image: run MediaPipe Face Mesh, compute
     our landmark-ratio metrics. Call this `raw`.
  2. Run dense perspective correction on each raw image, re-run MediaPipe
     on the corrected output, recompute metrics. Call this `corrected`.
  3. Aggregate by distance. The 16 ft ratios are the per-subject GT target.
  4. Plot mean nose_w/face_w across distances:
        - raw curve: shows perspective inflates the metric at close range
        - corrected curve: shows our correction pulls metrics toward GT
        - GT line: the 16 ft mean ratio
  5. Quantify: for each distance, what fraction of the raw-vs-GT gap does
     the correction close?

Output:
  results/cmdp_eval.png       three-panel plot (raw vs corrected vs GT)
  results/cmdp_eval.json      per-distance summary numbers
  results/cmdp_eval_raw.csv   per-image raw rows
  results/cmdp_eval_corr.csv  per-image corrected rows
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from main import landmarks_to_xyz, make_face_mesh
from metrics import compute_metrics
import warp as warp_mod

ROOT = Path(__file__).resolve().parents[1]
CMDP_DIR = ROOT / "data" / "cmdp"
IMG_DIR = CMDP_DIR / "images"
RESULTS_DIR = ROOT / "results"


def find_image(path_str: str) -> Path | None:
    """CMDP_1.zip and CMDP_2.zip both unpack into `images/`; the subject
    subdir lives under either CMDP_1/ or CMDP_2/. Try both."""
    for sub in ("CMDP_1", "CMDP_2"):
        p = IMG_DIR / sub / path_str
        if p.exists():
            return p
    return None


def measure(img: np.ndarray, fm) -> dict | None:
    res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    h, w = img.shape[:2]
    pts_xyz = landmarks_to_xyz(res.multi_face_landmarks[0], w, h)
    m = compute_metrics(pts_xyz[:, :2])
    if m is None:
        return None
    return {"metrics": m, "pts_xyz": pts_xyz}


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 8.5 CMDP evaluation")
    p.add_argument("--alpha", type=float, default=2.0,
                   help="dense correction alpha (Phase 3 hyperparam)")
    p.add_argument("--max-subjects", type=int, default=None,
                   help="cap for quick smoke-tests; default = all 51")
    p.add_argument("--out-prefix", default="cmdp_eval")
    p.add_argument("--ml-depth", action="store_true",
                   help="ALSO run a Depth Anything V2 variant per image and "
                        "add it to the comparison plot. Needs torch + "
                        "transformers; slow on CPU, fast on a CUDA GPU.")
    p.add_argument("--ml-model-size", default="small",
                   choices=["small", "base", "large"])
    p.add_argument("--ml-device", default=None,
                   help="cuda / cpu / mps; default auto")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dinfo = sio.loadmat(CMDP_DIR / "CMDP-ANNO" / "dinfo.mat",
                        squeeze_me=True, struct_as_record=False)
    distances_ft = list(dinfo["distancesVec_FT"])
    paths = dinfo["imagePathsOriginal"]  # (51, 7)
    n_subjects = int(dinfo["numSubjects"])
    if args.max_subjects:
        n_subjects = min(n_subjects, args.max_subjects)

    fm = make_face_mesh(refine=True)

    ml_est = None
    if args.ml_depth:
        from ml_depth import MLDepthEstimator
        ml_est = MLDepthEstimator(size=args.ml_model_size, device=args.ml_device)
        print(f"[ml] {ml_est.model_id} on {ml_est.device}", flush=True)

    raw_rows: list[dict] = []
    corr_rows: list[dict] = []
    ml_rows: list[dict] = []
    n_attempted = 0
    n_detected = 0
    n_corrected = 0
    n_ml = 0

    t0 = time.perf_counter()
    for s in range(n_subjects):
        for d, dft in enumerate(distances_ft):
            n_attempted += 1
            rel = str(paths[s, d])
            img_path = find_image(rel)
            if img_path is None:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            raw = measure(img, fm)
            if raw is None:
                continue
            n_detected += 1
            mr = raw["metrics"]
            raw_rows.append({
                "subject": s, "distance_ft": int(dft),
                "image": rel,
                "ipd_over_face_w": mr.ipd_over_face_w,
                "nose_w_over_face_w": mr.nose_w_over_face_w,
                "nose_chin_over_face_w": mr.nose_chin_over_face_w,
                "ear_ear_over_face_h": mr.ear_ear_over_face_h,
            })

            # apply correction and remeasure
            try:
                corrected, _ = warp_mod.correct(
                    img, raw["pts_xyz"], mode="dense",
                    alpha=args.alpha, feather=20.0,
                )
            except Exception as e:
                continue
            cm = measure(corrected, fm)
            if cm is None:
                continue
            n_corrected += 1
            mc = cm["metrics"]
            corr_rows.append({
                "subject": s, "distance_ft": int(dft),
                "image": rel,
                "ipd_over_face_w": mc.ipd_over_face_w,
                "nose_w_over_face_w": mc.nose_w_over_face_w,
                "nose_chin_over_face_w": mc.nose_chin_over_face_w,
                "ear_ear_over_face_h": mc.ear_ear_over_face_h,
            })

            if ml_est is not None:
                try:
                    depth_res = ml_est.estimate(img)
                    ml_out, _ = warp_mod.correct(
                        img, raw["pts_xyz"], mode="dense",
                        alpha=args.alpha, feather=20.0,
                        depth_override=depth_res.depth_px,
                    )
                    ml_meas = measure(ml_out, fm)
                    if ml_meas is not None:
                        mm = ml_meas["metrics"]
                        ml_rows.append({
                            "subject": s, "distance_ft": int(dft),
                            "image": rel,
                            "nose_w_over_face_w": mm.nose_w_over_face_w,
                        })
                        n_ml += 1
                except Exception as e:
                    print(f"  [ml-skip] {rel}: {e}", file=sys.stderr)
        if (s + 1) % 5 == 0 or s + 1 == n_subjects:
            dt = time.perf_counter() - t0
            print(f"  subject {s+1}/{n_subjects}  "
                  f"({n_detected} detected, {n_corrected} corrected, "
                  f"{dt:.1f}s)", flush=True)
    fm.close()

    print(f"\n[done] attempted {n_attempted}, MediaPipe detected "
          f"{n_detected}, also corrected {n_corrected}"
          + (f", ML depth corrected {n_ml}" if ml_est is not None else ""))

    # --- write per-image CSVs ---
    fields = ["subject", "distance_ft", "image", "ipd_over_face_w",
              "nose_w_over_face_w", "nose_chin_over_face_w",
              "ear_ear_over_face_h"]
    for tag, rows in (("raw", raw_rows), ("corr", corr_rows)):
        path = RESULTS_DIR / f"{args.out_prefix}_{tag}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
        print(f"[csv] {path}")

    # --- aggregate ---
    raw_by_d = {int(dft): [] for dft in distances_ft}
    corr_by_d = {int(dft): [] for dft in distances_ft}
    ml_by_d = {int(dft): [] for dft in distances_ft}
    for r in raw_rows:
        raw_by_d[r["distance_ft"]].append(r["nose_w_over_face_w"])
    for r in corr_rows:
        corr_by_d[r["distance_ft"]].append(r["nose_w_over_face_w"])
    for r in ml_rows:
        ml_by_d[r["distance_ft"]].append(r["nose_w_over_face_w"])

    summary = {"distance_ft": [int(d) for d in distances_ft],
               "raw_mean": [], "raw_std": [],
               "corr_mean": [], "corr_std": [],
               "ml_mean": [], "ml_std": [],
               "n_per_distance": []}
    for dft in distances_ft:
        ks = int(dft)
        rs = raw_by_d[ks]
        cs = corr_by_d[ks]
        ms = ml_by_d[ks]
        summary["raw_mean"].append(float(np.mean(rs)) if rs else float("nan"))
        summary["raw_std"].append(float(np.std(rs)) if rs else float("nan"))
        summary["corr_mean"].append(float(np.mean(cs)) if cs else float("nan"))
        summary["corr_std"].append(float(np.std(cs)) if cs else float("nan"))
        summary["ml_mean"].append(float(np.mean(ms)) if ms else float("nan"))
        summary["ml_std"].append(float(np.std(ms)) if ms else float("nan"))
        summary["n_per_distance"].append(len(rs))

    # ground truth = mean ratio at the farthest distance (16 ft)
    gt = summary["raw_mean"][-1]
    summary["gt_nose_w_over_face_w"] = gt

    # "gap-closed" per distance: 1.0 means we hit GT, 0 means we didn't move
    def _gap(raw_m, corr_m):
        gap = raw_m - gt
        if abs(gap) < 1e-6:
            return 0.0
        return float((raw_m - corr_m) / gap)
    gaps_closed = [_gap(r, c) for r, c in
                   zip(summary["raw_mean"], summary["corr_mean"])]
    gaps_closed_ml = [_gap(r, m) for r, m in
                      zip(summary["raw_mean"], summary["ml_mean"])]
    summary["gap_closed_fraction"] = gaps_closed
    summary["gap_closed_fraction_ml"] = gaps_closed_ml

    json_path = RESULTS_DIR / f"{args.out_prefix}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[json] {json_path}")

    # --- plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    x = summary["distance_ft"]
    ax1.errorbar(x, summary["raw_mean"], yerr=summary["raw_std"],
                 marker="o", capsize=3, label="raw", color="#888888",
                 linewidth=2)
    ax1.errorbar(x, summary["corr_mean"], yerr=summary["corr_std"],
                 marker="s", capsize=3,
                 label=f"MediaPipe depth (alpha={args.alpha})",
                 color="#d62728", linewidth=2)
    if ml_est is not None and any(not np.isnan(v) for v in summary["ml_mean"]):
        ax1.errorbar(x, summary["ml_mean"], yerr=summary["ml_std"],
                     marker="^", capsize=3,
                     label=f"ML depth (alpha={args.alpha})",
                     color="#1f77b4", linewidth=2)
    ax1.axhline(gt, color="black", linestyle="--",
                label=f"GT (16ft mean = {gt:.3f})")
    ax1.set_xlabel("camera-to-subject distance (ft)")
    ax1.set_ylabel("nose_w / face_w")
    ax1.set_title("CMDP: perspective distortion vs. distance")
    ax1.grid(alpha=0.3); ax1.legend()
    ax1.invert_xaxis()  # distortion strongest on the left

    if ml_est is not None and any(not np.isnan(v) for v in summary["ml_mean"]):
        # grouped bars
        xs = np.arange(len(x))
        bar_w = 0.4
        ax2.bar(xs - bar_w/2, [g * 100 for g in gaps_closed], bar_w,
                color="#d62728", label="MediaPipe")
        ax2.bar(xs + bar_w/2, [g * 100 for g in gaps_closed_ml], bar_w,
                color="#1f77b4", label="ML depth")
        ax2.set_xticks(xs)
        ax2.set_xticklabels([str(int(d)) for d in x])
        ax2.legend()
    else:
        ax2.bar(x, [g * 100 for g in gaps_closed],
                color=["#d62728" if g > 0 else "#888888" for g in gaps_closed])
        ax2.invert_xaxis()
        for xi, g in zip(x, gaps_closed):
            ax2.text(xi, g * 100, f"{g*100:.0f}%", ha="center",
                     va="bottom" if g >= 0 else "top", fontsize=9)
    ax2.axhline(0, color="black", linewidth=0.7)
    ax2.set_xlabel("distance (ft)")
    ax2.set_ylabel("% of raw->GT gap closed")
    ax2.set_title(f"alpha={args.alpha} correction effectiveness")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    png = RESULTS_DIR / f"{args.out_prefix}.png"
    fig.savefig(png, dpi=140)
    print(f"[png]  {png}")

    print("\n=== summary ===")
    if ml_est is not None:
        print(f"{'dist':>5} {'n':>4} {'raw':>8} {'mp':>8} {'ml':>8} "
              f"{'GT':>8} {'mp%':>7} {'ml%':>7}")
        for i, dft in enumerate(distances_ft):
            print(f"{int(dft):>5} {summary['n_per_distance'][i]:>4} "
                  f"{summary['raw_mean'][i]:>8.4f} "
                  f"{summary['corr_mean'][i]:>8.4f} "
                  f"{summary['ml_mean'][i]:>8.4f} "
                  f"{gt:>8.4f} {gaps_closed[i]*100:>6.1f}% "
                  f"{gaps_closed_ml[i]*100:>6.1f}%")
    else:
        print(f"{'dist':>5} {'n':>4} {'raw':>8} {'corr':>8} {'GT':>8} {'gap%':>8}")
        for i, dft in enumerate(distances_ft):
            print(f"{int(dft):>5} {summary['n_per_distance'][i]:>4} "
                  f"{summary['raw_mean'][i]:>8.4f} "
                  f"{summary['corr_mean'][i]:>8.4f} "
                  f"{gt:>8.4f} {gaps_closed[i]*100:>7.1f}%")


if __name__ == "__main__":
    main()
