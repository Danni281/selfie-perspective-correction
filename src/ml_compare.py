"""Phase 9 (ML extension): MediaPipe depth vs Depth Anything V2 depth.

Runs our dense perspective correction on the same input image twice:
    1. With the default sparse-landmark depth (MediaPipe z interpolated
       over the Delaunay mesh).
    2. With a dense depth map predicted by Depth Anything V2 (Yang et al.,
       NeurIPS 2024) substituted in via warp.dense_perspective_correct's
       new depth_override path.

Output figure has four panels:
    [raw | MediaPipe-depth corrected | ML-depth corrected | depth viz]
plus a small printout of post-correction nose_w / face_w for each variant.

Usage:
    # single image
    python src/ml_compare.py image captures/<ts>_raw.png

    # a whole directory of raws
    python src/ml_compare.py batch captures/

    # one image from CMDP at each distance, for the report
    python src/ml_compare.py cmdp-grid --subject 0

Hardware: works on CPU but ~500ms/frame; on the 5070 Ti expect ~10ms.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from main import landmarks_to_xyz, make_face_mesh
from metrics import compute_metrics
import warp as warp_mod

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA)
    return out


def _measure(img: np.ndarray, fm) -> tuple[np.ndarray, object] | tuple[None, None]:
    res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None, None
    pts = landmarks_to_xyz(res.multi_face_landmarks[0],
                           img.shape[1], img.shape[0])
    return pts, compute_metrics(pts[:, :2])


def compare_one(img_path: Path, est, alpha: float = 2.0,
                feather: float = 30.0) -> dict | None:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[skip] {img_path}: cv2.imread failed", file=sys.stderr)
        return None

    fm = make_face_mesh(refine=True)
    try:
        pts, m_raw = _measure(img, fm)
        if pts is None:
            print(f"[skip] {img_path.name}: no face detected", file=sys.stderr)
            return None

        t0 = time.perf_counter()
        out_mp, _ = warp_mod.correct(img, pts, mode="dense",
                                     alpha=alpha, feather=feather)
        t_mp = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        depth_res = est.estimate(img)
        t_inf = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        out_ml, _ = warp_mod.correct(img, pts, mode="dense",
                                     alpha=alpha, feather=feather,
                                     depth_override=depth_res.depth_px)
        t_ml = (time.perf_counter() - t0) * 1000

        # post-correction landmark ratios
        _, m_after_mp = _measure(out_mp, fm)
        _, m_after_ml = _measure(out_ml, fm)
    finally:
        fm.close()

    from ml_depth import colorize_depth
    depth_viz = colorize_depth(depth_res.depth_px)

    panels = [
        _label(img, "raw"),
        _label(out_mp, f"MediaPipe depth  warp={t_mp:.0f}ms"),
        _label(out_ml, f"ML depth  inf={t_inf:.0f}ms warp={t_ml:.0f}ms"),
        _label(depth_viz, "ML depth viz"),
    ]
    figure = np.hstack(panels)

    return {
        "image": str(img_path),
        "raw_nose_ratio": m_raw.nose_w_over_face_w,
        "mp_post_nose_ratio": m_after_mp.nose_w_over_face_w if m_after_mp else None,
        "ml_post_nose_ratio": m_after_ml.nose_w_over_face_w if m_after_ml else None,
        "ml_inference_ms": t_inf,
        "mp_warp_ms": t_mp,
        "ml_warp_ms": t_ml,
        "figure": figure,
    }


def cmd_image(args: argparse.Namespace) -> None:
    from ml_depth import MLDepthEstimator
    est = MLDepthEstimator(size=args.model_size, device=args.device)
    print(f"[ml] {est.model_id} on {est.device}")
    r = compare_one(Path(args.input), est, alpha=args.alpha, feather=args.feather)
    if r is None:
        sys.exit(1)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ml_compare_{Path(args.input).stem}.png"
    cv2.imwrite(str(out_path), r["figure"])
    print(f"  raw nose_w/face_w           = {r['raw_nose_ratio']:.4f}")
    print(f"  after MediaPipe-depth corr  = {r['mp_post_nose_ratio']:.4f}")
    print(f"  after ML-depth corr         = {r['ml_post_nose_ratio']:.4f}")
    print(f"  saved {out_path}")


def cmd_batch(args: argparse.Namespace) -> None:
    from ml_depth import MLDepthEstimator
    est = MLDepthEstimator(size=args.model_size, device=args.device)
    print(f"[ml] {est.model_id} on {est.device}")
    root = Path(args.input)
    paths = sorted(p for p in root.glob("*_raw.png"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        sys.exit(f"no *_raw.png under {root}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, p in enumerate(paths, 1):
        print(f"  [{i}/{len(paths)}] {p.name}", flush=True)
        r = compare_one(p, est, alpha=args.alpha, feather=args.feather)
        if r is None:
            continue
        out_path = RESULTS_DIR / f"ml_compare_{p.stem}.png"
        cv2.imwrite(str(out_path), r["figure"])
        rows.append({k: v for k, v in r.items() if k != "figure"})

    if not rows:
        sys.exit("no comparisons produced")

    raw_avg = float(np.mean([r["raw_nose_ratio"] for r in rows]))
    mp_avg = float(np.mean([r["mp_post_nose_ratio"] for r in rows]))
    ml_avg = float(np.mean([r["ml_post_nose_ratio"] for r in rows]))
    inf_avg = float(np.mean([r["ml_inference_ms"] for r in rows]))
    target = 0.22  # portrait reference
    print(f"\n=== {len(rows)} images, target = {target:.3f} ===")
    print(f"  raw         {raw_avg:.4f}")
    print(f"  MediaPipe   {mp_avg:.4f}   gap closed {(raw_avg-mp_avg)/(raw_avg-target)*100:5.1f}%")
    print(f"  ML depth    {ml_avg:.4f}   gap closed {(raw_avg-ml_avg)/(raw_avg-target)*100:5.1f}%")
    print(f"  ML inference avg {inf_avg:.1f} ms  ({est.device})")


def cmd_cmdp_grid(args: argparse.Namespace) -> None:
    """Build one subject's correction grid across distances for the report."""
    import scipy.io as sio
    from ml_depth import MLDepthEstimator
    cmdp = ROOT / "data" / "cmdp"
    dinfo = sio.loadmat(cmdp / "CMDP-ANNO" / "dinfo.mat",
                        squeeze_me=True, struct_as_record=False)
    paths = dinfo["imagePathsOriginal"]
    distances = list(dinfo["distancesVec_FT"])
    img_root = cmdp / "images"

    est = MLDepthEstimator(size=args.model_size, device=args.device)
    print(f"[ml] {est.model_id} on {est.device}")

    panels: list[np.ndarray] = []
    for d_idx, dft in enumerate(distances):
        rel = str(paths[args.subject, d_idx])
        for sub in ("CMDP_1", "CMDP_2"):
            p = img_root / sub / rel
            if p.exists():
                break
        else:
            print(f"  miss {rel}", file=sys.stderr); continue
        r = compare_one(p, est, alpha=args.alpha, feather=args.feather)
        if r is None:
            continue
        labeled = _label(r["figure"], f"{int(dft)} ft")
        panels.append(labeled)

    if not panels:
        sys.exit("no panels built")
    h = min(p.shape[0] for p in panels)
    panels = [p[:h] for p in panels]
    grid = np.vstack(panels)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ml_compare_cmdp_subject{args.subject}.png"
    cv2.imwrite(str(out_path), grid)
    print(f"saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="MediaPipe-depth vs ML-depth")
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--feather", type=float, default=30.0)
    p.add_argument("--model-size", default="small",
                   choices=["small", "base", "large"])
    p.add_argument("--device", default=None,
                   help="cuda / cpu / mps. Default auto-detect")

    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("image"); pi.add_argument("input")
    pi.set_defaults(func=cmd_image)
    pb = sub.add_parser("batch"); pb.add_argument("input")
    pb.add_argument("--limit", type=int, default=None)
    pb.set_defaults(func=cmd_batch)
    pg = sub.add_parser("cmdp-grid")
    pg.add_argument("--subject", type=int, default=0)
    pg.set_defaults(func=cmd_cmdp_grid)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
