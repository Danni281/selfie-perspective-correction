"""Phase 5/6: offline processing of a recorded clip.

Usage:
    # record a 10s webcam clip into clips/raw_<ts>.mp4
    python src/process_clip.py record --seconds 10

    # produce baseline.mp4 (per-frame correction, no smoothing)
    python src/process_clip.py run clips/raw_*.mp4 \
        --mode dense --alpha 2.0 --smooth none --out baseline

    # produce smoothed.mp4 with each smoother
    python src/process_clip.py run clips/raw_*.mp4 \
        --mode dense --alpha 2.0 --smooth ema    --out ema
    python src/process_clip.py run clips/raw_*.mp4 \
        --mode dense --alpha 2.0 --smooth kalman --out kalman
    python src/process_clip.py run clips/raw_*.mp4 \
        --mode dense --alpha 2.0 --smooth oneeuro --out oneeuro
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from main import landmarks_to_xyz, make_face_mesh
from metrics import CSV_FIELDS, compute_metrics, csv_row
from smoothing import SMOOTHERS, make_smoother
import warp as warp_mod

ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = ROOT / "clips"
RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

def cmd_record(args: argparse.Namespace) -> None:
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("could not open camera; grant Terminal Camera access")

    fps_target = args.fps
    out_path = CLIPS_DIR / f"raw_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_target,
                             (args.width, args.height))

    target_frames = int(args.seconds * fps_target)
    print(f"[record] recording {args.seconds}s ({target_frames} frames) "
          f"to {out_path}", flush=True)
    window = "record (press q to stop early)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)

    frame_idx = 0
    t0 = time.perf_counter()
    try:
        while frame_idx < target_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            writer.write(frame)
            cv2.putText(frame,
                        f"{frame_idx+1}/{target_frames}  "
                        f"{time.perf_counter()-t0:4.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame,
                        f"{frame_idx+1}/{target_frames}  "
                        f"{time.perf_counter()-t0:4.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(window, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
    print(f"[record] wrote {frame_idx} frames to {out_path}", flush=True)


# ---------------------------------------------------------------------------
# run pipeline on a clip
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    in_path = Path(args.input)
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open {in_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_stem = args.out or f"{args.mode}_{args.smooth}"
    out_path = RESULTS_DIR / f"{out_stem}.mp4"
    csv_path = RESULTS_DIR / f"{out_stem}.csv"
    landmarks_path = RESULTS_DIR / f"{out_stem}_landmarks.npy"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    face_mesh = make_face_mesh(refine=True)
    smoother_kwargs: dict[str, float] = {}
    if args.smooth == "ema":
        smoother_kwargs["alpha"] = args.ema_alpha
    elif args.smooth == "oneeuro":
        smoother_kwargs["min_cutoff"] = args.euro_min_cutoff
        smoother_kwargs["beta"] = args.euro_beta
    smoother = make_smoother(args.smooth, **smoother_kwargs)

    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    csv_w.writeheader()

    landmark_log = np.full((n_frames, 478, 3), np.nan, dtype=np.float32)

    print(f"[run] {in_path.name} -> {out_path.name}  "
          f"({n_frames} frames, mode={args.mode}, smooth={args.smooth})",
          flush=True)

    frame_idx = 0
    warp_total_ms = 0.0
    pipeline_total_ms = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_start = time.perf_counter()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            pts_xyz = None
            if res.multi_face_landmarks:
                pts_xyz = landmarks_to_xyz(res.multi_face_landmarks[0], w, h)
                if smoother is not None:
                    pts_xyz = smoother.update(pts_xyz, frame_idx / fps)
                landmark_log[frame_idx] = pts_xyz
            elif smoother is not None:
                smoother.reset()

            corrected = frame
            if pts_xyz is not None and pts_xyz.shape[0] >= 478 and args.mode != "raw":
                t_w = time.perf_counter()
                corrected, _ = warp_mod.correct(
                    frame, pts_xyz,
                    strength=args.strength, mode=args.mode,
                    feather=args.feather, alpha=args.alpha,
                )
                warp_total_ms += (time.perf_counter() - t_w) * 1000

            writer.write(corrected)
            m = compute_metrics(pts_xyz[:, :2]) if pts_xyz is not None else None
            csv_w.writerow(csv_row(frame_idx, frame_idx / fps,
                                   fps, m))
            pipeline_total_ms += (time.perf_counter() - t_start) * 1000
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  {frame_idx}/{n_frames}  "
                      f"(pipeline avg {pipeline_total_ms/frame_idx:.1f} ms, "
                      f"warp avg {warp_total_ms/max(1,frame_idx):.1f} ms)",
                      flush=True)
    finally:
        cap.release()
        writer.release()
        face_mesh.close()
        csv_f.close()
        np.save(landmarks_path, landmark_log[:frame_idx])

    print(f"[run] done: {frame_idx} frames, "
          f"avg pipeline {pipeline_total_ms/max(1,frame_idx):.1f} ms, "
          f"avg warp {warp_total_ms/max(1,frame_idx):.1f} ms", flush=True)
    print(f"[run]   video     -> {out_path}", flush=True)
    print(f"[run]   metrics   -> {csv_path}", flush=True)
    print(f"[run]   landmarks -> {landmarks_path}", flush=True)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Phase 5/6 offline clip processor")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("record", help="record a webcam clip")
    pr.add_argument("--seconds", type=float, default=10.0)
    pr.add_argument("--fps", type=int, default=30)
    pr.add_argument("--camera", type=int, default=0)
    pr.add_argument("--width", type=int, default=1280)
    pr.add_argument("--height", type=int, default=720)
    pr.set_defaults(func=cmd_record)

    pn = sub.add_parser("run", help="process a clip through the pipeline")
    pn.add_argument("input")
    pn.add_argument("--out", default=None,
                    help="output stem (default = mode_smooth)")
    pn.add_argument("--mode",
                    choices=["raw", "dense", "nose", "perspective", "uniform"],
                    default="dense",
                    help="'raw' skips the warp and just re-encodes (useful for "
                         "sanity-checking landmark detection on a clip)")
    pn.add_argument("--smooth", choices=list(SMOOTHERS.keys()), default="none")
    pn.add_argument("--strength", type=float, default=0.3)
    pn.add_argument("--alpha", type=float, default=2.0)
    pn.add_argument("--feather", type=float, default=30.0)
    pn.add_argument("--ema-alpha", type=float, default=0.7)
    pn.add_argument("--euro-min-cutoff", type=float, default=1.0)
    pn.add_argument("--euro-beta", type=float, default=0.007)
    pn.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
