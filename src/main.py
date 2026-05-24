"""Phase 1+2+3: webcam capture, MediaPipe Face Mesh, distortion metrics,
single-frame perspective correction (Delaunay piecewise-affine warp).

Keys:
  q / Esc  quit
  s        save snapshot (raw, overlay, landmarks; corrected too if 'c' is on)
  m        toggle metrics HUD on/off
  c        toggle correction view (side-by-side raw | corrected)
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from metrics import CSV_FIELDS, compute_metrics, csv_row, hud_text
import warp as warp_mod

CAPTURE_DIR = Path(__file__).resolve().parents[1] / "captures"
METRICS_DIR = Path(__file__).resolve().parents[1] / "metrics_logs"


def make_face_mesh(refine: bool) -> mp.solutions.face_mesh.FaceMesh:
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=refine,  # iris landmarks (468..477) needed for IPD
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def landmarks_to_xyz(landmarks, width: int, height: int) -> np.ndarray:
    """Return (N, 3) array: (x_px, y_px, z_mediapipe_normalized).

    MediaPipe's `z` is the depth at the landmark with the head's center as
    origin, in roughly image-width-normalized units; smaller z = closer to
    the camera.
    """
    pts = np.empty((len(landmarks.landmark), 3), dtype=np.float32)
    for i, lm in enumerate(landmarks.landmark):
        pts[i, 0] = lm.x * width
        pts[i, 1] = lm.y * height
        pts[i, 2] = lm.z * width  # match xy scale
    return pts


def draw_landmarks(frame: np.ndarray, pts: np.ndarray) -> None:
    for x, y in pts.astype(int):
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def _put(frame: np.ndarray, text: str, org: tuple[int, int],
         scale: float = 0.6, color: tuple[int, int, int] = (0, 255, 0)) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, fps: float, face_detected: bool,
             metrics_lines: list[str], show_metrics: bool) -> None:
    _put(frame, f"FPS: {fps:5.1f}  face: {'yes' if face_detected else 'no '}",
         (10, 30), scale=0.7)
    if show_metrics:
        # translucent backing strip so yellow text is readable on any background
        y_top = 50
        line_h = 32
        h = line_h * len(metrics_lines) + 12
        strip = frame[y_top:y_top + h, 0:560]
        if strip.size:
            strip[:] = (strip * 0.35).astype(strip.dtype)
        for i, line in enumerate(metrics_lines):
            _put(frame, line, (10, y_top + 26 + i * line_h), scale=0.7,
                 color=(0, 255, 255))  # BGR yellow
    hint = "q quit | s save | m metrics"
    _put(frame, hint, (10, frame.shape[0] - 12), scale=0.55,
         color=(255, 255, 255))


def save_capture(frame_raw: np.ndarray, frame_annotated: np.ndarray,
                 landmarks_px: np.ndarray | None,
                 frame_corrected: np.ndarray | None = None,
                 mode: str | None = None,
                 strength: float | None = None,
                 uniform_scale: float | None = None,
                 feather: float | None = None) -> Path:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    raw_path = CAPTURE_DIR / f"{stamp}_raw.png"
    overlay_path = CAPTURE_DIR / f"{stamp}_overlay.png"
    npy_path = CAPTURE_DIR / f"{stamp}_landmarks.npy"
    json_path = CAPTURE_DIR / f"{stamp}_meta.json"

    cv2.imwrite(str(raw_path), frame_raw)
    cv2.imwrite(str(overlay_path), frame_annotated)

    meta = {
        "timestamp": stamp,
        "width": int(frame_raw.shape[1]),
        "height": int(frame_raw.shape[0]),
        "num_landmarks": 0 if landmarks_px is None else int(landmarks_px.shape[0]),
        "raw": raw_path.name,
        "overlay": overlay_path.name,
        "landmarks": npy_path.name if landmarks_px is not None else None,
        "corrected": None,
        "sidebyside": None,
        "mode": mode,
        "strength": strength,
        "uniform_scale": uniform_scale,
        "feather": feather,
    }
    if landmarks_px is not None:
        np.save(npy_path, landmarks_px)
    if frame_corrected is not None:
        corrected_path = CAPTURE_DIR / f"{stamp}_corrected.png"
        side_path = CAPTURE_DIR / f"{stamp}_sidebyside.png"
        cv2.imwrite(str(corrected_path), frame_corrected)
        cv2.imwrite(str(side_path), np.hstack([frame_raw, frame_corrected]))
        meta["corrected"] = corrected_path.name
        meta["sidebyside"] = side_path.name
    json_path.write_text(json.dumps(meta, indent=2))
    return raw_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1+2 capture + metrics")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--backend", choices=["avfoundation", "any"],
                   default="avfoundation",
                   help="capture backend (use avfoundation on macOS)")
    p.add_argument("--no-csv", action="store_true",
                   help="skip the per-run metrics CSV log")
    p.add_argument("--strength", type=float, default=0.15,
                   help="z-weighted shrink strength: fraction by which the "
                        "closest-to-camera landmark is pulled toward face "
                        "center; farthest landmark is untouched. 0 disables.")
    p.add_argument("--uniform-scale", type=float, default=None,
                   help="ABLATION: use Phase 3 uniform shrink (e.g. 0.85) "
                        "instead of z-weighted. Overrides --strength.")
    p.add_argument("--feather", type=float, default=30.0,
                   help="Phase 4 alpha-mask feather radius in pixels "
                        "(0 = no blending, raw Phase 3 output)")
    p.add_argument("--correct-on-start", action="store_true",
                   help="start with side-by-side correction view enabled")
    return p.parse_args()


def open_camera(index: int, width: int, height: int,
                backend: str) -> cv2.VideoCapture:
    api = cv2.CAP_AVFOUNDATION if backend == "avfoundation" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, api)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(
            f"could not open camera index {index} (backend={backend}).\n"
            "  - on macOS, grant Camera permission to your terminal: "
            "System Settings -> Privacy & Security -> Camera -> enable Terminal/iTerm/VS Code\n"
            "  - then fully quit and reopen the terminal\n"
            "  - try --camera 1 if you have multiple cameras"
        )
    return cap


def open_metrics_csv(disabled: bool):
    if disabled:
        return None, None, None
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = METRICS_DIR / f"metrics_{stamp}.csv"
    f = open(path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writeheader()
    return path, f, writer


def main() -> None:
    args = parse_args()
    cap = open_camera(args.camera, args.width, args.height, args.backend)

    face_mesh = make_face_mesh(refine=True)
    frame_times: deque[float] = deque(maxlen=30)

    csv_path, csv_fh, csv_writer = open_metrics_csv(args.no_csv)
    if csv_path is not None:
        print(f"[metrics] logging to {csv_path}", flush=True)

    window = "perspective correction"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)

    flash_until = 0.0
    show_metrics = True
    show_correction = args.correct_on_start
    frame_idx = 0
    t_start = time.perf_counter()

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame_bgr = cap.read()
            if not ok:
                print("frame grab failed")
                break
            frame_bgr = cv2.flip(frame_bgr, 1)
            h, w = frame_bgr.shape[:2]

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(frame_rgb)

            pts_xyz = None
            pts_xy = None
            if result.multi_face_landmarks:
                pts_xyz = landmarks_to_xyz(result.multi_face_landmarks[0], w, h)
                pts_xy = pts_xyz[:, :2]

            m = compute_metrics(pts_xy) if pts_xy is not None else None

            overlay = frame_bgr.copy()
            if pts_xy is not None:
                draw_landmarks(overlay, pts_xy)

            corrected = None
            warp_ms = 0.0
            if show_correction and pts_xyz is not None and pts_xyz.shape[0] >= 478:
                t_w = time.perf_counter()
                corrected, _ = warp_mod.correct(
                    frame_bgr, pts_xyz,
                    strength=args.strength,
                    uniform_scale=args.uniform_scale,
                    feather=args.feather,
                )
                warp_ms = (time.perf_counter() - t_w) * 1000

            dt = time.perf_counter() - t0
            frame_times.append(dt)
            fps = len(frame_times) / sum(frame_times) if frame_times else 0.0
            draw_hud(overlay, fps, pts_xy is not None, hud_text(m), show_metrics)
            if show_correction:
                if args.uniform_scale is not None:
                    mode_str = f"UNIFORM scale={args.uniform_scale:.2f}"
                else:
                    mode_str = f"DEPTH strength={args.strength:.2f}"
                _put(overlay,
                     f"{mode_str}  feather={args.feather:.0f}"
                     f"  warp={warp_ms:4.1f}ms",
                     (10, h - 36), scale=0.55, color=(0, 200, 255))

            if time.time() < flash_until:
                _put(overlay, "SAVED", (w // 2 - 80, h // 2),
                     scale=2.0, color=(0, 255, 0))

            if show_correction and corrected is not None:
                _put(corrected, "CORRECTED", (10, 30), scale=0.7,
                     color=(0, 200, 255))
                display = np.hstack([overlay, corrected])
            else:
                display = overlay
            cv2.imshow(window, display)

            if csv_writer is not None:
                csv_writer.writerow(
                    csv_row(frame_idx, time.perf_counter() - t_start, fps, m)
                )

            frame_idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue
            if key == ord("q") or key == 27:
                break
            if key == ord("s"):
                path = save_capture(
                    frame_bgr, overlay, pts_xyz,
                    frame_corrected=corrected,
                    mode=("uniform" if args.uniform_scale is not None
                          else "depth") if corrected is not None else None,
                    strength=args.strength if corrected is not None else None,
                    uniform_scale=args.uniform_scale if corrected is not None else None,
                    feather=args.feather if corrected is not None else None,
                )
                tag = "with corrected" if corrected is not None else (
                    "landmarks only" if pts_xyz is not None else "no face detected")
                print(f"[save] {path.parent}/{path.stem.rsplit('_', 1)[0]}_*  "
                      f"({tag})", flush=True)
                flash_until = time.time() + 0.6
            elif key == ord("m"):
                show_metrics = not show_metrics
                print(f"[hud] metrics {'on' if show_metrics else 'off'}",
                      flush=True)
            elif key == ord("c"):
                show_correction = not show_correction
                print(f"[hud] correction {'on' if show_correction else 'off'}"
                      f" (scale={args.scale})", flush=True)
            else:
                print(f"[key] unknown keycode {key} (focus the video window, "
                      f"then press s/m/c/q)", flush=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        if csv_fh is not None:
            csv_fh.close()
            print(f"[metrics] wrote {frame_idx} rows to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
