"""Phase 1: webcam capture + MediaPipe Face Mesh overlay.

Keys:
  q  quit
  s  save current frame + landmarks to captures/
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

CAPTURE_DIR = Path(__file__).resolve().parents[1] / "captures"


def make_face_mesh() -> mp.solutions.face_mesh.FaceMesh:
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def landmarks_to_pixels(landmarks, width: int, height: int) -> np.ndarray:
    pts = np.empty((len(landmarks.landmark), 2), dtype=np.float32)
    for i, lm in enumerate(landmarks.landmark):
        pts[i, 0] = lm.x * width
        pts[i, 1] = lm.y * height
    return pts


def draw_landmarks(frame: np.ndarray, pts: np.ndarray) -> None:
    for x, y in pts.astype(int):
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def draw_hud(frame: np.ndarray, fps: float, face_detected: bool) -> None:
    text = f"FPS: {fps:5.1f}  face: {'yes' if face_detected else 'no '}"
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 1, cv2.LINE_AA)
    hint = "q quit  |  s save"
    cv2.putText(frame, hint, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, hint, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def save_capture(frame_raw: np.ndarray, frame_annotated: np.ndarray,
                 landmarks_px: np.ndarray | None) -> Path:
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
    }
    if landmarks_px is not None:
        np.save(npy_path, landmarks_px)
    json_path.write_text(json.dumps(meta, indent=2))
    return raw_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 capture + face mesh")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--backend", choices=["avfoundation", "any"],
                   default="avfoundation",
                   help="capture backend (use avfoundation on macOS)")
    return p.parse_args()


def open_camera(index: int, width: int, height: int, backend: str) -> cv2.VideoCapture:
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


def main() -> None:
    args = parse_args()
    cap = open_camera(args.camera, args.width, args.height, args.backend)

    face_mesh = make_face_mesh()
    frame_times: deque[float] = deque(maxlen=30)

    window = "perspective correction"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame_bgr = cap.read()
            if not ok:
                print("frame grab failed")
                break
            frame_bgr = cv2.flip(frame_bgr, 1)  # mirror for selfie ergonomics
            h, w = frame_bgr.shape[:2]

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(frame_rgb)

            pts = None
            if result.multi_face_landmarks:
                pts = landmarks_to_pixels(result.multi_face_landmarks[0], w, h)

            overlay = frame_bgr.copy()
            if pts is not None:
                draw_landmarks(overlay, pts)

            dt = time.perf_counter() - t0
            frame_times.append(dt)
            fps = len(frame_times) / sum(frame_times) if frame_times else 0.0
            draw_hud(overlay, fps, pts is not None)

            cv2.imshow(window, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = save_capture(frame_bgr, overlay, pts)
                print(f"saved -> {path.parent}/{path.stem.split('_')[0]}_*")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()


if __name__ == "__main__":
    main()
