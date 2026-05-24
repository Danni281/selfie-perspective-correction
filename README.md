# Real-Time Perspective Correction for Selfie Video

A real-time computer-vision pipeline that corrects close-range perspective distortion in selfie and webcam video.

Selfies and webcam video are shot at short camera-to-face distances, which makes
noses and foreheads look enlarged. This project builds a real-time pipeline that
reduces that perspective distortion and stays temporally stable across frames.

## Status

| Phase | Description | State |
|------:|-------------|-------|
| 0 | Webcam capture loop + FPS counter | done |
| 1 | MediaPipe Face Mesh overlay + save key | done |
| 2 | Per-frame landmark-ratio metrics + CSV log | done |
| 3 | Single-frame correction (Delaunay warp) | todo |
| 4 | Boundary blending mask | todo |
| 5 | Per-frame baseline video | todo |
| 6 | Temporal smoothing (EMA + Kalman) | todo |
| 7 | Real-time optimization (30 FPS live) | todo |
| 8 | Evaluation script | todo |


## Setup

Requires **Python 3.12** (mediapipe 0.10.21 has no wheels for 3.13/3.14,
and newer mediapipe drops the Face Mesh `solutions` API).

```bash
brew install python@3.12         # if you don't have it
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python src/main.py              # default webcam, 1280x720
python src/main.py --camera 1   # second camera
```

Keys inside the window:
- `q` / `Esc` — quit
- `s` — save current frame + landmarks to `captures/`
- `m` — toggle metrics HUD on/off

Each `s` writes four files under `captures/`:
`<timestamp>_raw.png`, `<timestamp>_overlay.png`, `<timestamp>_landmarks.npy`,
`<timestamp>_meta.json`.

Every run also writes a per-frame metrics CSV to
`metrics_logs/metrics_<timestamp>.csv` with columns:
`frame_idx, t_seconds, fps, face_detected, face_width, face_height, ipd,
nose_w, nose_chin, ear_ear, ipd_over_face_w, nose_w_over_face_w,
nose_chin_over_face_w, ear_ear_over_face_h`. Disable with `--no-csv`.

## Layout

```
src/main.py        # capture loop + face mesh + metrics HUD
src/metrics.py     # landmark indices + per-frame ratio computation
captures/          # saved frames + landmarks (gitignored)
metrics_logs/      # per-run metrics CSVs (gitignored)
requirements.txt
```
