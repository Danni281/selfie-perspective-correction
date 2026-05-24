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
| 2 | Per-frame landmark-ratio metrics + CSV log | todo |
| 3 | Single-frame correction (Delaunay warp) | todo |
| 4 | Boundary blending mask | todo |
| 5 | Per-frame baseline video | todo |
| 6 | Temporal smoothing (EMA + Kalman) | todo |
| 7 | Real-time optimization (30 FPS live) | todo |
| 8 | Evaluation script | todo |


## Setup

Requires Python 3.10+ (tested on 3.14).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python src/main.py              # default webcam, 1280x720
python src/main.py --camera 1   # second camera
```

Keys inside the window:
- `q` — quit
- `s` — save current frame + landmarks to `captures/`

Each `s` writes four files under `captures/`:
`<timestamp>_raw.png`, `<timestamp>_overlay.png`, `<timestamp>_landmarks.npy`,
`<timestamp>_meta.json`.

## Layout

```
src/main.py        # phase-1 capture + face mesh entry point
captures/          # saved frames + landmarks (gitignored)
requirements.txt
```
