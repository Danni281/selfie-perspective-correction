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
| 3 | Single-frame correction (Delaunay warp) | done |
| 4 | Boundary blending mask | done |
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
- `c` — toggle Phase 3 correction view (raw on left, corrected on right)

CLI flags:
- `--mode {dense, nose, perspective, uniform}` — correction mode.
  Default `dense`.
  - `dense` (NEW METHOD, default): treats MediaPipe (x, y, z) as a
    sparse depth signal, barycentric-interpolates to a dense per-pixel
    depth map over the Delaunay mesh, then applies the true pinhole
    perspective re-projection per pixel:
    `scale(u,v) = α(t+1)/(t+α)` with `t = z(u,v)/d_old`. Inverse-warp
    via `cv2.remap`. Controlled by `--alpha`. Smooth depth field =
    smooth warp = no sparse-landmark artifacts.
  - `nose`: localized uniform shrink of ~20 nose-region landmarks only.
    Subtle, artifact-free, no depth math involved.
  - `perspective`: full-face sparse-landmark depth re-projection.
    Visible peripheral artifacts (squashed brow / ghosting) because
    sparse 2D landmark warps can't represent the true 3D perspective
    transform. Kept for ablation.
  - `uniform`: legacy Phase 3 baseline (uniform radial shrink). Puffs
    the cheeks. Kept for ablation comparison.
- `--alpha 2.0` — `dense` mode: virtual camera distance ratio. `1.0`
  is no correction; `2.0` is "as if shot from 2× the distance"; large
  values approach orthographic projection. Default `2.0`.
- `--strength 0.3` — correction strength for `nose` and `perspective`
  modes. `0` = no correction; `1` = aggressive. Default `0.3`.
- `--auto-strength` — **derive `strength` per frame from the measured
  nose_w/face_w ratio**: `strength = clip(1 - target / current, 0, max)`.
  Closes the loop between Phase 2 (measure distortion) and Phase 3
  (apply correction). When you're close to the camera, correction is
  strong; when you back away, correction fades to zero.
- `--target-nose-ratio 0.22` — neutral nose_w/face_w from portrait
  photography references (~85mm lens at ~1m subject distance).
- `--max-strength 0.6` — upper clamp for auto-derived strength.
- `--uniform-scale 0.85` — Phase 3 ablation: shrink factor when
  `--mode uniform`.
- `--feather 30` — Phase 4 alpha-mask feather radius in pixels. The
  corrected face is alpha-blended back to the raw image across a band
  ~this wide straddling the face oval, so the boundary discontinuity
  disappears. `0` disables blending. Default `30`.
- `--correct-on-start` — open already in correction view.

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
src/main.py         # live capture + face mesh + metrics HUD + correction view
src/metrics.py      # landmark indices + per-frame ratio computation + auto-strength
src/warp.py         # piecewise affine + dense depth-driven warps
src/smoothing.py    # EMA / Kalman / 1-Euro Filter for landmark streams
src/process_clip.py # offline: record a clip, run pipeline frame-by-frame
captures/           # saved live snapshots (gitignored)
metrics_logs/       # per-run live metrics CSVs (gitignored)
clips/              # recorded raw mp4s (gitignored)
results/            # processed mp4s + per-clip CSVs (gitignored)
requirements.txt
```
