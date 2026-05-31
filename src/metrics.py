"""Phase 2: per-frame landmark-ratio metrics for perspective distortion.

Ratios (proposal section "Evaluation"):
    ipd_over_face_w      inter-pupillary distance / face width
    nose_w_over_face_w   nose width / face width
    nose_chin_over_face_w  nose-tip-to-chin / face width
    ear_ear_over_face_h  ear-to-ear width / face height

MediaPipe Face Mesh landmark indices used (refine_landmarks=True needed for
iris centers 468/473; the rest are in the base 468-point mesh):

    iris centers          468 (left), 473 (right)
    nose alar wings        49 (right), 279 (left)
    nose tip                4
    chin tip              152
    forehead top center    10
    cheek/face edges      234 (right), 454 (left)
    temple/ear proxies    127 (right), 356 (left)

"left/right" here is the subject's own left/right; in a non-mirrored frame
landmark 468 (left iris) sits on the image-right side.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- landmark indices (constants) ---
L_IRIS, R_IRIS = 468, 473
NOSE_TIP = 4
CHIN = 152
FOREHEAD = 10
NOSE_R_ALAR, NOSE_L_ALAR = 49, 279
FACE_R_EDGE, FACE_L_EDGE = 234, 454
EAR_R, EAR_L = 127, 356

REQUIRES_IRIS = (L_IRIS, R_IRIS)


@dataclass
class FrameMetrics:
    face_width: float
    face_height: float
    ipd: float
    nose_w: float
    nose_chin: float
    ear_ear: float
    ipd_over_face_w: float
    nose_w_over_face_w: float
    nose_chin_over_face_w: float
    ear_ear_over_face_h: float

    def as_dict(self) -> dict[str, float]:
        return {
            "face_width": self.face_width,
            "face_height": self.face_height,
            "ipd": self.ipd,
            "nose_w": self.nose_w,
            "nose_chin": self.nose_chin,
            "ear_ear": self.ear_ear,
            "ipd_over_face_w": self.ipd_over_face_w,
            "nose_w_over_face_w": self.nose_w_over_face_w,
            "nose_chin_over_face_w": self.nose_chin_over_face_w,
            "ear_ear_over_face_h": self.ear_ear_over_face_h,
        }


CSV_FIELDS = (
    "frame_idx",
    "t_seconds",
    "fps",
    "face_detected",
    "face_width",
    "face_height",
    "ipd",
    "nose_w",
    "nose_chin",
    "ear_ear",
    "ipd_over_face_w",
    "nose_w_over_face_w",
    "nose_chin_over_face_w",
    "ear_ear_over_face_h",
)


def _dist(pts: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(pts[a] - pts[b]))


def compute_metrics(pts: np.ndarray) -> FrameMetrics | None:
    """pts: (N, 2) pixel coords from MediaPipe Face Mesh.

    Returns None if the landmark set is too small (no iris refinement, no
    face) so callers can handle the gap explicitly.
    """
    if pts is None or pts.shape[0] < max(REQUIRES_IRIS) + 1:
        return None

    face_w = _dist(pts, FACE_R_EDGE, FACE_L_EDGE)
    face_h = _dist(pts, FOREHEAD, CHIN)
    if face_w <= 0 or face_h <= 0:
        return None

    ipd = _dist(pts, L_IRIS, R_IRIS)
    nose_w = _dist(pts, NOSE_R_ALAR, NOSE_L_ALAR)
    nose_chin = _dist(pts, NOSE_TIP, CHIN)
    ear_ear = _dist(pts, EAR_R, EAR_L)

    return FrameMetrics(
        face_width=face_w,
        face_height=face_h,
        ipd=ipd,
        nose_w=nose_w,
        nose_chin=nose_chin,
        ear_ear=ear_ear,
        ipd_over_face_w=ipd / face_w,
        nose_w_over_face_w=nose_w / face_w,
        nose_chin_over_face_w=nose_chin / face_w,
        ear_ear_over_face_h=ear_ear / face_h,
    )


def csv_row(frame_idx: int, t_seconds: float, fps: float,
            m: FrameMetrics | None) -> dict[str, object]:
    row: dict[str, object] = {
        "frame_idx": frame_idx,
        "t_seconds": f"{t_seconds:.4f}",
        "fps": f"{fps:.2f}",
        "face_detected": int(m is not None),
    }
    if m is None:
        for k in CSV_FIELDS:
            if k not in row:
                row[k] = ""
    else:
        for k, v in m.as_dict().items():
            row[k] = f"{v:.4f}"
    return row


# Reference landmark ratios for a "neutral" portrait shot at ~85mm focal
# length at ~1m distance. Values drawn from face-photography references and
# Burgos-Artizzu et al. 2014 (CMDP corpus, distance 7).
PORTRAIT_REFERENCE = {
    "nose_w_over_face_w": 0.22,
    "ipd_over_face_w": 0.42,
    "nose_chin_over_face_w": 0.55,
    "ear_ear_over_face_h": 0.85,
}


def auto_strength(m: FrameMetrics | None,
                  target_nose_ratio: float = 0.22,
                  max_strength: float = 0.6) -> float:
    """Derive Phase 3 nose-correction strength from measured distortion.

    Uses the primary perspective indicator (nose_width / face_width). At
    portrait distance the ratio sits near `target_nose_ratio`; at close
    selfie distance it inflates well above. The derived strength is the
    fractional shrink needed to bring the current ratio back to target:

        strength = clip(1 - target / current, 0, max_strength)

    This is the bridge between Phase 2 (measure distortion) and Phase 3
    (apply correction): correction is no longer
    a hand-tuned constant, it is a function of measured distortion per
    frame.
    """
    if m is None or m.nose_w_over_face_w <= 0:
        return 0.0
    raw = 1.0 - target_nose_ratio / m.nose_w_over_face_w
    return float(max(0.0, min(max_strength, raw)))


# Empirically-undistorted nose ratio: the CMDP 16 ft (farthest, GT) mean
# across 51 subjects. A face measured at or below this needs no correction;
# above it, the excess is perspective distortion to undo.
NEUTRAL_NOSE_RATIO = 0.277


def auto_alpha(m: FrameMetrics | None,
               neutral_ratio: float = NEUTRAL_NOSE_RATIO,
               gain: float = 10.0,
               max_alpha: float = 2.5,
               deadzone: float = 0.06) -> float:
    """Derive the dense-mode virtual-camera ratio `alpha` from measured
    distortion, so the live correction adapts to camera distance.

    alpha is the dense warp's knob: 1.0 = no correction, larger = stronger
    (as if shot from farther with a longer lens). We map the fractional
    inflation of nose_w/face_w above the empirically-undistorted reference
    (CMDP 16 ft mean, 0.277) to alpha, after subtracting a dead-zone:

        excess = current / neutral - 1
        alpha  = clip(1 + gain * max(0, excess - deadzone), 1.0, max_alpha)

    The `deadzone` (default 0.06 = 6%) means a face whose nose ratio is only
    mildly above neutral -- i.e. shot at a normal/far distance, or just a
    person whose face is naturally a bit wider than the population mean -- is
    left EXACTLY untouched (alpha = 1.0). The warp only engages once the
    measured distortion clearly exceeds the dead-zone, which happens at close
    selfie range. Subtracting the dead-zone (rather than hard-gating) keeps
    alpha continuous at the threshold, so there is no visible pop as you lean
    in. Returns 1.0 (no correction) when no face is measured.
    """
    if m is None or m.nose_w_over_face_w <= 0 or neutral_ratio <= 0:
        return 1.0
    excess = m.nose_w_over_face_w / neutral_ratio - 1.0
    alpha = 1.0 + gain * max(0.0, excess - deadzone)
    return float(max(1.0, min(max_alpha, alpha)))


def hud_text(m: FrameMetrics | None) -> list[str]:
    if m is None:
        return ["IPD/Wf:  ----   nose/Wf:  ----",
                "n-chin/Wf: ---   ear/Hf:  ----"]
    return [
        f"IPD/Wf:  {m.ipd_over_face_w:.3f}   nose/Wf:  {m.nose_w_over_face_w:.3f}",
        f"n-chin/Wf: {m.nose_chin_over_face_w:.3f}   ear/Hf:   {m.ear_ear_over_face_h:.3f}",
    ]
