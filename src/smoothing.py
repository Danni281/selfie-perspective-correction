"""Phase 6: temporal smoothing of MediaPipe landmark streams.

Three smoothers with the same interface. Each holds state across frames and
exposes `update(pts, t) -> smoothed_pts`:

    EMASmoother        exponentially-weighted moving average
    KalmanSmoother     per-landmark per-coord independent linear KF (vectorized)
    OneEuroSmoother    Casiez et al. 2012 -- adaptive low-pass

They all operate on (N, D) arrays (typically N=478 landmarks, D=2 or 3
coords) so the same smoother can be applied to either the (x, y) pixel
landmarks or the full (x, y, z) MediaPipe output.

The "none" baseline is the per-frame pipeline -- no smoothing, raw landmark
output every frame. That is what produces the flicker Phase 6 fixes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Smoother(ABC):
    name: str = "abstract"

    @abstractmethod
    def update(self, pts: np.ndarray, t: float) -> np.ndarray:
        """Consume a new landmark observation at time t, return smoothed pts."""

    def reset(self) -> None:
        """Discard internal state (call on face-lost / scene cut)."""


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMASmoother(Smoother):
    """state_t = alpha * state_{t-1} + (1 - alpha) * obs_t.

    alpha closer to 1.0 = heavier smoothing / more lag.
    The plan specifies alpha=0.7 (i.e. 70% weight on history, 30% new) as the
    Phase 6 starting point.
    """

    name = "ema"

    def __init__(self, alpha: float = 0.7) -> None:
        self.alpha = float(alpha)
        self.state: np.ndarray | None = None

    def reset(self) -> None:
        self.state = None

    def update(self, pts: np.ndarray, t: float) -> np.ndarray:
        pts = pts.astype(np.float32, copy=False)
        if self.state is None or self.state.shape != pts.shape:
            self.state = pts.copy()
            return self.state
        self.state = self.alpha * self.state + (1.0 - self.alpha) * pts
        return self.state


# ---------------------------------------------------------------------------
# Kalman (per-landmark per-coord, vectorized)
# ---------------------------------------------------------------------------

class KalmanSmoother(Smoother):
    """Vectorized linear Kalman filter, one filter per landmark coord.

    State per scalar coord: [position, velocity]. Constant-velocity dynamics:
        x_{k+1} = x_k + v_k * dt          (+ process noise)
        v_{k+1} = v_k                     (+ process noise)
    Observation: position only.

    Maintained as two (N, D) arrays for position and velocity plus a
    (N, D, 2, 2) covariance tensor, all updated in-place by numpy vector ops
    -- no per-landmark python loop, so cost scales with O(N*D) per frame.

    process_noise_pos / process_noise_vel govern how reactive the filter is;
    measurement_noise governs how much it trusts the raw MediaPipe output.
    Defaults are tuned for the ~30 FPS landmark stream this project consumes
    -- bigger process_noise = lighter smoothing, more responsive.
    """

    name = "kalman"

    def __init__(self,
                 process_noise_pos: float = 1.0,
                 process_noise_vel: float = 5.0,
                 measurement_noise: float = 9.0) -> None:
        self.q_pos = float(process_noise_pos)
        self.q_vel = float(process_noise_vel)
        self.r = float(measurement_noise)
        self.pos: np.ndarray | None = None
        self.vel: np.ndarray | None = None
        self.P: np.ndarray | None = None  # (N, D, 2, 2)
        self.t_prev: float | None = None

    def reset(self) -> None:
        self.pos = self.vel = self.P = None
        self.t_prev = None

    def _init(self, pts: np.ndarray) -> None:
        n, d = pts.shape
        self.pos = pts.astype(np.float32, copy=True)
        self.vel = np.zeros_like(self.pos)
        # initial covariance: diag(r, q_vel) per (landmark, coord)
        P = np.zeros((n, d, 2, 2), dtype=np.float32)
        P[..., 0, 0] = self.r
        P[..., 1, 1] = self.q_vel
        self.P = P

    def update(self, pts: np.ndarray, t: float) -> np.ndarray:
        pts = pts.astype(np.float32, copy=False)
        if (self.pos is None or self.pos.shape != pts.shape):
            self._init(pts)
            self.t_prev = t
            return self.pos.copy()
        assert self.pos is not None and self.vel is not None and self.P is not None
        dt = max(1e-3, t - (self.t_prev or t))
        self.t_prev = t

        # --- predict ---
        # state: x' = x + v*dt;  v' = v
        self.pos = self.pos + self.vel * dt
        # F = [[1, dt], [0, 1]], Q = diag(q_pos, q_vel)
        # P' = F P F^T + Q,  per (landmark, coord)
        P = self.P
        # F P
        FP00 = P[..., 0, 0] + dt * P[..., 1, 0]
        FP01 = P[..., 0, 1] + dt * P[..., 1, 1]
        FP10 = P[..., 1, 0]
        FP11 = P[..., 1, 1]
        # F P F^T
        P00 = FP00 + dt * FP01
        P01 = FP01
        P10 = FP10 + dt * FP11
        P11 = FP11
        # + Q
        P00 = P00 + self.q_pos
        P11 = P11 + self.q_vel

        # --- update with measurement z = pts ---
        # innovation y = z - H x   (H = [1, 0])
        y = pts - self.pos
        # S = H P H^T + R = P00 + R
        S = P00 + self.r
        # K = P H^T / S = [P00, P10] / S
        K0 = P00 / S
        K1 = P10 / S
        # x = x + K y
        self.pos = self.pos + K0 * y
        self.vel = self.vel + K1 * y
        # P = (I - K H) P
        new_P00 = (1.0 - K0) * P00
        new_P01 = (1.0 - K0) * P01
        new_P10 = P10 - K1 * P00
        new_P11 = P11 - K1 * P01

        P[..., 0, 0] = new_P00
        P[..., 0, 1] = new_P01
        P[..., 1, 0] = new_P10
        P[..., 1, 1] = new_P11
        return self.pos.copy()


# ---------------------------------------------------------------------------
# 1 Euro Filter (Casiez, Roussel, Vogel, CHI 2012)
# ---------------------------------------------------------------------------

def _euro_alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    """Discrete-time first-order low-pass alpha for cutoff (Hz) and dt (s)."""
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroSmoother(Smoother):
    """Casiez et al. 2012, "1 Euro Filter".

    Adaptive low-pass whose cutoff frequency rises with signal speed:
    at rest the cutoff is `min_cutoff` (heavy smoothing, kills jitter);
    during fast motion the cutoff is `min_cutoff + beta * |dx_smoothed|`
    (less smoothing, less lag).

    Tuning per the paper:
        min_cutoff: lower -> less jitter at rest, more lag during motion
        beta:       higher -> more aggressive when moving, less lag
        dcutoff:    cutoff for the smoothed-derivative path (usually ~1 Hz)

    Defaults tuned for the ~30 FPS landmark stream and pixel-scale magnitudes.
    """

    name = "oneeuro"

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007,
                 dcutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None
        self.t_prev: float | None = None

    def reset(self) -> None:
        self.x_prev = self.dx_prev = None
        self.t_prev = None

    def update(self, pts: np.ndarray, t: float) -> np.ndarray:
        pts = pts.astype(np.float32, copy=False)
        if self.x_prev is None or self.x_prev.shape != pts.shape:
            self.x_prev = pts.copy()
            self.dx_prev = np.zeros_like(pts)
            self.t_prev = t
            return self.x_prev.copy()
        assert self.dx_prev is not None
        dt = max(1e-3, t - (self.t_prev or t))
        self.t_prev = t

        # smoothed derivative
        dx = (pts - self.x_prev) / dt
        a_d = _euro_alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        # adaptive cutoff per landmark coord: |dx_hat| (a scalar speed proxy
        # per coord). The paper uses |dx_hat| directly per signal channel; we
        # do the same -- each landmark/coord adapts independently.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _euro_alpha(cutoff, dt)
        x_hat = a * pts + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat.copy()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

SMOOTHERS = {
    "none": None,
    "ema": EMASmoother,
    "kalman": KalmanSmoother,
    "oneeuro": OneEuroSmoother,
}


def make_smoother(name: str, **kwargs) -> Smoother | None:
    """Returns a fresh smoother instance, or None for 'none'."""
    cls = SMOOTHERS.get(name)
    if cls is None:
        return None
    return cls(**kwargs)
