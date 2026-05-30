"""Monocular depth estimation via Depth Anything V2.

Wraps Hugging Face transformers' depth-estimation pipeline so the rest of the
project can ask for a per-pixel depth map of a frame without touching torch
directly. The output depth is rescaled to match MediaPipe's coordinate scale
(image-width-normalized * width = pixels), and re-centered so the face plane
sits at z = z_median, exactly the convention warp.dense_perspective_correct
already uses.

Why this exists:
  Our dense pipeline interpolates 478 sparse MediaPipe z values across a
  Delaunay mesh. That's noisy (MediaPipe z is jittery) and only defined
  inside the face oval. A learned monocular depth model gives a clean,
  high-resolution depth field over the whole frame, which is exactly the
  signal our per-pixel re-projection consumes.

Lazy torch / transformers import so the existing pipeline keeps running on
CPU-only machines that don't have torch installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid hard dep on torch at import time
    import torch


# Hugging Face model IDs. Small is ~25M params, runs in ~10ms on a 5070 Ti.
# Base is ~98M, larger ~330M. Small is plenty for our use.
MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}
DEFAULT_MODEL = "small"


@dataclass
class DepthResult:
    depth_px: np.ndarray   # (H, W) float32, same scale as MediaPipe z * W
    depth_raw: np.ndarray  # (H, W) float32, raw model output before rescale


class MLDepthEstimator:
    """One-shot loader. Holds the pipeline + device.

    Usage:
        est = MLDepthEstimator(size="small", device="cuda")  # or "cpu", "mps"
        d = est.estimate(frame_bgr)
        # d.depth_px is HxW, ready to hand to dense_perspective_correct
    """

    def __init__(self, size: str = DEFAULT_MODEL,
                 device: str | None = None) -> None:
        try:
            import torch  # noqa: F401
            from transformers import pipeline
        except ImportError as e:
            raise SystemExit(
                "Depth Anything V2 requires torch + transformers. Install on "
                "the GPU box with:\n"
                "    pip install --index-url https://download.pytorch.org/whl/cu126 \\\n"
                "        torch torchvision\n"
                "    pip install transformers pillow\n"
                f"(import error: {e})"
            )
        import torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        model_id = MODELS.get(size, MODELS[DEFAULT_MODEL])
        self.device = device
        self.model_id = model_id
        # torch_dtype=float16 on cuda is ~2x faster and matches accuracy
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        self.pipe = pipeline(
            task="depth-estimation",
            model=model_id,
            device=device,
            torch_dtype=torch_dtype,
        )

    def estimate(self, frame_bgr: np.ndarray,
                 target_scale_px: float | None = None) -> DepthResult:
        """frame_bgr: HxWx3 uint8 OpenCV image.

        Returns DepthResult with depth_px rescaled so its standard deviation
        roughly matches the typical MediaPipe-z spread (about 0.05 * width in
        image-width-normalized units, i.e. ~0.05 * width pixels). That keeps
        the existing `--alpha` knob in the same operating range.
        """
        from PIL import Image
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR frame, got {frame_bgr.shape}")
        rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB
        img = Image.fromarray(rgb)
        out = self.pipe(img)
        # out["predicted_depth"] is a torch tensor in model space; Depth
        # Anything V2 returns INVERSE depth (larger = closer to camera). We
        # negate so smaller = closer to camera, matching MediaPipe z.
        depth_t = out["predicted_depth"]
        # Move to numpy
        depth = depth_t.detach().to("cpu").numpy().astype(np.float32)
        # Flatten any singleton dims and ensure HxW
        depth = np.squeeze(depth)
        if depth.shape != frame_bgr.shape[:2]:
            # resize to match the frame
            import cv2 as _cv2
            depth = _cv2.resize(depth, (frame_bgr.shape[1], frame_bgr.shape[0]),
                                interpolation=_cv2.INTER_LINEAR)
        # Negate so closer-to-camera is smaller, matching MediaPipe convention
        depth_oriented = -depth

        # Rescale to roughly match MediaPipe z * W spread. MediaPipe z * W
        # typically has std ~ 0.04 * W. We standardize the learned depth then
        # multiply by that target std.
        if target_scale_px is None:
            target_scale_px = 0.04 * frame_bgr.shape[1]
        mu = float(depth_oriented.mean())
        sigma = float(depth_oriented.std()) or 1.0
        depth_px = (depth_oriented - mu) / sigma * target_scale_px

        return DepthResult(depth_px=depth_px.astype(np.float32),
                           depth_raw=depth.astype(np.float32))


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Visualization helper: depth (HxW float) -> BGR uint8 turbo colormap."""
    import cv2 as _cv2
    d = depth.astype(np.float32)
    lo, hi = float(d.min()), float(d.max())
    if hi - lo < 1e-6:
        norm = np.zeros_like(d, dtype=np.uint8)
    else:
        norm = ((d - lo) / (hi - lo) * 255).astype(np.uint8)
    return _cv2.applyColorMap(norm, _cv2.COLORMAP_TURBO)
