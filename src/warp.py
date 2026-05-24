"""Phase 3: single-frame perspective correction via Delaunay piecewise-affine warp.

Approach (intentionally simple — ship a stable real-time
pipeline before getting fancy):

  1. Treat the MediaPipe FACE_OVAL ring as fixed boundary landmarks.
  2. Shrink every other landmark toward the face center by `scale` (0.92 to start).
  3. Add the 4 image corners + 4 edge midpoints as additional fixed anchors so
     the warp tiles the whole frame and the unwarped background is preserved.
  4. Delaunay-triangulate the point set; for each output-space triangle compute
     the affine transform from the *target* triangle to the *source* triangle
     and fill the bbox of that target triangle into map_x / map_y.
  5. cv2.remap with the resulting maps.

Phase 4 will replace the hard inside/outside split with a soft alpha mask;
Phase 7 will cache the triangulation and map arrays between frames.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.spatial import Delaunay

# MediaPipe Face Mesh FACE_OVAL ring (36 unique indices, ordered around the
# perimeter). Kept here as a tuple so it is hashable and trivially shareable
# with other modules.
FACE_OVAL: tuple[int, ...] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)


def _image_anchors(width: int, height: int) -> np.ndarray:
    w1, h1 = width - 1, height - 1
    return np.array(
        [
            [0, 0], [w1, 0], [w1, h1], [0, h1],          # corners
            [w1 // 2, 0], [w1, h1 // 2],                # edge midpoints
            [w1 // 2, h1], [0, h1 // 2],
        ],
        dtype=np.float32,
    )


def shrink_target(pts: np.ndarray, scale: float,
                  fixed_indices: tuple[int, ...] = FACE_OVAL) -> np.ndarray:
    """Return target landmarks: shrink non-fixed landmarks toward the centroid
    of the fixed (face-oval) ring by `scale` (1.0 = identity, <1 shrinks)."""
    target = pts.copy().astype(np.float32)
    fixed = np.asarray(fixed_indices, dtype=np.int64)
    center = pts[fixed].mean(axis=0)
    movable = np.ones(len(pts), dtype=bool)
    movable[fixed] = False
    target[movable] = center + scale * (pts[movable] - center)
    return target


def build_point_sets(pts_mediapipe: np.ndarray, shape: tuple[int, int, int],
                     scale: float) -> tuple[np.ndarray, np.ndarray]:
    """src_pts (source landmarks) and dst_pts (target landmarks), each with
    image anchors appended so the triangulation covers the whole frame."""
    h, w = shape[:2]
    anchors = _image_anchors(w, h)
    target = shrink_target(pts_mediapipe, scale)
    src_pts = np.vstack([pts_mediapipe.astype(np.float32), anchors])
    dst_pts = np.vstack([target.astype(np.float32), anchors])
    return src_pts, dst_pts


def triangulate(dst_pts: np.ndarray) -> np.ndarray:
    """Delaunay triangulation in target space. Returns (T, 3) index array."""
    return Delaunay(dst_pts).simplices.astype(np.int32)


def build_warp_maps(src_pts: np.ndarray, dst_pts: np.ndarray,
                    triangles: np.ndarray,
                    shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Construct (map_x, map_y) for cv2.remap implementing the piecewise-affine
    warp from dst (output) space to src (input) space.

    Identity-initialized: pixels outside every target triangle sample from
    themselves, so background outside the convex hull of the points is left
    untouched (in practice the image anchors make the hull cover the frame).
    """
    h, w = shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    map_x = xx.copy()
    map_y = yy.copy()

    for tri in triangles:
        s = src_pts[tri].astype(np.float32)
        d = dst_pts[tri].astype(np.float32)

        xmin = int(np.floor(d[:, 0].min()))
        ymin = int(np.floor(d[:, 1].min()))
        xmax = int(np.ceil(d[:, 0].max())) + 1
        ymax = int(np.ceil(d[:, 1].max())) + 1
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue

        # mask for which bbox pixels actually lie inside this target triangle
        mask = np.zeros((ymax - ymin, xmax - xmin), dtype=np.uint8)
        local = (d - np.array([[xmin, ymin]], dtype=np.float32)).astype(np.int32)
        cv2.fillConvexPoly(mask, local, 1)
        if not mask.any():
            continue

        # affine from target triangle to source triangle
        M = cv2.getAffineTransform(d, s)  # 2x3, maps dst -> src
        xx_w = xx[ymin:ymax, xmin:xmax]
        yy_w = yy[ymin:ymax, xmin:xmax]
        sx = M[0, 0] * xx_w + M[0, 1] * yy_w + M[0, 2]
        sy = M[1, 0] * xx_w + M[1, 1] * yy_w + M[1, 2]

        sel = mask.astype(bool)
        map_x[ymin:ymax, xmin:xmax][sel] = sx[sel]
        map_y[ymin:ymax, xmin:xmax][sel] = sy[sel]

    return map_x, map_y


def make_alpha_mask(pts_mediapipe: np.ndarray,
                    shape: tuple[int, int, int],
                    feather: float) -> np.ndarray:
    """Phase 4 boundary blend mask.

    Returns a (H, W) float32 array in [0, 1] where 1.0 means "use the corrected
    pixel" and 0.0 means "use the raw pixel." The mask is the filled FACE_OVAL
    polygon, eroded inward by ~feather/2 and then Gaussian-blurred, so the
    transition straddles the face boundary in a band of width ~`feather`.
    """
    h, w = shape[:2]
    binary = np.zeros((h, w), dtype=np.uint8)
    oval = pts_mediapipe[list(FACE_OVAL)].astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(binary, [oval], 255)

    if feather > 0:
        erode_r = max(1, int(round(feather * 0.5)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erode_r + 1, 2 * erode_r + 1)
        )
        binary = cv2.erode(binary, kernel)
        sigma = max(1.0, feather * 0.5)
        ksize = max(3, int(round(sigma * 6)) | 1)  # odd
        blurred = cv2.GaussianBlur(binary, (ksize, ksize), sigma)
    else:
        blurred = binary

    return blurred.astype(np.float32) * (1.0 / 255.0)


def blend(raw: np.ndarray, corrected: np.ndarray,
          alpha: np.ndarray) -> np.ndarray:
    """alpha * corrected + (1 - alpha) * raw, per channel."""
    a = alpha[..., None]
    out = a * corrected.astype(np.float32) + (1.0 - a) * raw.astype(np.float32)
    return np.clip(out, 0, 255).astype(raw.dtype)


def correct(image: np.ndarray, pts_mediapipe: np.ndarray,
            scale: float = 0.92,
            feather: float = 30.0) -> tuple[np.ndarray, dict]:
    """End-to-end Phase 3+4 correction. Returns (corrected_image, debug_info).

    feather > 0 enables Phase 4 boundary blending. feather=0 returns the raw
    piecewise-affine output (the discontinuous Phase 3 baseline).
    """
    src_pts, dst_pts = build_point_sets(pts_mediapipe, image.shape, scale)
    triangles = triangulate(dst_pts)
    map_x, map_y = build_warp_maps(src_pts, dst_pts, triangles, image.shape)
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    if feather > 0:
        alpha = make_alpha_mask(pts_mediapipe, image.shape, feather)
        corrected_img = blend(image, warped, alpha)
    else:
        corrected_img = warped
    return corrected_img, {
        "n_triangles": int(triangles.shape[0]),
        "n_src_pts": int(src_pts.shape[0]),
        "scale": scale,
        "feather": feather,
    }
