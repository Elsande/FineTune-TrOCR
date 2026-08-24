"""Tahap 8: Normalisasi ukuran (resize).

Resize ke rentang resolusi wajar untuk input model — tidak terlalu besar
(agar tidak lambat di CPU) dan tidak terlalu kecil (agar teks tetap
terbaca). Implementasi:
- Sisi terpanjang > MAX_SIDE   -> scale down agar sisi terpanjang == MAX_SIDE.
- Sisi terpendek < MIN_SIDE    -> scale up agar sisi terpendek == MIN_SIDE.
"""

from __future__ import annotations

import cv2
import numpy as np

MAX_SIDE: int = 1600
MIN_SIDE: int = 800


def _scale(image: np.ndarray, factor: float) -> np.ndarray:
    height, width = image.shape[:2]
    new_w = max(1, round(width * factor))
    new_h = max(1, round(height * factor))
    # INTER_AREA untuk downscale, INTER_CUBIC untuk upscale.
    interpolation = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def resize_to_range(image: np.ndarray) -> np.ndarray:
    """Resize gambar ke rentang ukuran wajar (MAX_SIDE / MIN_SIDE)."""
    height, width = image.shape[:2]
    longest = max(width, height)
    shortest = min(width, height)

    if longest > MAX_SIDE:
        image = _scale(image, MAX_SIDE / longest)
        height, width = image.shape[:2]
        shortest = min(width, height)

    if shortest < MIN_SIDE:
        image = _scale(image, MIN_SIDE / shortest)

    return image
