"""Tahap 7: Sharpening (unsharp masking).

Pertajam tepi teks yang blur ringan SETELAH tahap denoising (bukan sebelum),
agar yang dipertajam bukan noise yang belum dibersihkan.
"""

from __future__ import annotations

import cv2
import numpy as np

# Kekuatan penajaman (alpha) dan sigma blur Gaussian untuk unsharp mask.
SHARPEN_AMOUNT: float = 1.0
SHARPEN_SIGMA: float = 1.0


def sharpen(image: np.ndarray) -> np.ndarray:
    """Unsharp masking: detail = gambar - versi blur; hasil = gambar + amount*detail."""
    blurred = cv2.GaussianBlur(image, (0, 0), SHARPEN_SIGMA)
    sharp = cv2.addWeighted(image, 1.0 + SHARPEN_AMOUNT, blurred, -SHARPEN_AMOUNT, 0)
    return sharp
