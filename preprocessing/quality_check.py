"""Quality check awal + quality gate.

Metrik dasar yang dihitung SEBELUM tahap perbaikan gambar:
- blur_score : Laplacian variance (OpenCV) pada kanal grayscale.
- resolution : (lebar, tinggi) dalam piksel.

Ambang batas (BLUR_THRESHOLD, MIN_RESOLUTION) diletakkan di bagian atas file
ini agar mudah di-tuning tanpa mengubah logika pipeline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


# ============================================================
# AMBANG BATAS KUALITAS — silakan tuning di sini.
# ============================================================
# Laplacian variance di bawah nilai ini dianggap gambar buram.
# Nilai sengaja dilonggarkan (3.0) karena contoh dataset kecil dan beberapa
# dokumen scan/gambar HP yang valid secara visual punya blur score ~5–9.
# Hanya gambar yang benar-benar buram/tidak terbaca yang akan ditolak.
BLUR_THRESHOLD: float = 3.0

# Resolusi minimum (lebar, tinggi) agar dokumen layak diproses OCR.
# Dilonggarkan dari (100, 100) karena contoh dokumen bisa berukuran kecil.
MIN_RESOLUTION: Tuple[int, int] = (50, 50)


def compute_blur_score(image: np.ndarray) -> float:
    """Hitung Laplacian variance sebagai indikator ketajaman gambar."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_resolution(image: np.ndarray) -> Tuple[int, int]:
    """Kembalikan ukuran (lebar, tinggi) gambar dalam piksel."""
    height, width = image.shape[:2]
    return (width, height)


def check_quality(
    image: np.ndarray,
) -> Tuple[bool, Optional[str], float, Tuple[int, int]]:
    """Evaluasi kualitas gambar terhadap ambang batas.

    Returns:
        (passed, reject_reason, blur_score, resolution)
        reject_reason None jika lolos, berisi alasan spesifik jika gagal.
    """
    blur_score = compute_blur_score(image)
    resolution = compute_resolution(image)

    if blur_score < BLUR_THRESHOLD:
        return (
            False,
            f"Gambar terlalu buram (blur score {blur_score:.1f} < ambang "
            f"{BLUR_THRESHOLD:.1f}). Tolong upload gambar lagi.",
            blur_score,
            resolution,
        )
    if resolution[0] < MIN_RESOLUTION[0] or resolution[1] < MIN_RESOLUTION[1]:
        return (
            False,
            f"Resolusi gambar terlalu kecil ({resolution[0]}x{resolution[1]} < "
            f"minimum {MIN_RESOLUTION[0]}x{MIN_RESOLUTION[1]}). "
            f"Tolong upload gambar lagi.",
            blur_score,
            resolution,
        )
    return True, None, blur_score, resolution
