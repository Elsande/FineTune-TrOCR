"""Tahap 4: Denoising klasik (non-ML).

Menghilangkan noise hasil scan/foto (salt-and-pepper, grain) dengan teknik
klasik OpenCV saja — BUKAN deep-learning denoiser (di luar scope Fase 1).

Pilihan metode:
- "bilateral"  : bilateral filter — pertahankan tepi teks, lebih cepat.
- "fastNlMeans": fastNlMeansDenoising — kualitas terbaik, lebih lambat
                 (perhatikan performa CPU).

Default "bilateral" dipilih untuk keseimbangan kualitas/kecepatan pada
environment CPU-only.
"""

from __future__ import annotations

import cv2
import numpy as np

DENOISE_METHOD: str = "bilateral"


def _denoise_gray(gray: np.ndarray) -> np.ndarray:
    if DENOISE_METHOD == "fastNlMeans":
        return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # bilateral: d=5, sigmaColor=75, sigmaSpace=75 cukup untuk noise ringan
    # sambil tetap mempertahankan ketajaman tepi teks.
    return cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)


def denoise(image: np.ndarray) -> np.ndarray:
    """Kurangi noise gambar (RGB atau grayscale)."""
    if image.ndim == 3:
        return cv2.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)
    return _denoise_gray(image)
