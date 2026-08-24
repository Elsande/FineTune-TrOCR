"""Tahap 3: Deskew — koreksi kemiringan halaman.

Metode klasik OpenCV: binerisasi Otsu, ambil semua piksel non-zero, cari
sudut kemiringan lewat minAreaRect, lalu rotasi balik dengan warpAffine
(background putih). Sudut kecil (< DESKEW_MIN_ANGLE) diabaikan untuk
menghindari rotasi yang tidak perlu yang hanya menambah artefak.
"""

from __future__ import annotations

import cv2
import numpy as np

# Rotasi hanya dilakukan jika kemiringan melebihi nilai ini (derajat).
DESKEW_MIN_ANGLE: float = 0.5


def deskew(image: np.ndarray) -> np.ndarray:
    """Koreksi kemiringan halaman; kembalikan array RGB dengan background putih."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Otsu inverted: teks jadi putih di atas hitam (memudahkan findNonZero).
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    coords = cv2.findNonZero(binary)
    if coords is None:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    # Normalisasi ambiguitas minAreaRect: sudut ~±90° pada halaman yang
    # sebenarnya lurus (kotak minimal full-page) TIDAK boleh memicu rotasi.
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    angle = -angle

    if abs(angle) < DESKEW_MIN_ANGLE:
        return image

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rotated
