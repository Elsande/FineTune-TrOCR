"""Tahap 5: Contrast enhancement.

CLAHE (Contrast Limited Adaptive Histogram Equalization) diterapkan pada
kanal luminansi:
- Gambar RGB : konversi ke LAB, CLAHE pada kanal L, gabung kembali.
- Grayscale : CLAHE langsung.

Berguna khususnya untuk dokumen foto HP dengan pencahayaan tidak rata.
"""

from __future__ import annotations

import cv2
import numpy as np

# ukuran grid CLAHE (crop kecil = adaptasi lokal lebih kuat)
CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID_SIZE: int = 8


def contrast_enhance(image: np.ndarray) -> np.ndarray:
    """Perbaiki kontras teks vs background dengan CLAHE."""
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=(CLAHE_TILE_GRID_SIZE, CLAHE_TILE_GRID_SIZE),
        )
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=(CLAHE_TILE_GRID_SIZE, CLAHE_TILE_GRID_SIZE),
    )
    return clahe.apply(image)
