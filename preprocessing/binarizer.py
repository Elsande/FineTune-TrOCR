"""Tahap 6: Binarisasi — KONDISIONAL (opsional per-model).

Konversi ke hitam-putih untuk mempertegas teks. PENTING: langkah ini BERSIFAT
OPSIONAL dan TIDAK otomatis memaksa gambar final menjadi binary.

Model OCR berbasis transformer (TrOCR, VLM, dll.) sering bekerja
lebih baik dengan gambar RGB/grayscale utuh daripada binary murni. Karena itu
hasil binarisasi disimpan sebagai opsi toggle (default OFF), sampai ada
validasi eksperimen dari user bahwa binarisasi meningkatkan akurasi kedua
model target.
"""

from __future__ import annotations

import cv2
import numpy as np

# Metode thresholding: "adaptive" (adaptiveThreshold Gaussian) atau "otsu".
BINARIZE_METHOD: str = "adaptive"


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def binarize(image: np.ndarray) -> np.ndarray:
    """Binerisasi gambar menjadi hitam-putih (output single-channel)."""
    gray = _to_gray(image)

    if BINARIZE_METHOD == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    # adaptiveThreshold lebih tahan terhadap pencahayaan tidak rata.
    # blockSize harus ganjil; dibatasi minimal 31.
    block_size = 31
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        10,
    )
