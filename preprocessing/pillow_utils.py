"""Helper bersama untuk konversi PIL <-> numpy/OpenCV dan simpan/muat gambar.

Konsistensi: seluruh preprocessing bekerja dengan array numpy ber-channel
RGB (bukan BGR) untuk menghindari salah-urutan channel yang sering terjadi
saat bergantian antara OpenCV (BGR) dan PIL (RGB). Muat/simpan selalu lewat
PIL agar robust terhadap path berkarakter non-ASCII (mis. path berbahasa
Indonesia di Windows), yang tidak ditangani dengan baik oleh cv2.imread.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def load_as_rgb_array(path: str) -> np.ndarray:
    """Muat gambar dari path sebagai array numpy uint8 ber-channel RGB."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.asarray(im)


def array_to_pil(array: np.ndarray) -> Image.Image:
    """Konversi array numpy (RGB, atau single-channel grayscale) ke PIL Image."""
    if array.ndim == 2:
        return Image.fromarray(array.astype("uint8"), mode="L")
    if array.ndim == 3:
        return Image.fromarray(array.astype("uint8"), mode="RGB")
    raise ValueError(f"Array berdimensi tidak dikenal: {array.shape}")


def save_array_as_image(array: np.ndarray, path: str) -> str:
    """Simpan array numpy (RGB atau grayscale) sebagai file gambar."""
    img = array_to_pil(array)
    img.save(path)
    return path
