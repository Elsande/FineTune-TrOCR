"""Kontrak dasar untuk tiap tahap preprocessing.

Semua tahap menerima array gambar (numpy, channel RGB) dan mengembalikan
array gambar hasil proses. Tahap yang beroperasi pada grayscale akan
mengonversi sendiri di dalam implementasinya.

Catatan: file ini bersifat opsional (kontrak ringan). Tahap-tahap konkret
boleh dipanggil langsung lewat fungsinya masing-masing di pipeline.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PreprocessStep(ABC):
    """Kontrak minimal untuk satu tahap preprocessing."""

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Terima array gambar RGB, kembalikan array gambar hasil proses."""
        raise NotImplementedError
