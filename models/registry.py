"""Registry — satu-satunya daftar model yang aktif.

Untuk menambah/mengganti model: cukup import class baru dan tambahkan/ganti
satu baris di ``AVAILABLE_MODELS``. Tidak ada file lain yang perlu diubah.

Model utama default: GLM-OCR lokal (in-process, tanpa server). TrOCR tetap
terdaftar sebagai backup — pilih lewat env ACTIVE_OCR_MODEL.

Peran:
    GLM-OCR (lokal)        -> model OCR utama (read full-page, in-process).
    TrOCR (base-handwritten) -> backup (liga lama, segmentasi per baris).
"""

import config

from .glm_ocr_model import GlmOcrModel
from .trocr_model import TrOCRHandwrittenModel

AVAILABLE_MODELS = {
    "GLM-OCR (lokal)": GlmOcrModel,
    "TrOCR (base-handwritten)": TrOCRHandwrittenModel,
}

# Nama model yang dipakai pipeline (default GLM-OCR; ganti pakai ACTIVE_OCR_MODEL).
DEFAULT_OCR_MODEL = getattr(config, "ACTIVE_OCR_MODEL", "GLM-OCR (lokal)")


def get_ocr_model():
    """Buat instance model aktif sesuai DEFAULT_OCR_MODEL."""
    if DEFAULT_OCR_MODEL not in AVAILABLE_MODELS:
        raise KeyError(
            f"Model '{DEFAULT_OCR_MODEL}' tidak terdaftar. Pilihan: {list(AVAILABLE_MODELS)}"
        )
    return AVAILABLE_MODELS[DEFAULT_OCR_MODEL]()
