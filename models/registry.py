"""Registry — satu-satunya daftar model yang aktif.

Untuk menambah/mengganti model: cukup import class baru dan tambahkan/ganti
satu baris di ``AVAILABLE_MODELS``. Tidak ada file lain yang perlu diubah.

Peran:
    TrOCR (base-handwritten) -> model OCR utama (HuggingFace local),
    dikhususkan untuk handwritten kwitansi.
"""

from .trocr_model import TrOCRHandwrittenModel

AVAILABLE_MODELS = {
    "TrOCR (base-handwritten)": TrOCRHandwrittenModel,
}
