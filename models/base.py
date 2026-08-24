"""Interface dasar untuk semua model ekstraksi di nu-paddle.

Meniru pola AI-Document/models/base.py, diperluas untuk hasil terstruktur.

Role model:
    "main"    -> selalu dijalankan, sumber hasil akhir (TrOCR / LLM).
    "support" -> pendukung model utama (LayoutDetection, dll).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelResult:
    """Hasil eksekusi satu model pada satu gambar.

    Attributes
    ----------
    text:
        Teks mentah hasil model (untuk LLM: JSON string; untuk
        OCR engine: teks yang digabung).
    model_name:
        Nama model yang menghasilkan.
    elapsed_seconds:
        Waktu eksekusi (untuk perbandingan performa).
    fields:
        Hasil terstruktur berupa dict field (diisi model utama).
    error:
        Diisi jika proses gagal; ``None`` jika sukses.
    extra:
        Data tambahan yang dikonsumsi selector (mis. ``{"lines": [...]}``
        berisi bbox + confidence per baris OCR).
    """
    text: str
    model_name: str
    elapsed_seconds: float
    fields: Optional[dict] = None
    error: Optional[str] = None
    extra: Optional[dict] = None


class BaseExtractionModel(ABC):
    """Kontrak wajib untuk semua model di nu-paddle."""

    name: str = ""       # nama tampilan, wajib diisi tiap subclass
    role: str = "main"   # "main" | "support"

    @abstractmethod
    def load(self) -> None:
        """Load model ke memori. Dipanggil sesaat sebelum ``run()``."""
        ...

    @abstractmethod
    def run(self, image_path: str, **kwargs) -> ModelResult:
        """Jalankan inference pada satu gambar, kembalikan ``ModelResult``."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Lepas model dari memori (bebaskan RAM)."""
        ...
