"""Orkestrasi seluruh pipeline preprocessing.

Urutan TAHAPAN (WAJIB — jangan diubah tanpa alasan teknis yang didiskusikan):
    1. Konversi PDF -> gambar PER HALAMAN jika input berupa PDF (lihat
       ``render_pdf_pages``); `preprocess()` mengolah satu array per halaman.
    2. Quality check awal (blur score + resolusi).
    3. Quality gate — jika gagal, STOP (jangan diteruskan ke model apa pun).
    4. Deskew.
    5. Denoise (klasik, non-ML).
    6. Contrast enhancement (CLAHE).
    7. Binarisasi (KONDISIONAL — hanya jika `binarize=True`, default OFF).
    8. Sharpen (unsharp masking, sesudah denoise).
    9. Resize ke rentang ukuran wajar.

Fungsi `preprocess()` dipanggil SEKALI oleh app.py SEBELUM model manapun
dijalankan — model tidak boleh menerima file mentah dari upload user.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass

import numpy as np

from .binarizer import binarize as binarize_image
from .contrast_enhancer import contrast_enhance
from .denoiser import denoise
from .deskewer import deskew
from .pdf_to_image import pdf_to_image
from .pillow_utils import save_array_as_image
from .quality_check import check_quality
from .resizer import resize_to_range
from .sharpener import sharpen


@dataclass
class PreprocessResult:
    processed_image_path: str | None = None  # path gambar hasil, siap ke model
    passed_quality_gate: bool = False
    reject_reason: str | None = None  # diisi jika passed_quality_gate == False
    blur_score: float | None = None
    resolution: tuple[int, int] | None = None


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Deskew DINONAKTIFKAN secara default. Temuan project paddleocr (agent_guide.md):
# koreksi kemiringan berbasis minAreaRect dapat salah meng-rotasi halaman yang
# sebenarnya lurus (sudut ~±90°), yang justru menghilangkan baris teks kecil
# (header) dari hasil deteksi PaddleOCR. Aktifkan hanya bila dokumen benar-benar
# miring dan telah diuji.
USE_DESKEW: bool = False


def _is_pdf(input_path: str) -> bool:
    return os.path.splitext(input_path)[1].lower() == ".pdf"


def _new_output_dir() -> str:
    root = tempfile.gettempdir()
    out_dir = os.path.join(root, "preprocess_out", uuid.uuid4().hex)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _load_initial_array(input_path: str) -> np.ndarray:
    if _is_pdf(input_path):
        return pdf_to_image(input_path)
    return np.asarray(_load_rgb(input_path))


def _load_rgb(input_path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(input_path) as im:
        return np.asarray(im.convert("RGB"))


def preprocess(input_path: str, binarize: bool = False, image_array: np.ndarray | None = None) -> PreprocessResult:
    """Jalankan seluruh pipeline preprocessing untuk satu file input.

    Args:
        input_path: path file yang di-upload user (gambar atau PDF).
        binarize:   jika True, aktifkan tahap binarisasi (default OFF —
                    model VLM umumnya bekerja lebih baik dengan RGB/grayscale).
        image_array: array gambar yang SUDAH dirender (mis. satu halaman PDF
                    dari ``render_pdf_pages``). Bila diberikan, ``input_path``
                    tidak dibaca ulang.

    Returns:
        PreprocessResult. Jika gagal quality gate, `passed_quality_gate=False`
        dan `reject_reason` berisi alasan spesifik — JANGAN lanjut ke model.
    """
    try:
        image = _load_initial_array(input_path) if image_array is None else image_array
    except Exception as e:  # noqa: BLE001 — laporkan alasan konkret ke user
        return PreprocessResult(
            passed_quality_gate=False,
            reject_reason=f"Tidak dapat membaca file sebagai gambar/PDF: {e}",
        )

    # Tahap 2-3: quality check awal + quality gate (STOP jika gagal).
    passed, reason, blur_score, resolution = check_quality(image)
    if not passed:
        return PreprocessResult(
            passed_quality_gate=False,
            reject_reason=reason,
            blur_score=blur_score,
            resolution=resolution,
        )

    # Tahap 4-9: perbaikan gambar.
    if USE_DESKEW:
        image = deskew(image)
    image = denoise(image)
    image = contrast_enhance(image)
    if binarize:
        image = binarize_image(image)
    image = sharpen(image)
    image = resize_to_range(image)

    out_dir = _new_output_dir()
    out_path = os.path.join(out_dir, "processed.png")
    save_array_as_image(image, out_path)

    return PreprocessResult(
        processed_image_path=out_path,
        passed_quality_gate=True,
        blur_score=blur_score,
        resolution=resolution,
    )
