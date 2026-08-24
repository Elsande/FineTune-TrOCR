"""
config.py
=========
Konfigurasi terpusat untuk project OCR TrOCR (Handwritten Kwitansi).

Pipeline:
    preprocessing (PDF -> halaman, quality gate, denoise/CLAHE/sharpen/resize)
      -> TrOCR (base-handwritten) LOCAL via HuggingFace Transformers
      -> teks OCR + baris/level per halaman
      -> (opsional) LayoutDetection PP-DocLayoutV3 + OCR per zona layout

Model dijalankan LOKAL (tidak perlu server/API). Hanya LayoutDetection yang
opsional (butuh package paddleocr yang TIDAK diinstal default).
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# TrOCR (HuggingFace local — microsoft/trocr-base-handwritten)
# ---------------------------------------------------------------------------
# Set TROCR_MODEL_NAME ke path folder hasil fine-tuning untuk model kwitansi custom.
# Mis. TROCR_MODEL_NAME="./models/trocr-kwitansi" (folder berisi config.json + pytorch_model.bin).
# Atau biarkan default untuk pakai model pretrained dari HuggingFace Hub.
TROCR_MODEL_NAME = os.environ.get(
    "TROCR_MODEL_NAME", "microsoft/trocr-base-handwritten"
)
# "auto" -> pilih cuda/cpu otomatis; "cpu" -> paksa CPU; "cuda" -> paksa GPU.
TROCR_DEVICE = os.environ.get("TROCR_DEVICE", "auto")
TROCR_MAX_NEW_TOKENS = int(os.environ.get("TROCR_MAX_NEW_TOKENS", "128"))
TROCR_NUM_BEAMS = int(os.environ.get("TROCR_NUM_BEAMS", "4"))

# Segmentasi baris gambar full-page.
TROCR_LINE_MIN_HEIGHT = int(os.environ.get("TROCR_LINE_MIN_HEIGHT", "15"))
TROCR_LINE_TARGET_HEIGHT = int(os.environ.get("TROCR_LINE_TARGET_HEIGHT", "384"))
TROCR_LINE_MAX_WIDTH = int(os.environ.get("TROCR_LINE_MAX_WIDTH", "1024"))

# Baris OCR dengan confidence di bawah ini dibuang dari output bersih.
OCR_CONFIDENCE_THRESHOLD = float(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.0"))

# ---------------------------------------------------------------------------
# PREPROCESSING (sebelum dikirim ke TrOCR)
# ---------------------------------------------------------------------------
# True  -> dokumen yang gagal quality gate DITOLAK (tidak diproses).
# False -> quality metrics tetap dicatat, dokumen tetap diproses (anti miss).
STRICT_QUALITY_GATE = os.environ.get("STRICT_QUALITY_GATE", "0") == "1"

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Batas jumlah halaman PDF yang diproses (kuitansi umumnya 1 halaman).
PDF_MAX_PAGES = int(os.environ.get("PDF_MAX_PAGES", "8"))

# --- Quality check thresholds ---
BLUR_THRESHOLD = float(os.environ.get("BLUR_THRESHOLD", "3.0"))
MIN_RESOLUTION_WIDTH = int(os.environ.get("MIN_RESOLUTION_WIDTH", "50"))
MIN_RESOLUTION_HEIGHT = int(os.environ.get("MIN_RESOLUTION_HEIGHT", "50"))

# --- CLAHE (Contrast Limited Adaptive Histogram Equalization) ---
CLAHE_CLIP_LIMIT = float(os.environ.get("CLAHE_CLIP_LIMIT", "2.0"))
CLAHE_TILE_GRID_SIZE = int(os.environ.get("CLAHE_TILE_GRID_SIZE", "8"))

# --- Denoising ---
DENOISE_METHOD = os.environ.get("DENOISE_METHOD", "bilateral")  # "bilateral" | "fastNlMeans"
DENOISE_BILATERAL_D = int(os.environ.get("DENOISE_BILATERAL_D", "5"))
DENOISE_BILATERAL_SIGMA_COLOR = int(os.environ.get("DENOISE_BILATERAL_SIGMA_COLOR", "75"))
DENOISE_BILATERAL_SIGMA_SPACE = int(os.environ.get("DENOISE_BILATERAL_SIGMA_SPACE", "75"))
DENOISE_FASTNLMEANS_H = int(os.environ.get("DENOISE_FASTNLMEANS_H", "10"))
DENOISE_FASTNLMEANS_TEMPLATE_WINDOW = int(os.environ.get("DENOISE_FASTNLMEANS_TEMPLATE_WINDOW", "7"))
DENOISE_FASTNLMEANS_SEARCH_WINDOW = int(os.environ.get("DENOISE_FASTNLMEANS_SEARCH_WINDOW", "21"))

# --- Sharpening (unsharp mask) ---
SHARPEN_AMOUNT = float(os.environ.get("SHARPEN_AMOUNT", "1.0"))
SHARPEN_SIGMA = float(os.environ.get("SHARPEN_SIGMA", "1.0"))

# --- Binarization ---
BINARIZE_METHOD = os.environ.get("BINARIZE_METHOD", "adaptive")  # "adaptive" | "otsu"
BINARIZE_BLOCK_SIZE = int(os.environ.get("BINARIZE_BLOCK_SIZE", "31"))
BINARIZE_C = int(os.environ.get("BINARIZE_C", "10"))

# --- Resize ---
RESIZE_MAX_SIDE = int(os.environ.get("RESIZE_MAX_SIDE", "1600"))
RESIZE_MIN_SIDE = int(os.environ.get("RESIZE_MIN_SIDE", "800"))

# --- PDF render ---
PDF_RENDER_MIN_WIDTH = int(os.environ.get("PDF_RENDER_MIN_WIDTH", "1800"))

# --- Spatial line grouping tolerance (sort_boxes_spatially) ---
SORT_GROUP_TOLERANCE_RATIO = float(os.environ.get("SORT_GROUP_TOLERANCE_RATIO", "0.5"))
SORT_GROUP_TOLERANCE_FALLBACK = int(os.environ.get("SORT_GROUP_TOLERANCE_FALLBACK", "10"))

# ---------------------------------------------------------------------------
# LAYOUT ANALYSIS — PP-DocLayoutV3 (deteksi zona, offline, OPSIONAL)
# ---------------------------------------------------------------------------
# LayoutDetection butuh package `paddleocr` yang tidak diinstal default.
# Bila tidak tersedia, deteksi zona dilewati (OCR full-page tetap jalan).
LAYOUT_ENABLED = os.environ.get("LAYOUT_ENABLED", "0") == "1"
LAYOUT_MODEL_NAME = "PP-DocLayoutV3"

# Zona yang ditambah OCR per-layout (DI ATAS OCR halaman penuh).
LAYOUT_ZONE_TARGETS = frozenset({"table"})

# Faktor upscale saat OCR per zona (teks kecil butuh perbesaran lebih besar).
LAYOUT_ZONE_SCALE = {
    "header_image": 3.0,
    "table": 1.8,
    "header": 1.0,
    "text": 1.0,
    "footer": 1.0,
}

# Padding (px) di sekitar bbox zona saat crop, agar teks tepi tidak terpotong.
LAYOUT_ZONE_MARGIN = 20

# Ukuran maksimal sisi terpanjang hasil upscale zona (batasi runtime OCR zona).
LAYOUT_ZONE_MAX_DIM = 2400

# Fallback bila deteksi layout gagal/kosong (OCR ulang area header).
LAYOUT_HEADER_FALLBACK_TOP_RATIO = 0.22
LAYOUT_HEADER_FALLBACK_SCALE = 2.0

# ---------------------------------------------------------------------------
# DIREKTORI
# ---------------------------------------------------------------------------
SAMPLE_DIR = Path(os.environ.get("SAMPLE_DIR", str(PROJECT_DIR / "Kwitansi")))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(PROJECT_DIR / "results")))
