# OCR Handwritten Kwitansi — TrOCR

Pipeline OCR untuk **kwitansi tulisan tangan** menggunakan model **TrOCR** (microsoft/trocr-base-handwritten) via HuggingFace Transformers.

## Arsitektur

```
Preprocessing (CLAHE/denoise/sharpen)
  -> TrOCR (base-handwritten) LOCAL
  -> Teks OCR + baris per halaman
  -> Field extraction (kwitansi)
```

Model berjalan **100% lokal** — tidak perlu API server. Model dimuat ke GPU/CPU via PyTorch + Transformers.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Batch semua contoh kwitansi
./run.sh

# Batch file tertentu
./run.sh Kwitansi/Kwitansi_Sumber\ Jaya\ Fastindo.jpg

# Pipeline end-to-end
python run_full_pipeline.py Kwitansi/Kuitansi_Ekadata.pdf
```

## Fine-Tuning untuk Kwitansi

Model default `microsoft/trocr-base-handwritten` sudah dilatih pada dataset handwriting umum. Untuk hasil terbaik pada kwitansi spesifik, lakukan fine-tuning:

```bash
python train_trocr.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --output_dir models/trocr-kwitansi \
    --epochs 50 \
    --batch_size 8 \
    --learning_rate 5e-5
```

Setelah fine-tuning, set environment variable:
```bash
export TROCR_MODEL_NAME="./models/trocr-kwitansi"
./run.sh
```

## Format Dataset Fine-Tuning

CSV dengan kolom `image_path` dan `text`:
```csv
image_path,text
data/imgs/kwitansi_001_line_01.png,Seratus Ribu Rupiah
data/imgs/kwitansi_001_line_02.png,Kwitansi No. 001/ABC/2026
```

## Environment Variables

| Variable | Default | Deskripsi |
|---|---|---|
| `TROCR_MODEL_NAME` | `microsoft/trocr-base-handwritten` | Model HuggingFace atau path folder hasil fine-tuning |
| `TROCR_DEVICE` | `auto` | `auto`, `cuda`, atau `cpu` |
| `TROCR_MAX_NEW_TOKENS` | `128` | Max token output per baris |
| `TROCR_NUM_BEAMS` | `4` | Beam search width |
| `TROCR_LINE_MIN_HEIGHT` | `15` | Min tinggi baris terdeteksi (px) |
| `TROCR_LINE_TARGET_HEIGHT` | `384` | Tinggi normalisasi baris input model |
| `TROCR_LINE_MAX_WIDTH` | `1024` | Lebar maks crop baris |
| `SAMPLE_DIR` | `./Kwitansi` | Folder dokumen contoh |
| `RESULTS_DIR` | `./results` | Folder output hasil OCR |

## Struktur Project

```
models/
  trocr_model.py      -> Model TrOCR (OCR engine utama)
  base.py             -> BaseExtractionModel ABC
  registry.py         -> Model registry (satu model aktif)
  layout_model.py     -> Optional layout detection (paddleocr)

preprocessing/
  pipeline.py         -> Preprocessing orchestrator
  contrast_enhancer.py -> CLAHE
  denoiser.py         -> Bilateral filter
  sharpener.py        -> Unsharp masking
  quality_check.py    -> Blur/resolution gate
  pdf_to_image.py     -> PDF rendering (PyMuPDF)

train_trocr.py        -> Script fine-tuning TrOCR
run_batch.py          -> CLI batch entry point
run_full_pipeline.py  -> End-to-end pipeline
config.py             -> Konfigurasi terpusat
```
