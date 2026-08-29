# OCR Kwitansi — GLM-OCR Lokal

Pipeline OCR dokumen/kwitansi Indonesia (cetak & tulisan tangan) menggunakan
**GLM-OCR** (`zai-org/GLM-OCR`, VLM ±1B) yang dijalankan **100% lokal, in-process,
tanpa server** (tanpa Ollama/vLLM, tanpa API key, tanpa kredit).

## Arsitektur

```
Preprocessing (CLAHE/denoise/sharpen/resize)
  -> GLM-OCR lokal via HuggingFace Transformers (baca halaman PENUH sekali jalan)
     + region fallback (page besar -> split zona horizontal, OCR resolusi asli)
  -> Teks OCR per halaman (markdown rapi)
  -> Field extraction (kwitansi) — opsional lewat run_full_pipeline.py
```

Tidak ada lagi **segmentasi baris** seperti TrOCR — GLM-OCR memahami layout
halaman langsung, sehingga kuitansi tulisan tangan yang rapat sekalipun terbaca.

## Quick Start

```bash
# Persiapkan environment (sekali)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Batch semua contoh kwitansi
./run.sh

# Batch file tertentu
./run.sh "Kwitansi/Kwitansi_Sumber Jaya Fastindo.jpg"

# Pipeline end-to-end (OCR + field extraction)
python run_full_pipeline.py Kwitansi/Kuitansi_Ekadata.pdf
```

> Catatan: hasil OCR disimpan ke `results/{nama}.json` dan `results/{nama}.txt`.
> Di mesin CPU (~8 core) butuh ±1–2 menit per halaman.

## Model GLM-OCR Lokal

Model diunduh sekali ke folder lokal (offline setelahnya):

- Sumber resmi: [GitHub zai-org/GLM-OCR](https://github.com/zai-org/GLM-OCR) ·
  [HuggingFace](https://huggingface.co/zai-org/GLM-OCR) ·
  [ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-OCR) (licence MIT).
- Unduh ke `models/glm-ocr/` (berisi `config.json` + `model.safetensors` + tokenizer).
- Jalankan tanpa server: `GlmOcrModel` memuat model langsung di proses via
  `transformers` + `torch` (CPU float32 default).

## Environment Variables

| Variable | Default | Deskripsi |
|---|---|---|
| `ACTIVE_OCR_MODEL` | `GLM-OCR (lokal)` | Model aktif dari registry (`GLM-OCR (lokal)` \| `TrOCR (base-handwritten)`) |
| `GLM_OCR_MODEL_PATH` | `./models/glm-ocr` | Folder model GLM-OCR lokal |
| `GLM_OCR_DEVICE` | `cpu` | `cpu`, `cuda`, atau `auto` |
| `GLM_OCR_MAX_SIDE` | `1400` | Sisi terpanjang (px) sebelum OCR satu-pass (kecil = cepat) |
| `GLM_OCR_MAX_NEW_TOKENS` | `512` | Max token output per pass |
| `GLM_OCR_REGION_FALLBACK` | `1` | 1 = page besar yang hasilnya kosong di-OCR ulang per zona |
| `GLM_OCR_REGION_SPLITS` | `3` | Jumlah zona horizontal saat region fallback |
| `STRICT_QUALITY_GATE` | `0` | 1 = dokumen gagal quality check langsung ditolak |
| `SAMPLE_DIR` | `./Kwitansi` | Folder dokumen contoh |
| `RESULTS_DIR` | `./results` | Folder output hasil OCR |

Variabel lama TrOCR (`TROCR_MODEL_NAME`, `TROCR_DEVICE`, dst.) tetap ada hanya
jika `ACTIVE_OCR_MODEL=TrOCR (base-handwritten)` (backup).

## Struktur Project

```
models/
  glm_ocr_model.py     -> GlmOcrModel (OCR utama: full-page + region fallback)
  trocr_model.py       -> TrOCR (backup, segmentasi per baris)
  registry.py          -> Model registry + model aktif (ACTIVE_OCR_MODEL)
  base.py              -> BaseExtractionModel ABC + ModelResult
  layout_model.py      -> Optional layout detection (paddleocr)

preprocessing/
  pipeline.py          -> Preprocessing orchestrator
  contrast_enhancer.py -> CLAHE
  denoiser.py          -> Bilateral filter
  sharpener.py         -> Unsharp masking
  quality_check.py     -> Blur/resolution gate
  pdf_to_image.py      -> PDF rendering (PyMuPDF)

run_batch.py           -> CLI batch entry point
run_full_pipeline.py   -> End-to-end pipeline (+ field extraction)
config.py              -> Konfigurasi terpusat
compare_models.py      -> (legacy) komparasi manual TrOCR full vs LoRA
```