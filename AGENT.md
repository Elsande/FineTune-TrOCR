# AGENT.md — Aturan Kerja di Project

Project OCR **handwritten kwitansi**: model utama adalah **TrOCR** (`microsoft/trocr-base-handwritten`) via HuggingFace Transformers. Model berjalan LOKAL (GPU/CPU), tidak perlu API server. Pipeline: preprocessing gambar -> TrOCR (segmentasi baris + OCR) -> hasil teks + baris per halaman.

## Struktur & Aturan Wajib

1. **Entry point**: `run_batch.py` (batch CLI) via `run.sh`.
   - `./run.sh` -> batch SEMUA dokumen contoh (folder `Kwitansi/`).
2. **Registry adalah satu-satunya daftar model** (`models/registry.py`).
   Saat ini hanya `TrOCR (base-handwritten)`. Jangan import class model spesifik dari
   `run_batch.py`. LayoutDetection dipakai sebagai helper OPSIONAL (impor
   langsung `models.layout_model`), bukan model registry.
3. **Setiap dokumen WAJIB lewat `preprocessing/pipeline.preprocess()`** sebelum
   dikirim ke TrOCR. Quality gate dihitung & dicatat;
   `STRICT_QUALITY_GATE=False` (default) => dokumen tetap diproses (anti miss).
4. **Alur per dokumen** (`run_batch.process_one`):
   - `prepare_image` -> daftar path halaman hasil preprocessing
     (PDF multi-halaman dirender, dibatasi `PDF_MAX_PAGES=8`).
   - Per halaman: `TrOCR (base-handwritten).run(page)` -> teks + `lines`.
     Error dicatat per halaman, dokumen TETAP diproses.
   - Hasil ditulis ke `results/<nama>.json` (teks + baris per halaman)
     dan `results/<nama>.txt` (teks polos).
5. **PENTING (jangan diubah tanpa alasan teknis)**:
   - Gambar WAJIB melewati preprocessing; deskew DINONAKTIFKAN
     (`USE_DESKEW=False` di `preprocessing/pipeline.py`).
   - Model TrOCR dimuat SEKALI via singleton `get_ocr_model()`.
     Jangan load/unload berulang.
   - LayoutDetection butuh package `paddleocr` yang TIDAK diinstal default;
     bila tidak tersedia, `get_layout_model()` return None (graceful, OCR tetap jalan).
   - Untuk fine-tuning kwitansi, jalankan `train_trocr.py` dan set
     `TROCR_MODEL_NAME` ke path hasil training.
6. **Tidak ada file lain yang boleh diubah saat menambah model** selain
   `models/` + `models/registry.py`.

## Perintah Verifikasi

- Batch semua dokumen: `./run.sh` atau `venv/bin/python run_batch.py`
- Batch dokumen tertentu: `./run.sh "Kwitansi/Kwitansi_Sumber Jaya Fastindo.jpg"`
- Fine-tuning: `python train_trocr.py --train_csv data/train.csv --output_dir models/trocr-kwitansi`

## Catatan Lingkungan

- Model: `microsoft/trocr-base-handwritten` (HuggingFace, lokal via PyTorch).
- `validation.py` dipertahankan tapi TIDAK dipanggil pipeline saat ini.
- `results/` tidak di-commit.
