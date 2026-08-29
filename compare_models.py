"""
compare_models.py
=================
Komparasi dua model TrOCR (trocr-indonesia vs trocr-indonesian-lora)
langsung pada dokumen kwitansi bahasa Indonesia.

Pemakaian:
    python compare_models.py                        # pakai default SAMPLE_DIR
    python compare_models.py file1 [file2 ...]      # dokumen tertentu

Output dikirim ke results/compare/ agar tidak menimpa hasil run_batch.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from models import trocr_model
from preprocessing.pipeline import preprocess
from preprocessing.pdf_to_image import render_pdf_pages
from preprocessing.pillow_utils import save_array_as_image

MODEL_A = "models/trocr-indonesia"           # full checkpoint (fine-tune penuh)
MODEL_B = "models/trocr-indonesian-lora"     # LoRA adapter di atas base model


def _load_image(input_path: str) -> str:
    """preprocess satu dokumen -> kembalikan path gambar bersih."""
    p = Path(input_path)
    if p.suffix.lower() == ".pdf":
        pages = render_pdf_pages(str(p), max_pages=config.PDF_MAX_PAGES)
        if not pages:
            raise RuntimeError("PDF tidak punya halaman")
        arr = pages[0]
    else:
        import numpy as np
        from PIL import Image

        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB"))

    res = preprocess(str(p), binarize=False, image_array=arr)
    if res.processed_image_path:
        return res.processed_image_path
    save_array_as_image(arr, "/tmp/trocr_compare_raw.png")
    return "/tmp/trocr_compare_raw.png"


def _ocr_txt(model, img_path: str) -> tuple[str, float]:
    t0 = time.time()
    res = model.run(img_path)
    elapsed = time.time() - t0
    if res is None or res.error:
        return f"[ERROR] {res.error if res else 'tidak ada hasil'}", elapsed
    return (res.text or "").strip(), elapsed


def discover(paths: list[str]) -> list[Path]:
    if paths:
        return [Path(p) for p in paths if Path(p).suffix.lower() in config.SUPPORTED_IMAGE_EXTS | {".pdf"}]
    return [
        p
        for p in sorted(config.SAMPLE_DIR.rglob("*"))
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_IMAGE_EXTS | {".pdf"}
    ]


def main() -> None:
    docs = discover(sys.argv[1:])
    if not docs:
        print(f"[SKIP] Tidak ada dokumen di {config.SAMPLE_DIR}")
        return

    # Proses hanya dokumen yang belum punya hasil di results/compare (kalau --resume).
    if "--resume" in sys.argv:
        out_dir = config.RESULTS_DIR / "compare"
        done = {p.stem for p in out_dir.glob("*.txt")} if out_dir.is_dir() else set()
        docs = [d for d in docs if d.stem not in done]
        print(f"[RESUME] {len(docs)} dokumen tersisa")

    if not docs:
        print("Semua dokumen sudah dikerjakan.")
        return

    print(f"[INFO] {len(docs)} dokumen | model A: {MODEL_A} | model B: {MODEL_B}\n")

    out_dir = config.RESULTS_DIR / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Memuat kedua model sekaligus (sekali jalan)...")
    model_a = trocr_model.TrOCRHandwrittenModel(model_name=MODEL_A, device="cpu")
    model_a.load()
    model_b = trocr_model.TrOCRHandwrittenModel(model_name=MODEL_B, device="cpu")
    model_b.load()
    print("[INFO] Kedua model siap.\n")

    for doc in docs:
        print("=" * 72)
        print(f"DOKUMEN: {doc.name}")
        try:
            img_path = _load_image(str(doc))
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] gagal siapkan gambar: {exc}\n")
            continue

        banner = f"=== {doc.name} ===\n"

        m1 = trocr_model.TrOCRHandwrittenModel  # noqa
        for label, model in (
            ("FULL CHECKPOINT (trocr-indonesia)", model_a),
            ("LoRA (trocr-indonesian-lora)", model_b),
        ):
            try:
                text, elapsed = _ocr_txt(model, img_path)
            except Exception as exc:  # noqa: BLE001
                text, elapsed = f"[ERROR] {exc}", 0.0
            print(f"--- {label} ({elapsed:.1f}s) ---")
            print(text if text else "(kosong)")
            print()
            banner += f"--- {label} ({elapsed:.1f}s) ---\n{text}\n\n"

        (out_dir / f"{doc.stem}.txt").write_text(banner, encoding="utf-8")

    print("=" * 72)
    print(f"Hasil tersimpan di: {out_dir}")


if __name__ == "__main__":
    main()