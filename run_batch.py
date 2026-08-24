"""
run_batch.py
============
Entry point OCR TrOCR (Handwritten Kwitansi): batch extraction (CLI).

Pemakaian:
    python run_batch.py                        # batch SEMUA dokumen contoh
    python run_batch.py [file1 file2 ...]      # batch dokumen tertentu

Alur per dokumen:
    preprocessing (PDF -> halaman, quality gate, denoise/CLAHE/sharpen/resize)
      -> TrOCR (base-handwritten) per halaman: teks OCR + baris
      -> (opsional) LayoutDetection PP-DocLayoutV3: OCR tambahan per zona
      -> hasil JSON + TXT ke results/

Model dimuat SEKALI (singleton) lalu dipakai ulang.
Hanya memakai models.registry.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from models.registry import AVAILABLE_MODELS
from preprocessing.pipeline import preprocess
from preprocessing.pdf_to_image import pdf_to_image
from preprocessing.pillow_utils import save_array_as_image

SUPPORTED_EXTS = config.SUPPORTED_IMAGE_EXTS | {".pdf"}


# ---------------------------------------------------------------------------
# MODEL SINGLETON
# ---------------------------------------------------------------------------
_OCR: object | None = None
_LAYOUT: object | None = None


def get_ocr_model():
    """TrOCR singleton — model lokal, tidak perlu API."""
    global _OCR
    if _OCR is None:
        _OCR = AVAILABLE_MODELS["TrOCR (base-handwritten)"]()
        _OCR.load()
    return _OCR


def get_layout_model():
    """LayoutDetection PP-DocLayoutV3 (opsional)."""
    global _LAYOUT
    if not config.LAYOUT_ENABLED:
        return None
    if _LAYOUT is None:
        from models.layout_model import LayoutModel
        _LAYOUT = LayoutModel()
        try:
            _LAYOUT.load()
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] LayoutDetection tidak tersedia: {exc}")
            _LAYOUT = None
    return _LAYOUT


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _load_initial_array(input_path: str):
    if Path(input_path).suffix.lower() == ".pdf":
        return pdf_to_image(input_path)
    from PIL import Image
    import numpy as np
    with Image.open(input_path) as im:
        return np.asarray(im.convert("RGB"))


def prepare_image(input_path: str):
    """preprocess SEMUA halaman (PDF multi-halaman didukung)."""
    from preprocessing.pdf_to_image import render_pdf_pages

    if Path(input_path).suffix.lower() == ".pdf":
        try:
            pages = render_pdf_pages(input_path, max_pages=config.PDF_MAX_PAGES)
        except Exception as exc:  # noqa: BLE001
            return None, {"reject_reason": f"Gagal merender PDF: {exc}", "page_count": 0}
        if not pages:
            return None, {"reject_reason": "PDF tidak memiliki halaman yang bisa dirender", "page_count": 0}
    else:
        try:
            pages = [_load_initial_array(input_path)]
        except Exception as exc:  # noqa: BLE001
            return None, {"reject_reason": f"Gambar tidak terbaca: {exc}", "page_count": 0}

    out_paths: list[str] = []
    meta: dict = {
        "preprocessed": True,
        "passed_quality_gate": True,
        "page_count": len(pages),
        "preprocessed_pages": 0,
        "preprocessed_skipped": False,
        "blur_scores": [],
        "resolutions": [],
    }
    for i, arr in enumerate(pages, start=1):
        res = preprocess(input_path, binarize=False, image_array=arr)
        if res.processed_image_path:
            out_paths.append(res.processed_image_path)
            meta["preprocessed_pages"] += 1
            meta["passed_quality_gate"] = meta["passed_quality_gate"] and res.passed_quality_gate
            if res.blur_score is not None:
                meta["blur_scores"].append(round(res.blur_score, 3))
            if res.resolution:
                meta["resolutions"].append(list(res.resolution))
            continue

        if config.STRICT_QUALITY_GATE:
            meta["passed_quality_gate"] = False
            meta["reject_reason"] = res.reject_reason
            continue
        try:
            out_dir = os.path.join(tempfile.gettempdir(), "trocr_fallback", uuid.uuid4().hex)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"page_{i}.png")
            save_array_as_image(arr, out_path)
            out_paths.append(out_path)
            meta["preprocessed_pages"] += 1
            meta["preprocessed_skipped"] = True
        except Exception as exc:  # noqa: BLE001
            meta["fallback_error"] = str(exc)
            print(f"  [WARN] Fallback render gagal untuk halaman {i}: {exc}")

    if not out_paths:
        meta["preprocessed"] = False
        meta.setdefault("reject_reason", meta.get("reject_reason") or "Gagal menyiapkan gambar")
        return None, meta
    return out_paths, meta


def discover_documents(paths: list[str] | None = None) -> list[Path]:
    if paths:
        return [Path(p) for p in paths if Path(p).suffix.lower() in SUPPORTED_EXTS]
    docs: list[Path] = []
    if config.SAMPLE_DIR.is_dir():
        for p in sorted(config.SAMPLE_DIR.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                docs.append(p)
    return docs


def _merge_unique_lines(lines: list[dict], additions: list[dict]) -> list[dict]:
    out = list(lines)
    seen = {_normalize_line_key(ln.get("text") or "") for ln in out}
    for ln in additions:
        key = _normalize_line_key(ln.get("text") or "")
        if key and key not in seen:
            seen.add(key)
            out.append(ln)
    return out


def _normalize_line_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


# ---------------------------------------------------------------------------
# PROSES SATU DOKUMEN
# ---------------------------------------------------------------------------
def _empty_result(doc: Path, error: str) -> dict:
    return {
        "document_id": str(uuid.uuid4()),
        "file_name": doc.name,
        "ocr": {"model": None, "source": "local", "pages": [], "total_lines": 0, "elapsed_seconds": 0.0},
        "preprocessing": {"reject_reason": error},
        "processing_time_ms": 0.0,
        "error": error,
    }


def process_one(doc: Path, ocr_model, layout_model) -> dict:
    t_start = time.time()

    image_paths, prep_meta = prepare_image(str(doc))
    if image_paths is None:
        return _empty_result(doc, prep_meta.get("reject_reason") or "Gagal menyiapkan gambar")

    ocr_pages: list[dict] = []
    ocr_elapsed = 0.0
    for i, img in enumerate(image_paths, start=1):
        page: dict = {"page": i}

        res = ocr_model.run(img)
        if res is None or res.error:
            page["error"] = res.error if res is not None else "OCR tidak aktif"
            ocr_pages.append(page)
            continue

        extra = res.extra or {}
        lines = list(extra.get("lines") or [])
        page["text"] = (res.text or "").strip()
        page["all_text"] = (extra.get("all_text") or "").strip()
        page["lines"] = lines
        page["elapsed_seconds"] = round(res.elapsed_seconds, 2)
        ocr_elapsed += res.elapsed_seconds

        if layout_model is not None:
            zones = layout_model.detect_zones(img)
            zone_records: list[dict] = []
            merged = list(lines)
            for z in zones:
                rec = {"label": z.label, "box": z.box, "score": round(z.score, 3), "ocr_text": ""}
                if z.label in config.LAYOUT_ZONE_TARGETS:
                    scale = config.LAYOUT_ZONE_SCALE.get(z.label, 1.0)
                    zres = ocr_model.run_region(img, z.box, scale=scale, label=z.label)
                    if zres is not None and not zres.error:
                        rec["ocr_text"] = (zres.text or "").strip()
                        merged = _merge_unique_lines(merged, list((zres.extra or {}).get("lines") or []))
                zone_records.append(rec)
            page["layout_zones"] = zone_records
            page["lines"] = merged
            page["all_text"] = "\n".join(ln.get("text") or "" for ln in merged)

        ocr_pages.append(page)

    total_lines = sum(len(p.get("lines") or []) for p in ocr_pages)
    elapsed_ms = round((time.time() - t_start) * 1000, 2)

    result = {
        "document_id": str(uuid.uuid4()),
        "file_name": doc.name,
        "ocr": {
            "model": ocr_model.name,
            "source": "local",
            "pages": ocr_pages,
            "total_lines": total_lines,
            "elapsed_seconds": round(ocr_elapsed, 2),
        },
        "preprocessing": prep_meta,
        "processing_time_ms": elapsed_ms,
        "error": None,
    }

    # ---- SIMPAN HASIL JSON ----
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = config.RESULTS_DIR / f"{doc.stem}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ---- SIMPAN HASIL RAW TEXT ----
    txt_lines = [f"=== {doc.name} ===", f"Model: {ocr_model.name}", f"Total halaman: {len(ocr_pages)}", ""]
    for p in ocr_pages:
        txt_lines.append(f"--- Halaman {p['page']} ---")
        if p.get("error"):
            txt_lines.append(f"[ERROR] {p['error']}")
        else:
            txt_lines.append(p.get("text") or "")
        txt_lines.append("")
    out_txt = config.RESULTS_DIR / f"{doc.stem}.txt"
    out_txt.write_text("\n".join(txt_lines), encoding="utf-8")

    print(f"  -> JSON: {out_json}")
    print(f"  -> TXT : {out_txt}")

    return result


def run(paths: list[str] | None = None) -> list[dict]:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    documents = discover_documents(paths)
    if not documents:
        print(f"[SKIP] Tidak ada dokumen ditemukan di {config.SAMPLE_DIR}")
        return []

    print(f"[INFO] {len(documents)} dokumen akan diproses\n")
    for i, doc in enumerate(documents, 1):
        shown = doc.relative_to(config.SAMPLE_DIR) if str(doc).startswith(str(config.SAMPLE_DIR)) else doc
        print(f"  {i:>2}. {shown}")

    ocr_model = get_ocr_model()
    layout_model = get_layout_model()
    print(f"\n[INFO] Model OCR: {ocr_model.name} | Layout: {'aktif' if layout_model is not None else 'nonaktif'}")
    print("[INFO] Memulai ekstraksi...\n")

    results: list[dict] = []
    for doc in documents:
        print(f"\n{'=' * 70}\nDOKUMEN: {doc.name}")
        result = process_one(doc, ocr_model, layout_model)
        results.append(result)
        _print_result(result)

    print(f"\n{'=' * 70}\nRINGKASAN\n{'=' * 70}")
    n_ok = sum(1 for r in results if not r.get("error"))
    for r in results:
        err = r.get("error")
        print(f"  [{'OK' if not err else 'ERR':>4}] {r['file_name']:<40} {len(r['ocr']['pages'])} hal / {r['ocr']['total_lines']} baris{'  (' + err + ')' if err else ''}")
    print(f"\n  OK: {n_ok}/{len(results)}")
    print(f"  Hasil tersimpan di: {config.RESULTS_DIR}")
    return results


def _print_result(result: dict) -> None:
    print(f"  waktu      : {result['processing_time_ms']} ms")
    if result.get("error"):
        print(f"  ERROR      : {result['error']}")
    for p in result["ocr"].get("pages", []):
        text = p.get("text") or ""
        n = len(p.get("lines") or [])
        print(f"  halaman {p['page']}: {n} baris, {len(text)} karakter")
        preview = " | ".join(text.splitlines()[:3])
        if preview:
            print(f"    > {preview[:200]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR TrOCR (Handwritten Kwitansi): batch")
    parser.add_argument("docs", nargs="*", help="path dokumen tertentu (opsional; default: semua contoh kwitansi)")
    args = parser.parse_args()
    run(args.docs or None)


if __name__ == "__main__":
    main()
