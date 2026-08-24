#!/usr/bin/env python3
"""
run_full_pipeline.py
====================
End-to-end: PDF/IMG -> preprocessing -> TrOCR -> field extraction -> save results

Usage:
    python run_full_pipeline.py Kwitansi/Kwitansi_Sumber\ Jaya\ Fastindo.jpg
    python run_full_pipeline.py Kwitansi/Kuitansi_Ekadata.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import date
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from run_batch import get_ocr_model, get_layout_model
from preprocessing.pipeline import preprocess
from preprocessing.pdf_to_image import render_pdf_pages


# ---------------------------------------------------------------------------
# FIELD EXTRACTION (rule-based untuk kwitansi)
# ---------------------------------------------------------------------------
def extract_fields_from_ocr(ocr_data: dict) -> dict:
    """Extract structured fields dari OCR raw text kwitansi."""
    import re
    text = ""
    for page in ocr_data.get("ocr", {}).get("pages", []):
        text += page.get("all_text", "") + "\n"

    fields: dict = {}
    lines = text.split("\n")

    for i, line in enumerate(lines):
        ll = line.lower().strip()

        if any(kw in ll for kw in ("kwitansi", "kuitansi", "kwt/")):
            m = re.search(r"[:\s]([\w/\\-\.]+)", line)
            if m:
                fields["kwitansi_number"] = m.group(1).strip()

        if "tanggal" in ll or "tgl" in ll:
            m = re.search(r"[:\s](.+)", line)
            if m:
                fields["kwitansi_date"] = m.group(1).strip()

        if "terima dari" in ll or ("dari" in ll and "untuk" not in ll):
            m = re.search(r"(?:terima\s+dari|dari)\s+(.+)", line, re.IGNORECASE)
            if m:
                fields["received_from"] = m.group(1).strip()

        if any(kw in ll for kw in ("jumlah", "total", "grand total")):
            m = re.search(r"[\d\.,]+", line.replace("Rp", "").replace("rp", ""))
            if m:
                fields["total_value"] = m.group(0).replace(".", "").replace(",", "")

        if "rupiah" in ll:
            fields["amount_in_words"] = line.strip()

        if "bank" in ll:
            m = re.search(r"bank\s+(.+)", line, re.IGNORECASE)
            if m:
                fields["bank"] = m.group(1).strip()

        if "rekening" in ll or "rek." in ll:
            m = re.search(r"[\d\.]+", line)
            if m:
                fields["payment_account"] = m.group(0).replace(".", "")

        if "npwp" in ll:
            m = re.search(r"[\d\.\-]{10,}", line)
            if m:
                fields["npwp"] = m.group(0).strip()

    return fields


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------
async def run_pipeline(input_path: str):
    input_path = Path(input_path)
    print(f"\n{'='*60}")
    print(f"FULL PIPELINE: {input_path.name}")
    print(f"{'='*60}")

    # 1. OCR
    print("\n[1/3] OCR (TrOCR Handwritten)...")
    ocr_model = get_ocr_model()

    if input_path.suffix.lower() == ".pdf":
        print("  Rendering PDF pages...")
        page_images = render_pdf_pages(str(input_path), max_pages=config.PDF_MAX_PAGES)
        print(f"  OK: {len(page_images)} halaman dirender")
    else:
        from PIL import Image
        import numpy as np
        with Image.open(str(input_path)) as im:
            page_images = [np.asarray(im.convert("RGB"))]
        print("  OK: 1 halaman (gambar)")

    all_ocr_results = []
    for i, page_img in enumerate(page_images):
        print(f"  Halaman {i+1}/{len(page_images)}...")
        import tempfile
        import cv2
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, cv2.cvtColor(page_img, cv2.COLOR_RGB2BGR))
            preprocess_result = preprocess(tmp.name, binarize=False)
            if not preprocess_result.passed_quality_gate:
                print(f"    X Quality gate failed: {preprocess_result.reject_reason}")
                continue
            model_result = ocr_model.run(preprocess_result.processed_image_path)
            result = {
                "page": i + 1,
                "total_lines": len(model_result.extra.get("lines", [])) if model_result.extra else 0,
                "all_text": model_result.text,
                "elapsed_seconds": model_result.elapsed_seconds,
                "lines": model_result.extra.get("lines", []) if model_result.extra else [],
            }
            all_ocr_results.append(result)
            print(f"    OK: {result['total_lines']} baris")

    if not all_ocr_results:
        print("  X Tidak ada hasil OCR")
        return

    ocr_output = {
        "document_id": str(uuid.uuid4()),
        "file_name": input_path.name,
        "ocr": {"model": ocr_model.name, "pages": all_ocr_results},
        "preprocessing": {"preprocessed": True, "page_count": len(all_ocr_results)},
        "processing_time_ms": sum(r.get("elapsed_seconds", 0) for r in all_ocr_results) * 1000,
    }
    print(f"  OK: OCR selesai: {len(all_ocr_results)} halaman")

    # 2. Field extraction
    print("\n[2/3] Extract fields...")
    extracted_fields = extract_fields_from_ocr(ocr_output)
    full_text = "\n".join(r.get("all_text", "") for r in all_ocr_results)
    extracted_fields["raw_text_preview"] = full_text[:500]
    extracted_fields["doc_type"] = "kwitansi"
    print(f"  OK: {list(extracted_fields.keys())}")

    # 3. Save
    print("\n[3/3] Save results...")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ocr_output["extracted_fields"] = extracted_fields
    ocr_output["doc_type"] = "kwitansi"

    out_json = config.RESULTS_DIR / f"{input_path.stem}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ocr_output, f, indent=2, ensure_ascii=False)
    print(f"  -> JSON: {out_json}")

    out_txt = config.RESULTS_DIR / f"{input_path.stem}.txt"
    txt_lines = [f"=== {input_path.name} ===", f"Model: {ocr_model.name}", ""]
    for r in all_ocr_results:
        txt_lines.append(f"--- Halaman {r['page']} ---")
        txt_lines.append(r.get("all_text", ""))
        txt_lines.append("")
    out_txt.write_text("\n".join(txt_lines), encoding="utf-8")
    print(f"  -> TXT : {out_txt}")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    return {"ocr_output": ocr_output, "extracted_fields": extracted_fields}


def main():
    parser = argparse.ArgumentParser(description="Full pipeline: PDF/IMG -> TrOCR -> Fields")
    parser.add_argument("input_path", help="Path to PDF or image file")
    args = parser.parse_args()
    import asyncio
    asyncio.run(run_pipeline(args.input_path))


if __name__ == "__main__":
    main()
