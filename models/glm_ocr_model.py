"""GLM-OCR (zai-org/GLM-OCR) sebagai model OCR utama — LOKAL, in-process, TANPA server.

GLM-OCR adalah VLM OCR (CogViT encoder + GLM-0.5B decoder, ±1B param) yang mampu
membaca HANYA dari satu prompt "Text Recognition:" pada SELURUH halaman sekaligus.
Beda dengan TrOCR (per-baris + segmentasi), GLM-OCR tidak butuh segmentasi baris:
cukup kirim gambar halaman penuh.

Strategi performa di CPU:
1. OCR satu pass: halaman di-resize ke sisi terpanjang <= GLM_OCR_MAX_SIDE agar
   jumlah token visual terkendali (generasi CPU cepat).
2. Region fallback (bila hasil kosong/pendek pd dokumen besar): halaman dipecah
   menjadi N zona horizontal (dengan overlap) lalu tiap zona di-OCR di resolusi
   asli — sesuai arsitektur dua-tahap resmi GLM-OCR (layout -> OCR per zona).

Model dimuat langsung via HuggingFace Transformers di proses ini — tidak ada
proses Ollama/vLLM terpisah, tidak ada API key, 100% offline setelah download.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any

import cv2
import numpy as np

from config import (
    GLM_OCR_DEVICE,
    GLM_OCR_MAX_NEW_TOKENS,
    GLM_OCR_MAX_SIDE,
    GLM_OCR_MODEL_PATH,
    GLM_OCR_REGION_FALLBACK,
    GLM_OCR_REGION_SPLITS,
    LAYOUT_ZONE_MARGIN,
    LAYOUT_ZONE_MAX_DIM,
)

from models.base import BaseExtractionModel, ModelResult

# Prompt resmi GLM-OCR untuk document parsing (recognize semua isi halaman).
OCR_PROMPT = "Text Recognition:"


class GlmOcrModel(BaseExtractionModel):
    """GLM-OCR lokal via transformers (in-process, CPU/CUDA, tanpa server)."""

    name = "GLM-OCR (lokal)"
    role = "main"

    def __init__(
        self,
        model_path: str = GLM_OCR_MODEL_PATH,
        device: str | None = None,
        max_new_tokens: int = GLM_OCR_MAX_NEW_TOKENS,
        max_side: int = GLM_OCR_MAX_SIDE,
        region_fallback: bool = GLM_OCR_REGION_FALLBACK,
        region_splits: int = GLM_OCR_REGION_SPLITS,
    ) -> None:
        self._model_path = model_path
        self._device_pref = device or GLM_OCR_DEVICE
        self._max_new_tokens = max_new_tokens
        self._max_side = max_side
        self._region_fallback = region_fallback
        self._region_splits = max(2, region_splits)
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self.device = "cpu"
        import tempfile
        self.tmp_dir = os.path.join(tempfile.gettempdir(), "glm_ocr_input")
        os.makedirs(self.tmp_dir, exist_ok=True)

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self._torch = torch
        print(f"[INFO] Memuat GLM-OCR (local): {self._model_path} ...")

        if self._device_pref == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self._device_pref

        torch_dtype = torch.float32 if self.device == "cpu" else torch.float16
        self._processor = AutoProcessor.from_pretrained(
            self._model_path, trust_remote_code=True
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        self._model.to(self.device)
        self._model.eval()
        print(f"[INFO] GLM-OCR siap di device: {self.device}")

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if self._torch is not None:
            try:
                if self.device == "cuda":
                    self._torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            self._torch = None

    # -- helper gambar ------------------------------------------------------
    def _read_bgr(self, image_path: str) -> np.ndarray:
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"Gambar tidak terbaca: {image_path}")
        return bgr

    def _resize_for_cpu(self, bgr: np.ndarray, max_side: int) -> np.ndarray:
        h, w = bgr.shape[:2]
        longest = max(h, w)
        if longest > max_side:
            f = max_side / float(longest)
            return cv2.resize(
                bgr, (int(w * f), int(h * f)), interpolation=cv2.INTER_CUBIC
            )
        return bgr

    # -- inference ----------------------------------------------------------
    def _generate(self, bgr: np.ndarray, max_new_tokens: int) -> str:
        """OCR satu gambar (BGR) via prompt "Text Recognition:"."""
        self.load()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tmp = os.path.join(self.tmp_dir, "glm_input.png")
        os.makedirs(self.tmp_dir, exist_ok=True)
        cv2.imwrite(tmp, rgb)  # ensures path pas utk processor
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": tmp},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        inputs.pop("token_type_ids", None)

        with self._torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        text = self._processor.decode(
            generated[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return (text or "").strip()

    def _ocr_bgr(self, bgr: np.ndarray, max_side: int | None = None) -> str:
        if max_side is None:
            max_side = self._max_side
        prepared = self._resize_for_cpu(bgr, max_side)
        return self._generate(prepared, self._max_new_tokens)

    def _split_bands(self, bgr: np.ndarray, n: int, overlap_ratio: float = 0.12) -> list[np.ndarray]:
        h, w = bgr.shape[:2]
        step = h / n
        bands: list[np.ndarray] = []
        for i in range(n):
            y0 = max(0, int(i * step - overlap_ratio * step))
            y1 = min(h, int((i + 1) * step + overlap_ratio * step))
            bands.append(bgr[y0:y1, :])
        return bands

    # -- kontrak utama ------------------------------------------------------
    def run(self, image_path: str, **kwargs) -> ModelResult:
        t0 = time.time()
        self.load()
        try:
            bgr = self._read_bgr(image_path)
            text = self._ocr_bgr(bgr)

            native_max = max(bgr.shape[:2])
            # Region fallback: hasil kosong/pendek pd dokumen besar -> OCR per zona.
            if (
                self._region_fallback
                and native_max >= self._max_side
                and len(text) < 40
            ):
                zones: list[np.ndarray] = []
                for i, band in enumerate(self._split_bands(bgr, self._region_splits)):
                    zone_text = self._ocr_bgr(band, max_side=max(self._max_side, band.shape[1]))
                    zones.append(zone_text)
                    print(f"  [GLM-OCR] zona {i + 1}/{self._region_splits}: {len(zone_text)} karakter")
                if any(len(z) > len(text) for z in zones):
                    text = "\n".join(z for z in zones if z.strip())

            lines = [
                {"bbox": None, "text": ln, "confidence": None}
                for ln in (text.splitlines() or [text])
                if ln.strip()
            ]
            extra = {
                "label": "full_page",
                "lines": lines,
                "kept_lines": lines,
                "total_lines": len(lines),
                "all_lines": [{"text": ln["text"], "confidence": None} for ln in lines],
                "all_text": text,
            }
            return ModelResult(text, self.name, time.time() - t0, extra=extra)
        except Exception as exc:  # noqa: BLE001
            return ModelResult("", self.name, time.time() - t0, error=str(exc))

    def run_region(
        self,
        image_path: str,
        box,
        scale: float = 1.0,
        label: str = "",
        margin: int | None = None,
    ) -> ModelResult:
        """OCR satu wilayah: crop bbox (+margin, opsional upscale) lalu OCR."""
        t0 = time.time()
        self.load()
        try:
            bgr = self._read_bgr(image_path)
            h, w = bgr.shape[:2]
            x0, y0, x1, y1 = [int(v) for v in box]
            pad = LAYOUT_ZONE_MARGIN if margin is None else margin
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
            if x1 <= x0 or y1 <= y0:
                return ModelResult("", self.name, time.time() - t0, extra={"label": label, "box": list(box), "scale": scale})

            crop = bgr[y0:y1, x0:x1]
            if scale > 0 and scale != 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            max_dim = max(crop.shape[:2])
            if max_dim > LAYOUT_ZONE_MAX_DIM:
                f = LAYOUT_ZONE_MAX_DIM / max_dim
                crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

            text = self._ocr_bgr(crop)
            lines = [
                {"bbox": None, "text": ln, "confidence": None}
                for ln in (text.splitlines() or [text])
                if ln.strip()
            ]
            extra = {
                "label": label,
                "box": list(box),
                "scale": scale,
                "lines": lines,
                "kept_lines": lines,
                "total_lines": len(lines),
                "all_lines": [{"text": ln["text"], "confidence": None} for ln in lines],
                "all_text": text,
            }
            return ModelResult(text, self.name, time.time() - t0, extra=extra)
        except Exception as exc:  # noqa: BLE001
            return ModelResult("", self.name, time.time() - t0, error=str(exc))

    def run_header(self, image_path: str, top_ratio: float = 0.22, scale: float = 1.5) -> ModelResult:
        """OCR ulang area HEADER (kop kwitansi) sebagai fallback ohne layout."""
        raw = cv2.imread(image_path)
        if raw is None:
            return ModelResult("", self.name, 0.0, error=f"Gambar tidak terbaca: {image_path}")
        h, w = raw.shape[:2]
        return self.run_region(image_path, [0, 0, w, int(h * top_ratio)], scale=scale, label="header_fallback")