"""TrOCR (microsoft/trocr-base-handwritten) sebagai sumber OCR — LOKAL.

Project ini dikhususkan untuk **handwritten kwitansi** (kuitansi tulisan tangan),
maka model diganti dari PaddleOCR-VL (API) menjadi TrOCR base-handwritten yang
dijalankan lokal via HuggingFace Transformers:

    TROCR_MODEL_NAME = microsoft/trocr-base-handwritten

TrOCR bekerja pada SATU BARIS teks per inference (input 384x384). Untuk OCR
full-page, gambar di-segmentasi per baris via horizontal projection profile,
lalu tiap baris di-OCR dan digabung sesuai urutan baca.

Fitur:
- CLAHE sebelum inference (enhance kontras tulisan tangan).
- Segmentasi baris otomatis (projection + fallback contour-based).
- ``run_region`` (crop zona + upscale) — kompatibel dengan pipeline lama.
- ``run_header`` (wide-band 22% atas).
- Mendukung checkpoint hasil fine-tuning kwitansi (set TROCR_MODEL_NAME ke
  path folder hasil `train_trocr.py`, mis. models/trocr-kwitansi/).

Catatan: TrOCR tidak menghasilkan bounding box — bbox selalu None. Confidence
diambil dari rata-rata token logit-softmax per baris (bukan skor palsu).
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from typing import Any

import cv2
import numpy as np

from config import (
    LAYOUT_ZONE_MARGIN,
    LAYOUT_ZONE_MAX_DIM,
    TROCR_DEVICE,
    TROCR_LINE_MAX_WIDTH,
    TROCR_LINE_MIN_HEIGHT,
    TROCR_LINE_TARGET_HEIGHT,
    TROCR_MODEL_NAME,
    TROCR_MAX_NEW_TOKENS,
    TROCR_NUM_BEAMS,
)

from models.base import BaseExtractionModel, ModelResult

# ---------------------------------------------------------------------------
# HELPER GAMBAR
# ---------------------------------------------------------------------------
def _apply_clahe(image_np: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """CLAHE pada channel Luminance (YCrCb). Input/output: OpenCV BGR."""
    if image_np is None or image_np.size == 0:
        return image_np
    if image_np.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image_np)
    ycrcb = cv2.cvtColor(image_np, cv2.COLOR_BGR2YCrCb)
    y_chan, cr_chan, cb_chan = cv2.split(ycrcb)
    y_enhanced = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(y_chan)
    merged = cv2.merge((y_enhanced, cr_chan, cb_chan))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)


def _normalize_line_crop(crop: np.ndarray, target_height: int, max_width: int) -> np.ndarray:
    """Normalisasi ukuran crop baris: tinggi seragam, lebar dibatasi."""
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_height, target_height), dtype=np.uint8)
    scale = target_height / float(h)
    new_w = int(w * scale)
    if new_w > max_width:
        new_w = max_width
        scale = new_w / float(w)
    target_w = max(1, new_w)
    return cv2.resize(crop, (target_w, target_height), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)


def segment_lines(
    image_bgr: np.ndarray,
    min_height: int = TROCR_LINE_MIN_HEIGHT,
    target_height: int = TROCR_LINE_TARGET_HEIGHT,
    max_width: int = TROCR_LINE_MAX_WIDTH,
) -> list[np.ndarray]:
    """Segmentasi gambar full-page menjadi crop per baris tulisan tangan.

    Metode: horizontal projection profile pada versi biner (adaptive threshold)
    gambar grayscale. Baris = run piksel "bertinta" yang terpisah oleh gap kosong.
    Tulisan tangan sering miring/menurun, jadi gap minimum longgar.

    Returns list crop BGR (sudah dinormalisasi tinggi -> target_height).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        ~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )
    # Gabung stroke horizontal satu baris agar profil lebih solid.
    kernel_w = max(10, image_bgr.shape[1] // 40)
    kernel_h = max(3, min_height // 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    profile = dilated.sum(axis=1) / 255.0
    threshold = max(1.0, float(profile.max()) * 0.02)
    ink_rows = profile > threshold

    crops: list[np.ndarray] = []
    in_line = False
    start = 0
    h, w = image_bgr.shape[:2]
    ink_list = list(ink_rows) + [False]
    for y in range(len(ink_list)):
        has_ink = bool(ink_list[y])
        if has_ink and not in_line:
            in_line = True
            start = y
        elif not has_ink and in_line:
            in_line = False
            end = y
            if end - start >= min_height:
                pad = 4
                y0 = max(0, start - pad)
                y1 = min(h, end + pad)
                crop = image_bgr[y0:y1, :]
                crops.append(_normalize_line_crop(crop, target_height, max_width))

    # Fallback: projection gagal menemukan baris -> satu crop penuh.
    if not crops:
        crops.append(_normalize_line_crop(image_bgr, target_height, max_width))
    return crops


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
class TrOCRHandwrittenModel(BaseExtractionModel):
    """TrOCR microsoft/trocr-base-handwritten (atau fine-tuned kwitansi)."""

    name = "TrOCR (base-handwritten)"
    role = "main"

    def __init__(
        self,
        model_name: str = TROCR_MODEL_NAME,
        device: str | None = None,
        max_new_tokens: int = TROCR_MAX_NEW_TOKENS,
        num_beams: int = TROCR_NUM_BEAMS,
    ) -> None:
        self._model_name = model_name
        self._device_pref = device or TROCR_DEVICE
        self._max_new_tokens = max_new_tokens
        self._num_beams = num_beams
        self._model: Any = None
        self._image_processor: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self.device: str = "cpu"
        self.tmp_dir = os.path.join(tempfile.gettempdir(), "trocr_kwitansi")
        os.makedirs(self.tmp_dir, exist_ok=True)

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import ViTImageProcessor, RobertaTokenizer, VisionEncoderDecoderModel

        self._torch = torch
        print(f"[INFO] Memuat TrOCR: {self._model_name} ...")

        # Load image processor + tokenizer dari model_name (bisa LoRA atau base)
        try:
            self._image_processor = ViTImageProcessor.from_pretrained(self._model_name)
        except Exception:
            from transformers import AutoImageProcessor
            self._image_processor = AutoImageProcessor.from_pretrained("microsoft/trocr-base-handwritten")

        try:
            self._tokenizer = RobertaTokenizer.from_pretrained(self._model_name, use_fast=False)
        except Exception:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-base-handwritten", use_fast=False)

        # Cek apakah ini LoRA checkpoint atau base model
        lora_bin = os.path.join(self._model_name, "lora_adapter.bin")
        lora_cfg = os.path.join(self._model_name, "lora_config.json")
        if os.path.isfile(lora_bin) and os.path.isfile(lora_cfg):
            # LoRA: load base model, apply LoRA via peft
            import json
            from peft import LoraConfig, get_peft_model, TaskType, PeftModel

            print(f"[INFO] Terdeteksi LoRA adapter, load base model + apply adapter")
            base_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

            with open(lora_cfg) as f:
                lora_info = json.load(f)

            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=lora_info["r"],
                lora_alpha=lora_info["alpha"],
                lora_dropout=0.1,
                target_modules=lora_info["target_modules"],
                bias="none",
            )

            # Apply LoRA ke base model
            lora_model = get_peft_model(base_model, lora_config)
            # Load LoRA weights
            lora_state = torch.load(lora_bin, map_location="cpu", weights_only=True)
            lora_model.load_state_dict(lora_state, strict=False)
            print(f"[INFO] LoRA adapter loaded ({len(lora_state)} weights)")
            # Simpan sebagai base_model untuk generate
            self._model = lora_model
        else:
            self._model = VisionEncoderDecoderModel.from_pretrained(self._model_name)

        if self._device_pref == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self._device_pref

        # Handle PeftModel vs regular model
        self._is_peft = hasattr(self._model, "base_model")
        if self._is_peft:
            self._model.base_model.to(self.device)
            self._model.base_model.eval()
        else:
            self._model.to(self.device)
            self._model.eval()
        print(f"[INFO] TrOCR siap di device: {self.device}")

    def unload(self) -> None:
        self._model = None
        self._image_processor = None
        self._tokenizer = None
        if self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            self._torch = None

    # -- helper -------------------------------------------------------------
    def _read_image(self, image_path: str) -> np.ndarray:
        raw = cv2.imread(image_path)
        if raw is None:
            raise ValueError(f"Gambar tidak terbaca: {image_path}")
        return _apply_clahe(raw)

    def _recognize_crop(self, crop_bgr: np.ndarray) -> dict:
        """OCR satu crop (baris/zona): return {text, confidence}."""
        self.load()
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pixel_values = self._image_processor(images=rgb, return_tensors="pt").pixel_values.to(self.device)

        # PeftModel.generate() butuh base_model.generate()
        gen_model = self._model.base_model if self._is_peft else self._model
        with self._torch.no_grad():
            generated = gen_model.generate(
                pixel_values,
                max_new_tokens=self._max_new_tokens,
                num_beams=self._num_beams,
                output_scores=True,
                return_dict_in_generate=True,
            )
        text = self._tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()

        # Confidence = rata-rata probabilitas token terpilih.
        confidence = 0.0
        try:
            scores = generated.scores
            if scores:
                seq = generated.sequences[0]
                probs = []
                for step_idx, score in enumerate(scores):
                    token_id = seq[step_idx + 1]
                    probs.append(float(self._torch.softmax(score, dim=-1)[0, token_id]))
                confidence = sum(probs) / len(probs) if probs else 0.0
        except Exception:  # noqa: BLE001 — confidence bersifat best-effort
            confidence = 0.0
        return {"text": text, "confidence": round(confidence, 4)}

    def _ocr_image_array(self, image_bgr: np.ndarray, line_mode: bool = True) -> tuple[list[dict], list[dict]]:
        """OCR array BGR: segmentasi baris (bila line_mode) lalu recognize.

        Returns (all_lines, kept_lines) dengan struktur
        ``{bbox, text, confidence}`` per baris (bbox=None; TrOCR tanpa detektor).
        """
        if line_mode and max(image_bgr.shape[:2]) > TROCR_LINE_TARGET_HEIGHT * 2:
            crops = segment_lines(image_bgr)
        else:
            crops = [_normalize_line_crop(image_bgr, TROCR_LINE_TARGET_HEIGHT, TROCR_LINE_MAX_WIDTH)]

        all_lines: list[dict] = []
        for crop in crops:
            rec = self._recognize_crop(crop)
            if rec["text"]:
                all_lines.append({"bbox": None, "text": rec["text"], "confidence": rec["confidence"]})
        kept = [ln for ln in all_lines]
        return all_lines, kept

    def _result_extra(self, lines: list[dict], label: str = "", box=None, scale: float = 1.0) -> dict:
        return {
            "label": label,
            "box": box,
            "scale": scale,
            "lines": lines,
            "kept_lines": lines,
            "total_lines": len(lines),
            "all_lines": [{"text": ln["text"], "confidence": ln["confidence"]} for ln in lines],
            "all_text": "\n".join(ln["text"] for ln in lines),
        }

    # -- kontrak ------------------------------------------------------------
    def run(self, image_path: str, **kwargs) -> ModelResult:
        t0 = time.time()
        self.load()
        try:
            enhanced = self._read_image(image_path)
            lines, _kept = self._ocr_image_array(enhanced, line_mode=True)
            text = "\n".join(ln["text"] for ln in lines)
            return ModelResult(text, self.name, time.time() - t0, extra=self._result_extra(lines))
        except Exception as exc:  # noqa: BLE001
            return ModelResult("", self.name, time.time() - t0, error=str(exc))

    def run_region(self, image_path: str, box, scale: float = 1.0, label: str = "", margin: int | None = None) -> ModelResult:
        """OCR satu wilayah (zona): crop bbox (+margin) lalu upscale opsional."""
        t0 = time.time()
        self.load()
        try:
            raw_img = cv2.imread(image_path)
            if raw_img is None:
                return ModelResult("", self.name, 0.0, error=f"Gambar tidak terbaca: {image_path}")
            h, w = raw_img.shape[:2]
            x0, y0, x1, y1 = [int(v) for v in box]
            pad = LAYOUT_ZONE_MARGIN if margin is None else margin
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
            if x1 <= x0 or y1 <= y0:
                return ModelResult("", self.name, time.time() - t0, extra={"label": label, "box": list(box), "scale": scale})

            crop = _apply_clahe(raw_img[y0:y1, x0:x1])
            if scale > 0 and scale != 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            max_dim = max(crop.shape[:2])
            if max_dim > LAYOUT_ZONE_MAX_DIM:
                f = LAYOUT_ZONE_MAX_DIM / max_dim
                crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

            lines, _kept = self._ocr_image_array(crop, line_mode=True)
            text = "\n".join(ln["text"] for ln in lines)
            return ModelResult(text, self.name, time.time() - t0, extra=self._result_extra(lines, label=label, box=list(box), scale=scale))
        except Exception as exc:  # noqa: BLE001
            return ModelResult("", self.name, time.time() - t0, error=str(exc))

    def run_header(self, image_path: str, top_ratio: float = 0.22, scale: float = 2.0) -> ModelResult:
        """OCR ulang area HEADER (kop kwitansi) sebagai fallback tanpa layout."""
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            return ModelResult("", self.name, 0.0, error=f"Gambar tidak terbaca: {image_path}")
        h, w = raw_img.shape[:2]
        return self.run_region(image_path, [0, 0, w, int(h * top_ratio)], scale=scale, label="header_fallback")
