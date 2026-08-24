"""Deteksi zona layout dokumen (PP-DocLayoutV3) — model lokal, offline.

Menghasilkan zona seperti ``header_image`` (logo), ``header``, ``table``,
``text``, ``footer``. Dipakai untuk OCR per-layout: crop tiap zona + upscale
sesuai jenis zona (lihat ``config.LAYOUT_ZONE_SCALE``).

Model dimuat sekali (singleton di ``run_batch.get_layout_model()``). Visualisasi
TIDAK dipakai — kita hanya membaca ``res.json`` (tanpa render, tanpa download
font tambahan).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import LAYOUT_MODEL_NAME


@dataclass
class Zone:
    """Satu zona layout hasil deteksi."""

    label: str
    box: list[int]   # [x0, y0, x1, y1]
    score: float


class LayoutModel:
    """Wrapper PP-DocLayoutV3 (deteksi zona layout)."""

    def __init__(self, model_name: str = LAYOUT_MODEL_NAME):
        self._model_name = model_name
        self._engine = None

    def load(self) -> None:
        if self._engine is not None:
            return
        from paddleocr import LayoutDetection

        self._engine = LayoutDetection(model_name=self._model_name)

    def unload(self) -> None:
        self._engine = None

    def detect_zones(self, image_path: str) -> list[Zone]:
        """Deteksi zona layout dari *image_path*.

        Returns
        -------
        list[Zone]
            Zona terdeteksi (label + bbox + score). Kosong bila gagal.
        """
        self.load()
        zones: list[Zone] = []
        try:
            result = self._engine.predict(image_path)[0]
            data = result.json.get("res", {})
            for box in data.get("boxes", []):
                label = box.get("label")
                coord = box.get("coordinate")
                if not label or not coord or len(coord) != 4:
                    continue
                zones.append(
                    Zone(
                        label=label,
                        box=[int(v) for v in coord],
                        score=float(box.get("score", 0.0)),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] LayoutDetection gagal: {exc}")
        return zones
