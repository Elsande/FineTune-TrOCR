"""Tahap 1: Konversi PDF -> gambar (per halaman).

Render dilakukan lewat PyMuPDF (pymupdf) yang self-contained dan tidak butuh
binary poppler tambahan. PDF multi-halaman dirender SEMUA halamannya
(``render_pdf_pages``) — dokumen bisnis nyata sering 2 halaman dan isinya
terbagi (header di halaman 1, isi PO di halaman 2).

Zoom render dipilih agar lebar output berada di sekitar MIN_RENDER_WIDTH
piksel, cukup untuk keterbacaan OCR tanpa terlalu besar sehingga lambat.
"""

from __future__ import annotations

import numpy as np
import pymupdf  # PyMuPDF


# Lebar target (piksel) hasil render halaman PDF.
MIN_RENDER_WIDTH = 1800


def pdf_to_image(pdf_path: str) -> np.ndarray:
    """Render halaman pertama PDF menjadi array numpy ber-channel RGB."""
    pages = render_pdf_pages(pdf_path, max_pages=1)
    return pages[0]


def render_pdf_pages(pdf_path: str, max_pages: int = 8) -> list[np.ndarray]:
    """Render SEMUA halaman PDF (dibatasi ``max_pages``) menjadi list array RGB.

    Dokumen PO nyata sering 2 halaman: halaman 1 cuma header (nomor PO/tanggal),
    sedangkan isi PO (supplier/buyer/item/total) ada di halaman 2. Maka PDF
    multi-halaman wajib dirender per halaman, bukan cuma halaman pertama.
    """
    doc = pymupdf.open(pdf_path)
    try:
        if doc.page_count < 1:
            raise ValueError("PDF tidak memiliki halaman sama sekali.")
        n = min(doc.page_count, max_pages)
        out: list[np.ndarray] = []
        for i in range(n):
            page = doc.load_page(i)

            # Hitung zoom agar lebar render >= MIN_RENDER_WIDTH.
            base_width = page.rect.width
            zoom = max(1.0, MIN_RENDER_WIDTH / base_width)
            matrix = pymupdf.Matrix(zoom, zoom)

            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = np.frombuffer(pix.samples, dtype=np.uint8)
            image = image.reshape(pix.height, pix.width, pix.n)
            # pix.n == 3 untuk RGB, == 1 untuk grayscale. Normalisasi ke RGB.
            if pix.n == 1:
                image = np.repeat(image, 3, axis=2)
            out.append(image)
        return out
    finally:
        doc.close()
