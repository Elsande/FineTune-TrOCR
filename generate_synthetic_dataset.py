#!/usr/bin/env python3
"""
generate_synthetic_dataset.py
==============================
Generate synthetic Indonesian handwriting dataset untuk fine-tune TrOCR.

Mengambil teks Indonesian, merender dengan berbagai handwriting-style fonts,
lalu augmentasi (noise, rotation, blur) agar menyerupai tulisan tangan asli.

Output: folder dengan gambar baris + file CSV (image_path, text).

Cara pakai:
    python generate_synthetic_dataset.py --output_dir data/synthetic_id --num_samples 5000
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# ---------------------------------------------------------------------------
# TEKS SAMPLE INDONESIAN
# ---------------------------------------------------------------------------

# Kata & frasa umum kwitansi / dokumen bisnis
_COMMON_WORDS = [
    "Kwitansi", "Kuitansi", "Telah", "Terima", "Dari", "Untuk",
    "Jumlah", "Rp", "Rupiah", "Bayar", "Tunai", "Transfer",
    "Bank", "Rekening", "Tanggal", "Nomor", "No", "Nomer",
    "Tanda", "Tangan", "Oleh", "Karena", "Untuk", "Pembayaran",
    "Barang", "Jasa", "Harga", "Satuan", "Total", "Subtotal",
    "Pajak", "PPN", "Diskon", "Potongan", "DPP", "Grand Total",
    "Seratus", "Dua Ribu", "Tiga Ribu", "Lima Ribu", "Sepuluh Ribu",
    "Dua Puluh Ribu", "Lima Puluh Ribu", "Seratus Ribu",
    "Dua Ratus Ribu", "Lima Ratus Ribu", "Satu Juta",
    "PT", "CV", "TBK", "Tbk", "Indonesia", "Jakarta", "Surabaya",
    "Bandung", "Semarang", "Medan", "Makassar", "Yogyakarta",
    "Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu",
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
    "Diterima", "Dibayar", "Ditagihkan", "Penerima", "Pembayar",
    "Kepada", "Alamat", "Kota", "Propinsi", "Kode", "Pos",
    "Telepon", "Fax", "Email", "Website",
    "Barang", "Produk", "Kuantitas", "Qty", "Unit", "Piece",
    "Box", "Pack", "Carton", "Karton", "Pcs",
    "Uang", "Cash", "Cheque", "Cek", "Giro",
    "Terbilang", "Terterakan", "Tertulis", "Tertandatangani",
]

# Angka & simbol yang sering muncul di kwitansi
_NUMBERS = [str(i) for i in range(100)] + [
    "1.000", "2.500", "5.000", "10.000", "15.000", "20.000",
    "25.000", "50.000", "75.000", "100.000", "150.000", "200.000",
    "250.000", "500.000", "750.000", "1.000.000", "2.500.000",
    "08123456789", "021-123456", "(021) 123456",
]

# Pola nomor dokumen
_DOC_NUMBERS = [
    "KWT/001/ABC/2026", "KWT/0685/ISA/VI/2026", "INV/001/2026",
    "PO/2026/001", "DO/001/2026", "FP/04002600198056261",
    "No. 001/KWT/2026", "No: 0685/ISA/KWT/VI/2026",
    "0029939626606000", "0315710376611000",
    "142.0010251626", "888.0012345678",
]

# Frasa lengkap (multi-kata)
_PHRASES = [
    "Telah terima dari",
    "Untuk pembayaran",
    "Sejumlah uang",
    "Tunai / Transfer",
    "Dengan ini menyatakan",
    "Barang telah diterima dengan baik",
    "Kondisi barang sesuai pesanan",
    "Terima kasih atas kerjasamanya",
    "Atas perhatiannya kami ucapkan terima kasih",
    "Harap pembayaran dilakukan sebelum jatuh tempo",
    "Bukti pembayaran ini merupakan kwitansi yang sah",
    "Segera lakukan pembayaran",
    "Nota ini harus dilampirkan",
    "Tidak ada pengembalian uang",
    "Berlaku untuk satu bulan kalender",
    "Harga belum termasuk PPN 11%",
    "Harga sudah termasuk PPN 11%",
    "Subtotal harga barang",
    "Total yang harus dibayar",
    "Jumlah yang terbilang",
    "Rp. 116.550,00",
    "Rp 1.250.000,00",
    "Rp 50.000,00",
    "Seratus Enam Belas Ribu Lima Ratus Lima Puluh Rupiah",
    "Satu Juta Dua Ratus Lima Puluh Ribu Rupiah",
    "Lima Puluh Ribu Rupiah",
    "Dua Ratus Ribu Ripiah",
    "Bank Mandiri",
    "Bank BCA",
    "Bank BRI",
    "Bank BNI",
    "Rekening: 142.0010251626",
    "a/n PT Surya Multi Cemerlang",
    "Kepada Yth.",
    "PT Inti Solusindo Abadi",
    "Jl. Raya Bekasi Km 28",
    "Jakarta Timur 13910",
    "Telp: 021-888-1234",
    "Email: finance@company.co.id",
]

ALL_TEXTS = _COMMON_WORDS + _NUMBERS + _DOC_NUMBERS + _PHRASES


# ---------------------------------------------------------------------------
# FONT LOADING
# ---------------------------------------------------------------------------
def load_fonts(fonts_dir: str) -> list[tuple[str, ImageFont.FreeTypeFont]]:
    """Load semua .ttf font dari folder."""
    fonts = []
    font_dir = Path(fonts_dir)
    if not font_dir.exists():
        print(f"[WARN] Folder font tidak ditemukan: {font_dir}")
        return fonts
    for f in sorted(font_dir.glob("*.ttf")):
        try:
            font = ImageFont.truetype(str(f), 48)
            fonts.append((f.stem, font))
            print(f"  Loaded: {f.name}")
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")
    return fonts


# ---------------------------------------------------------------------------
# AUGMENTATION
# ---------------------------------------------------------------------------
def augment_image(img_np: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """Augmentasi gambar agar menyerupai tulisan tangan asli."""
    result = img_np.copy()

    # 1. Gaussian noise (reaksi kamera/scan)
    if random.random() < 0.7:
        noise = np.random.normal(0, random.uniform(2, 8) * intensity, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 2. Blur ringan (tidak fokus)
    if random.random() < 0.4:
        k = random.choice([3, 5])
        result = cv2.GaussianBlur(result, (k, k), 0)

    # 3. Brightness variation
    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2)
        result = cv2.convertScaleAbs(result, alpha=alpha)

    # 4. Slight rotation (tulisan tangan jarang lurus sempurna) - kurangi rentang agar tidak memotong teks
    if random.random() < 0.5:
        angle = random.uniform(-1.5, 1.5) * intensity
        h, w = result.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h), borderValue=(255, 255, 255))

    # 5. Perspective warp ringan (kertas sedikit melengkung) - kurangi intensitas
    if random.random() < 0.2:
        h, w = result.shape[:2]
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dx = int(w * 0.01 * intensity)
        dy = int(h * 0.01 * intensity)
        pts2 = np.float32([
            [random.randint(0, dx), random.randint(0, dy)],
            [w - random.randint(0, dx), random.randint(0, dy)],
            [random.randint(0, dx), h - random.randint(0, dy)],
            [w - random.randint(0, dx), h - random.randint(0, dy)],
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(result, M, (w, h), borderValue=(255, 255, 255))

    # 6. Compression artifact (JPEG simulation)
    if random.random() < 0.3:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(60, 90)]
        result = cv2.imencode('.jpg', result, encode_param)[1]
        result = cv2.imdecode(result, cv2.IMREAD_COLOR)

    return result


# ---------------------------------------------------------------------------
# GENERATE
# ---------------------------------------------------------------------------
def generate_line_image(
    text: str,
    font: ImageFont.FreeTypeFont,
    font_name: str,
    target_height: int = 64,
    max_width: int = 1600,
) -> np.ndarray | None:
    """Render satu baris teks dengan font tertentu, return gambar BGR."""
    # Render teks dengan PIL
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    if text_w < 5 or text_h < 5:
        return None

    # Buat gambar dengan padding yang lebih besar (khususnya bawah untuk mencegah clipping descender)
    # Beberapa font memiliki descender/glyph yang melebihi bbox, jadi tambah padding ekstra
    # Padding diperbesar (horizontal 40, vertikal 50) sebagai safety margin untuk augmentasi rotasi/warp
    # Caveat font khususnya memiliki descender yang jauh melebihi bbox
    pad_x, pad_y = 40, 50
    img_w = text_w + 2 * pad_x
    img_h = text_h + 2 * pad_y

    img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((pad_x, pad_y), text, font=font, fill=(0, 0, 0))

    # Resize ke target height
    scale = target_height / img_h
    new_w = max(1, int(img_w * scale))
    
    # Jika lebar melebihi max_width, skala ulang berdasarkan lebar
    if new_w > max_width:
        scale = max_width / img_w
        new_w = max_width
    
    img = img.resize((new_w, target_height), Image.LANCZOS)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def generate_dataset(
    output_dir: str,
    num_samples: int = 5000,
    fonts_dir: str = "fonts",
    target_height: int = 64,
    augment: bool = True,
) -> str:
    """Generate synthetic Indonesian handwriting dataset.

    Returns path ke file CSV.
    """
    out_path = Path(output_dir)
    imgs_dir = out_path / "images"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading fonts dari {fonts_dir}...")
    fonts = load_fonts(fonts_dir)
    if not fonts:
        print("[ERROR] Tidak ada font tersedia! Download font dulu.")
        return ""

    print(f"[INFO] Generating {num_samples} samples...")
    rows: list[dict] = []

    for i in range(num_samples):
        text = random.choice(ALL_TEXTS)

        # Kadang gabung 2-3 kata jadi kalimat
        if random.random() < 0.3:
            parts = [random.choice(_COMMON_WORDS) for _ in range(random.randint(2, 4))]
            text = " ".join(parts)

        font_name, font = random.choice(fonts)
        font_size = random.randint(28, 72)

        # Re-create font dengan ukuran acak
        try:
            font_path = Path(fonts_dir) / f"{font_name}.ttf"
            font = ImageFont.truetype(str(font_path), font_size)
        except Exception:
            pass

        img = generate_line_image(text, font, font_name, target_height=target_height)
        if img is None:
            continue

        if augment:
            intensity = random.uniform(0.5, 1.5)
            img = augment_image(img, intensity=intensity)

        img_name = f"syn_{i:06d}_{font_name}.png"
        cv2.imwrite(str(imgs_dir / img_name), img)

        rows.append({
            "image_path": str(imgs_dir / img_name),
            "text": text,
        })

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{num_samples} generated...")

    # Write CSV
    csv_path = out_path / "dataset.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[INFO] Selesai!")
    print(f"  Total samples: {len(rows)}")
    print(f"  Output: {out_path}")
    print(f"  CSV: {csv_path}")
    print(f"  Images: {imgs_dir}")
    return str(csv_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Indonesian handwriting dataset")
    parser.add_argument("--output_dir", default="data/synthetic_id", help="Folder output")
    parser.add_argument("--num_samples", type=int, default=5000, help="Jumlah sample")
    parser.add_argument("--fonts_dir", default="fonts", help="Folder font .ttf")
    parser.add_argument("--target_height", type=int, default=64, help="Tinggi gambar output")
    parser.add_argument("--no_augment", action="store_true", help="Tanpa augmentasi")
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        fonts_dir=args.fonts_dir,
        target_height=args.target_height,
        augment=not args.no_augment,
    )


if __name__ == "__main__":
    main()
