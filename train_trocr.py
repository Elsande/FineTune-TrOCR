#!/usr/bin/env python3
"""
train_trocr.py
==============
Fine-tune TrOCR (microsoft/trocr-base-handwritten) pada dataset
handwritten kwitansi.

Cara pakai:
    # 1. Siapkan dataset CSV (image_path, text)
    # 2. Jalankan training
    python train_trocr.py \\
        --train_csv data/train.csv \\
        --val_csv data/val.csv \\
        --output_dir models/trocr-kwitansi \\
        --epochs 50 \\
        --batch_size 8 \\
        --learning_rate 5e-5

    # 3. Set TROCR_MODEL_NAME untuk inference
    export TROCR_MODEL_NAME="./models/trocr-kwitansi"

Format CSV dataset:
    image_path,text
    data/imgs/kwitansi_001_line_01.png,Seratus Ribu Rupiah
    data/imgs/kwitansi_001_line_02.png,Kwitansi No. 001/ABC/2026

Preprocessing:
    Gambar baris harus sudah di-crop (satu baris per gambar, tinggi ~64-128px).
    Bila dataset belum di-crop, gunakan fungsi `prepare_dataset_images()`
    untuk segmentasi otomatis dari gambar kwitansi full-page.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ---------------------------------------------------------------------------
# DATASET PREPARATION
# ---------------------------------------------------------------------------
def prepare_dataset_images(
    images_dir: str,
    output_dir: str,
    target_height: int = 384,
    max_width: int = 1024,
) -> str:
    """Segmentasi gambar kwitansi full-page menjadi baris-baris terpisah.

    Setiap gambar input (full-page kwitansi) dipotong per baris menggunakan
    horizontal projection profile. Output: folder output_dir dengan crop
    per baris + file CSV (image_path, text) yang bisa diisi manual.

    Args:
        images_dir: Folder berisi gambar kwitansi (JPG/PNG) full-page.
        output_dir: Folder output untuk crop baris + CSV.
        target_height: Tinggi normalisasi setiap crop baris.
        max_width: Lebar maks per crop baris.

    Returns:
        Path ke file CSV yang dihasilkan.
    """
    import cv2
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    imgs_dir = Path(images_dir)
    out_path = Path(output_dir)
    csv_rows: list[dict] = []

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    for img_file in sorted(imgs_dir.rglob("*")):
        if not img_file.is_file() or img_file.suffix.lower() not in image_exts:
            continue

        print(f"  Segmentasi: {img_file.name}")
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            ~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
        )

        h, w = img.shape[:2]
        kernel_w = max(10, w // 40)
        kernel_h = max(3, 15 // 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        profile = dilated.sum(axis=1) / 255.0
        threshold = max(1.0, float(profile.max()) * 0.02)
        ink_rows = profile > threshold

        stem = img_file.stem
        line_idx = 0
        in_line = False
        start = 0
        ink_list = list(ink_rows) + [False]
        for y in range(len(ink_list)):
            has_ink = bool(ink_list[y])
            if has_ink and not in_line:
                in_line = True
                start = y
            elif not has_ink and in_line:
                in_line = False
                end = y
                if end - start >= 15:
                    pad = 4
                    y0 = max(0, start - pad)
                    y1 = min(h, end + pad)
                    crop = img[y0:y1, :]
                    crop_h, crop_w = crop.shape[:2]
                    scale = target_height / float(crop_h)
                    new_w = min(int(crop_w * scale), max_width)
                    resized = cv2.resize(crop, (max(1, new_w), target_height),
                                        interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
                    crop_name = f"{stem}_line_{line_idx:03d}.png"
                    crop_path = out_path / crop_name
                    cv2.imwrite(str(crop_path), resized)
                    csv_rows.append({
                        "image_path": str(crop_path),
                        "text": "",  # user isi manual
                    })
                    line_idx += 1

        if line_idx == 0:
            scale = target_height / float(h)
            new_w = min(int(w * scale), max_width)
            resized = cv2.resize(img, (max(1, new_w), target_height),
                                interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
            crop_name = f"{stem}_line_000.png"
            crop_path = out_path / crop_name
            cv2.imwrite(str(crop_path), resized)
            csv_rows.append({
                "image_path": str(crop_path),
                "text": "",
            })

    csv_file = out_path / "dataset.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "text"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n  Total baris tersegmentasi: {len(csv_rows)}")
    print(f"  CSV template: {csv_file}")
    print(f"  Silakan isi kolom 'text' dengan transkripsi manual, lalu jalankan:")
    print(f"    python train_trocr.py --train_csv {csv_file} --output_dir models/trocr-kwitansi")
    return str(csv_file)


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
def load_dataset(csv_path: str, val_split: float = 0.1):
    """Load dataset dari CSV (image_path, text)."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row["image_path"]
            text = row["text"].strip()
            if text and Path(img_path).is_file():
                rows.append({"image_path": img_path, "text": text})

    if not rows:
        raise ValueError(f"Tidak ada baris valid di {csv_path} (pastikan text tidak kosong)")

    import random
    random.shuffle(rows)
    split = max(1, int(len(rows) * val_split))
    return rows[split:], rows[:split] if split > 0 else rows[:1]


class KwitansiDataset:
    """Dataset PyTorch untuk fine-tuning TrOCR."""

    def __init__(self, data: list[dict], image_processor, tokenizer, max_length: int = 128):
        self.data = data
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from PIL import Image
        item = self.data[idx]
        img_path = item["image_path"]
        # Handle relative paths (relative to project dir)
        if not Path(img_path).is_absolute():
            img_path = str(PROJECT_DIR / img_path)
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.tokenizer(
            item["text"],
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        ).input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def train(args):
    """Fine-tune TrOCR pada dataset Indonesian handwriting (LoRA untuk hemat memori)."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import ViTImageProcessor, RobertaTokenizer, VisionEncoderDecoderModel
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"[INFO] Loading model: {args.model_name}")
    # Load components separately (compat transformers5)
    image_processor = ViTImageProcessor.from_pretrained(args.model_name)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # Setup model config (decoder tokens)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = tokenizer.sep_token_id

    # Setup generation_config (transformers5: jangan set di model.config)
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        decoder_start_token_id=tokenizer.cls_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.sep_token_id,
        max_length=args.max_length,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=2.0,
        num_beams=4,
    )

    # --- LoRA: fine-tune hanya ~0.1% parameter ---
    if args.lora:
        print("[INFO] Menggunakan LoRA untuk memory-efficient fine-tuning")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,                      # rank LoRA (default=8)
            lora_alpha=args.lora_alpha,          # scaling factor
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],  # attention layers di encoder & decoder
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Gradient checkpointing untuk hemat memori
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    model.to(device)

    # Optimizer: hanya LoRA params jika LoRA aktif
    if args.lora:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable_params = list(model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Load data
    train_data, val_data = load_dataset(args.train_csv, args.val_split)
    print(f"[INFO] Train: {len(train_data)} baris | Val: {len(val_data)} baris")

    train_ds = KwitansiDataset(train_data, image_processor, tokenizer, max_length=args.max_length)
    val_ds = KwitansiDataset(val_data, image_processor, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history: list[dict] = []

    n_params = sum(p.numel() for p in trainable_params)
    print(f"\n[INFO] Mulai training ({args.epochs} epochs, batch_size={args.batch_size}, lr={args.learning_rate})")
    print(f"[INFO] Trainable params: {n_params:,}")
    print(f"[INFO] Output: {output_dir}\n")

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch}/{args.epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / max(len(train_loader), 1)
        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

                # Sample prediction — PeftModel perlu base_model untuk generate
                gen_model = model.base_model if hasattr(model, "base_model") else model
                generated = gen_model.generate(pixel_values, max_new_tokens=args.max_length, num_beams=4)
                preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
                for i in range(len(preds)):
                    gt_labels = labels[i]
                    gt_tokens = [t for t in gt_labels if t != -100 and t != tokenizer.pad_token_id]
                    gt_text = tokenizer.decode(gt_tokens, skip_special_tokens=True)
                    if preds[i].strip().lower() == gt_text.strip().lower():
                        val_correct += 1
                    val_total += 1

        avg_val_loss = val_loss / max(len(val_loader), 1)
        accuracy = val_correct / max(val_total, 1)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2%} | LR: {lr_now:.2e}")

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "accuracy": round(accuracy, 4),
            "lr": lr_now,
        })

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  -> New best! Saving to {output_dir}")
            if args.lora:
                # Simpan LoRA adapter langsung dari state_dict
                lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
                torch.save(lora_state, str(output_dir / "lora_adapter.bin"))
                # Simpan config LoRA manual
                import json
                lora_info = {"r": args.lora_r, "alpha": args.lora_alpha,
                             "target_modules": ["q_proj", "v_proj"]}
                with open(output_dir / "lora_config.json", "w") as f:
                    json.dump(lora_info, f, indent=2)
            else:
                model.save_pretrained(str(output_dir))
            image_processor.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

    # Save history
    history_file = output_dir / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Training selesai. History: {history_file}")
    print(f"[INFO] Model terbaik: {output_dir}")
    print(f"[INFO] Untuk inference, set: TROCR_MODEL_NAME=\"{output_dir}\"")

    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR untuk handwritten kwitansi")
    sub = parser.add_subparsers(dest="command")

    # --- prepare ---
    p_prep = sub.add_parser("prepare", help="Segmentasi gambar full-page kwitansi menjadi baris-baris terpisah")
    p_prep.add_argument("--images_dir", required=True, help="Folder berisi gambar kwitansi full-page")
    p_prep.add_argument("--output_dir", default="data/segmented_lines", help="Folder output crop + CSV")

    # --- train ---
    p_train = sub.add_parser("train", help="Fine-tune TrOCR pada dataset kwitansi")
    p_train.add_argument("--train_csv", required=True, help="CSV training (image_path, text)")
    p_train.add_argument("--model_name", default="microsoft/trocr-base-handwritten", help="Base model HuggingFace atau path model hasil fine-tuning sebelumnya")
    p_train.add_argument("--output_dir", default="models/trocr-kwitansi", help="Folder output model hasil training")
    p_train.add_argument("--epochs", type=int, default=50, help="Jumlah epoch")
    p_train.add_argument("--batch_size", type=int, default=8, help="Batch size")
    p_train.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    p_train.add_argument("--max_length", type=int, default=128, help="Max token length")
    p_train.add_argument("--val_split", type=float, default=0.1, help="Persentase validasi")
    p_train.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    p_train.add_argument("--fp16", action="store_true", help="Gunakan mixed precision fp16 (butuh GPU)")
    p_train.add_argument("--gradient_checkpointing", action="store_true", help="Aktifkan gradient checkpointing (hemat memori)")
    p_train.add_argument("--lora", action="store_true", help="Gunakan LoRA (hemat memori ~70%)")
    p_train.add_argument("--lora_r", type=int, default=8, help="LoRA rank (default: 8)")
    p_train.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")

    # --- backward compat: tanpa subcommand ---
    parser.add_argument("--train_csv", dest="legacy_train_csv", help="(legacy) CSV training path")
    parser.add_argument("--output_dir", dest="legacy_output_dir", help="(legacy) output dir")

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_dataset_images(args.images_dir, args.output_dir)
    elif args.command == "train":
        train(args)
    elif args.legacy_train_csv:
        args.train_csv = args.legacy_train_csv
        args.output_dir = args.legacy_output_dir or "models/trocr-kwitansi"
        args.model_name = "microsoft/trocr-base-handwritten"
        args.epochs = 50
        args.batch_size = 8
        args.learning_rate = 5e-5
        args.max_length = 128
        args.val_split = 0.1
        args.max_grad_norm = 1.0
        args.fp16 = False
        args.gradient_checkpointing = False
        train(args)
    else:
        parser.print_help()
        print("\nContoh:")
        print("  # 1. Segmentasi gambar full-page jadi baris:")
        print("  python train_trocr.py prepare --images_dir Kwitansi/ --output_dir data/lines")
        print("  # 2. Isi kolom 'text' di data/lines/dataset.csv")
        print("  # 3. Training:")
        print("  python train_trocr.py train --train_csv data/lines/dataset.csv --output_dir models/trocr-kwitansi")


if __name__ == "__main__":
    main()
