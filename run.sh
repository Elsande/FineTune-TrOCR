#!/usr/bin/env bash
# run.sh — peluncur OCR GLM-OCR Kwitansi (batch CLI)
# Pakai:
#   ./run.sh                    -> batch SEMUA dokumen contoh
#   ./run.sh [file1 file2 ...]  -> batch dokumen tertentu
set -euo pipefail
cd "$(dirname "$0")"

PY="$PWD/venv/bin/python"
if [[ ! -x "$PY" ]]; then
    PY="$(command -v python3.12 || command -v python3)"
fi

exec "$PY" run_batch.py "$@"
