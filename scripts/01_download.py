"""01_download.py - LLVIP veri setini Google Drive'dan indirir ve kaydını tutar.
Ama internet nedeniyle manuel indirdim // 2.10.2026

Kaynak: https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md
Bu bağlantı resmi depodaki Google Drive dosyasının ID'si.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import gdown

FILE_ID = "1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv"   # resmi depodaki Google Drive ID
RAW_DIR = Path("data/raw")
LOG_PATH = Path("outputs/logs/01_download_log.json")


def sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Dosyanın SHA-256 özetini hesaplar (bütünlük ve izlenebilirlik için)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Klasördeki manuel indirilmiş dosyayı bul
    files = [p for p in RAW_DIR.iterdir() if p.is_file() and not p.name.startswith(".")]
    
    if not files:
        # Eğer dosya henüz konmadıysa gdown dene
        out_path = gdown.download(id=FILE_ID, output=str(RAW_DIR) + "/", quiet=False)
        if out_path is None:
            raise SystemExit("Dosyayı tarayıcıdan indirip data/raw/ içine koyun.")
        out_path = Path(out_path)
    else:
        out_path = files[0]

    record = {
        "file_id": FILE_ID,
        "downloaded_file": out_path.name,
        "size_bytes": out_path.stat().st_size,
        "sha256": sha256_of(out_path),
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
    }
    LOG_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()