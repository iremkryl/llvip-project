"""02b_manual_provenance.py - Elle indirilen verinin izlenebilirlik kaydi.

Kullanim:
  python scripts/02b_manual_provenance.py                      (arsiv yoksa)
  python scripts/02b_manual_provenance.py "C:\\yol\\LLVIP.zip"   (arsiv varsa)
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("data/raw/LLVIP_extracted")
LOG_PATH = Path("outputs/logs/02b_manual_provenance.json")
DRIVE_URL = "https://drive.google.com/file/d/1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv/view"


def sha256_of(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    record = {
        "method": "manual browser download (Google Drive quota exceeded for gdown)",
        "source_url": DRIVE_URL,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "extracted_root": str(ROOT),
    }

    if len(sys.argv) > 1:
        archive = Path(sys.argv[1])
        record["archive_name"] = archive.name
        record["archive_size_bytes"] = archive.stat().st_size
        record["archive_sha256"] = sha256_of(archive)
    else:
        record["archive_name"] = None
        record["note"] = "Arsiv dosyasi saklanmadi; ozet yalnizca acilmis dosyalardan."

    # Acilmis dosyalarin toplu ozeti: dosya adi + boyut listesinin hash'i
    entries = sorted(
        f"{p.relative_to(ROOT).as_posix()}|{p.stat().st_size}"
        for p in ROOT.rglob("*") if p.is_file()
    )
    record["n_extracted_files"] = len(entries)
    record["file_list_sha256"] = hashlib.sha256("\n".join(entries).encode()).hexdigest()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()