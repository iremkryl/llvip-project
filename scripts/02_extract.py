"""02_extract.py - Indirilen arsivi data/raw/LLVIP_extracted altina acar."""
import json
import sys
import zipfile
from pathlib import Path

RAW_DIR = Path("data/raw")
DEST = RAW_DIR / "LLVIP_extracted"
LOG_PATH = Path("outputs/logs/02_extract_log.json")


def main() -> None:
    candidates = [p for p in RAW_DIR.iterdir() if p.is_file()]
    if len(candidates) != 1:
        raise SystemExit(
            f"data/raw altinda tam 1 dosya bekleniyordu, bulunan: "
            f"{[p.name for p in candidates]}"
        )
    archive = candidates[0]

    if not zipfile.is_zipfile(archive):
        raise SystemExit(
            f"{archive.name} bir ZIP dosyasi degil. Dosya turunu bana soyle, "
            f"ona uygun acma yontemi ekleyelim."
        )

    DEST.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = zf.namelist()
        n_files_in_zip = sum(1 for m in members if not m.endswith("/"))
        zf.extractall(DEST)

    n_extracted = sum(1 for p in DEST.rglob("*") if p.is_file())
    record = {
        "archive": archive.name,
        "files_in_archive": n_files_in_zip,
        "files_extracted": n_extracted,
        "match": n_files_in_zip == n_extracted,
    }
    LOG_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    if not record["match"]:
        sys.exit("UYARI: arsivdeki ve acilan dosya sayilari esit degil!")


if __name__ == "__main__":
    main()