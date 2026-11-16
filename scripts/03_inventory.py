"""03_inventory.py - Ham verinin dosya envanterini cikarir (varsayimsiz).

Cikti:
  outputs/logs/raw_inventory.csv    -> her dosya icin bir satir
  outputs/logs/raw_inventory_summary.json -> ozet sayilar
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path("data/raw/LLVIP_extracted")
CSV_PATH = Path("outputs/logs/raw_inventory.csv")
SUMMARY_PATH = Path("outputs/logs/raw_inventory_summary.json")


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"{ROOT} bulunamadi. Once 02_extract.py calistirin.")

    rows = []
    for p in ROOT.rglob("*"):
        if p.is_file():
            rel = p.relative_to(ROOT)
            rows.append(
                {
                    "rel_path": rel.as_posix(),
                    "parent_dir": rel.parent.as_posix(),
                    "stem": p.stem,
                    "ext": p.suffix.lower(),
                    "size_bytes": p.stat().st_size,
                }
            )

    df = pd.DataFrame(rows).sort_values("rel_path").reset_index(drop=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)

    # Klasor bazinda sayim (hangi klasorde kac dosya, hangi uzantilar)
    per_dir = (
        df.groupby(["parent_dir", "ext"]).size().reset_index(name="n_files")
    )
    print("=== KLASOR x UZANTI SAYIMI ===")
    print(per_dir.to_string(index=False))

    print("\n=== UZANTI SAYIMI (toplam) ===")
    print(df["ext"].value_counts().to_string())

    print("\n=== ILK 3 DOSYA / KLASOR (yapi kontrolu) ===")
    for d, g in df.groupby("parent_dir"):
        print(f"[{d}]  ornek: {g['rel_path'].head(3).tolist()}")

    summary = {
        "total_files": int(len(df)),
        "total_size_gb": round(df["size_bytes"].sum() / 1024**3, 3),
        "ext_counts": dict(Counter(df["ext"])),
        "n_dirs_with_files": int(df["parent_dir"].nunique()),
        "zero_byte_files": int((df["size_bytes"] == 0).sum()),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== OZET ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()