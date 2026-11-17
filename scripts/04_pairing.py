"""04_pairing.py - RGB / IR / XML eslestirme kontrolu.

Eslestirme anahtari: dosya adi koku (stem)
Bu betik anahtarin gercekten bire-bir calisip calismadigini sayarak dogrular.

Cikti:
  outputs/logs/pairing_table.csv
  outputs/logs/pairing_summary.json
  outputs/figures/fig01_paired_samples.png
"""
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path("data/raw/LLVIP_extracted")
CSV_PATH = Path("outputs/logs/pairing_table.csv")
JSON_PATH = Path("outputs/logs/pairing_summary.json")
FIG_PATH = Path("outputs/figures/fig01_paired_samples.png")
SPLITS = ["train", "test"]


def stems(folder: Path, ext: str) -> set:
    return {p.stem for p in folder.glob(f"*{ext}")}


def main() -> None:
    ann = stems(ROOT / "Annotations", ".xml")

    rows, summary = [], {"n_xml_total": len(ann)}
    all_img_stems = set()

    for split in SPLITS:
        rgb = stems(ROOT / "visible" / split, ".jpg")
        ir = stems(ROOT / "infrared" / split, ".jpg")
        both = rgb & ir
        all_img_stems |= (rgb | ir)

        summary[split] = {
            "n_rgb": len(rgb),
            "n_ir": len(ir),
            "n_paired": len(both),
            "rgb_without_ir": len(rgb - ir),
            "ir_without_rgb": len(ir - rgb),
            "paired_without_xml": len(both - ann),
        }
        for s in sorted(rgb | ir):
            rows.append({
                "stem": s, "split": split,
                "has_rgb": s in rgb, "has_ir": s in ir, "has_xml": s in ann,
            })

    train_s = {r["stem"] for r in rows if r["split"] == "train"}
    test_s = {r["stem"] for r in rows if r["split"] == "test"}
    summary["train_test_stem_overlap"] = len(train_s & test_s)
    summary["xml_without_image"] = len(ann - all_img_stems)
    summary["stem_length_set"] = sorted({len(s) for s in all_img_stems})
    summary["all_stems_numeric"] = all(s.isdigit() for s in all_img_stems)

    df = pd.DataFrame(rows)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    # --- Numara yapisi: on ek (ilk 2 hane) ve bosluklar ---
    df["num"] = df["stem"].astype(int)
    df["prefix"] = df["stem"].str[:2]
    print("\n=== ON EK (ilk 2 hane) ANALIZI ===")
    for split in SPLITS:
        g = df[df["split"] == split].groupby("prefix")["num"].agg(
            ["count", "min", "max"]
        )
        g["beklenen"] = g["max"] - g["min"] + 1
        g["eksik_numara"] = g["beklenen"] - g["count"]
        print(f"\n[{split}]  on ek sayisi: {len(g)}")
        print(g.to_string())

    # --- Gorsel kanit: 2 train + 2 test rastgele cift ---
    random.seed(42)
    picks = []
    for split in SPLITS:
        pool = sorted(df[(df["split"] == split) & df["has_rgb"] & df["has_ir"]]["stem"])
        picks += [(split, s) for s in random.sample(pool, 2)]

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(picks), figsize=(4 * len(picks), 6))
    for j, (split, s) in enumerate(picks):
        rgb_img = Image.open(ROOT / "visible" / split / f"{s}.jpg")
        ir_img = Image.open(ROOT / "infrared" / split / f"{s}.jpg")
        axes[0, j].imshow(rgb_img)
        axes[0, j].set_title(f"RGB {split}/{s}", fontsize=9)
        axes[1, j].imshow(ir_img, cmap="gray" if ir_img.mode == "L" else None)
        axes[1, j].set_title(
            f"IR {split.upper()}/{s} ({ir_img.width} x {ir_img.height})",
            fontsize=9
        )
        axes[0, j].axis("off")
        axes[1, j].axis("off")
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    print(f"\nGorsel kaydedildi: {FIG_PATH}")
    print("Secilen ciftler:", picks)


if __name__ == "__main__":
    main()