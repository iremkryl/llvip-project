
"""
ADIM 15A - GECERSIZ ANOTASYONLARIN TEMIZLENMESI

AMAC:
LLVIP veri setindeki gecersiz bounding box
kayitlarini temizlemek ve yapilan islemin
veri uzerindeki etkisini belgelemek.

YONTEM:
1. Ham anotasyon envanteri okunur.
2. Sifir/negatif boyutlu, eksik koordinatli veya
   goruntu sinirlari disindaki kutular belirlenir.
3. Gecerli ve gecersiz kutular ayri kaydedilir.
4. Train/test veri kaybi sayisal olarak raporlanir.
5. Hatalarin bulundugu goruntulerde, anotasyonlarin
   temizleme oncesi ve sonrasi gosterilir.
6. Ham goruntuler ve XML dosyalari degistirilmez.

GIRDI:
outputs/logs/boxes_table.csv
data/raw/LLVIP_extracted/visible/train/

CIKTI:
data/processed/boxes_cleaned.csv
data/processed/removed_invalid_boxes.csv
outputs/logs/step15a_cleaning_summary.json
outputs/figures/fig31_annotations_before_after.png
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib.patches import Rectangle

ROOT = Path("data/raw/LLVIP_extracted")
LOG = Path("outputs/logs")
PROCESSED = Path("data/processed")
FIG = Path("outputs/figures")

PROCESSED.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# 1. Anotasyon envanterini oku
df = pd.read_csv(
    LOG / "boxes_table.csv",
    dtype={"stem": str, "prefix": str}
)

df["stem"] = df["stem"].str.zfill(6)

coordinates = [
    "xmin", "ymin", "xmax", "ymax",
    "w", "h", "rgb_w", "rgb_h"
]

for col in coordinates:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 2. Gecerlilik kosullari
valid = (
    df[coordinates].notna().all(axis=1) &
    (df["w"] > 0) &
    (df["h"] > 0) &
    (df["xmin"] >= 0) &
    (df["ymin"] >= 0) &
    (df["xmax"] <= df["rgb_w"]) &
    (df["ymax"] <= df["rgb_h"])
)

df["is_valid"] = valid

clean = df.loc[valid].drop(
    columns=["is_valid"]
).copy()

removed = df.loc[~valid].drop(
    columns=["is_valid"]
).copy()

# 3. Temizlenmis anotasyon tablosunu kaydet
clean_path = PROCESSED / "boxes_cleaned.csv"
removed_path = PROCESSED / "removed_invalid_boxes.csv"

clean.to_csv(clean_path, index=False)
removed.to_csv(removed_path, index=False)

print("===== ADIM 15A - TEMIZLEME =====")
print(f"Ham kutu sayisi: {len(df)}")
print(f"Gecerli kutu sayisi: {len(clean)}")
print(f"Cikarilan kutu sayisi: {len(removed)}")

summary = {
    "raw_boxes": int(len(df)),
    "clean_boxes": int(len(clean)),
    "removed_boxes": int(len(removed)),
    "affected_images": int(
        removed[["split", "stem"]]
        .drop_duplicates().shape[0]
    ),
    "split_details": {}
}

for split in ["train", "test"]:
    raw_n = int((df["split"] == split).sum())
    clean_n = int((clean["split"] == split).sum())

    summary["split_details"][split] = {
        "before": raw_n,
        "after": clean_n,
        "removed": raw_n - clean_n
    }

    print(f"\n{split.upper()}")
    print(f"Once: {raw_n}")
    print(f"Sonra: {clean_n}")
    print(f"Cikarilan: {raw_n - clean_n}")

summary_path = LOG / "step15a_cleaning_summary.json"

summary_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

# 4. Temizleme oncesi/sonrasi gorsellestirme
# Kirmizi X: gecersiz kutu koordinati
# Yesil dikdortgen: gecerli insan kutusu

if len(removed) > 0:

    fig, axes = plt.subplots(
        len(removed), 2,
        figsize=(11, 3.3 * len(removed)),
        squeeze=False
    )

    for i, bad in enumerate(
        removed.itertuples(index=False)
    ):

        stem = bad.stem
        split = bad.split

        image_path = (
            ROOT / "visible" / split / f"{stem}.jpg"
        )

        with Image.open(image_path) as im:
            img = im.convert("RGB")

            width, height = img.size

            cx = int(bad.xmin)
            cy = int(bad.ymin)

            # Hata koordinati etrafinda yakin gorunum
            left = max(0, cx - 160)
            top = max(0, cy - 120)
            right = min(width, cx + 160)
            bottom = min(height, cy + 120)

            crop = np.asarray(
                img.crop((left, top, right, bottom))
            )

        records = df[
            (df["split"] == split) &
            (df["stem"] == stem)
        ]

        for j, stage in enumerate(
            ["ONCE", "SONRA"]
        ):
            ax = axes[i, j]
            ax.imshow(crop)

            for _, obj in records.iterrows():

                if not obj["is_valid"]:
                    if stage == "ONCE":
                        ax.scatter(
                            obj["xmin"] - left,
                            obj["ymin"] - top,
                            color="red",
                            s=100,
                            marker="x",
                            linewidths=2.5
                        )
                    continue

                x = obj["xmin"] - left
                y = obj["ymin"] - top

                rect = Rectangle(
                    (x, y),
                    obj["w"],
                    obj["h"],
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.5
                )

                ax.add_patch(rect)

            
            # ONCE ve SONRA panellerinde ayni goruntu sinirlarini ve olcegi koru.
            # Crop disindaki kutular eksenleri buyutmesin.
            ax.set_xlim(-0.5, crop.shape[1] - 0.5)
            ax.set_ylim(crop.shape[0] - 0.5, -0.5)
            ax.set_aspect("equal", adjustable="box")

            ax.set_title(
                f"{stem} - {stage}"
            )
            ax.axis("off")

    fig.suptitle(
        "Anotasyon Temizleme: Once ve Sonra\n"
        "Kirmizi X: Gecersiz anotasyon | "
        "Yesil: Gecerli kutu",
        fontsize=13
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    fig_path = (
        FIG / "fig31_annotations_before_after.png"
    )

    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\nGorsel kaydedildi: {fig_path}")

print(f"Temiz veri: {clean_path}")
print(f"Cikarilan kayitlar: {removed_path}")
print(f"Ozet: {summary_path}")
print("\nADIM 15A TAMAMLANDI.")
