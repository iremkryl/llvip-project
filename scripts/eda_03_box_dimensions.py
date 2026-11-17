"""
EDA 03 - BOUNDING BOX BOYUT VE ALAN DAGILIMI

AMAC:
LLVIP veri setindeki insan bounding box kutularinin
genislik, yukseklik ve goruntuye gore alan dagilimlarini
incelemek. Train ve test kumelerini karsilastirmak.

YONTEM:
1. boxes_table.csv dosyasindan anotasyonlar okunur.
2. Gecersiz veya sinir disi kutular analizden ayrilir.
3. Kutu genislik ve yukseklik dagilimlari hesaplanir.
4. Kutu alanlari goruntu alanina oranlanir.
5. Train/test icin karsilastirmali histogramlar uretilir.
6. Kucuk kutular ve ozet istatistikler raporlanir.

Girdi: outputs/logs/boxes_table.csv
Cikti: outputs/figures/fig22_box_dimensions.png
Terminalde kutu sayilari ve boyut istatistikleri.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG = Path("outputs/logs")
FIG = Path("outputs/figures")
FIG.mkdir(parents=True, exist_ok=True)

# 1. Kutu tablosunu oku
df = pd.read_csv(LOG / "boxes_table.csv")

print("===== HAM ANOTASYONLAR =====")
print(f"Toplam kutu: {len(df)}")

# 2. Gecersiz kutulari analiz disinda tut
# Ham XML veya CSV dosyalari degistirilmez.

valid = (
    (df["w"] > 0) &
    (df["h"] > 0) &
    (df["xmin"] >= 0) &
    (df["ymin"] >= 0) &
    (df["xmax"] <= df["rgb_w"]) &
    (df["ymax"] <= df["rgb_h"])
)

clean = df.loc[valid].copy()

print(f"Gecerli kutu: {len(clean)}")
print(f"Gecersiz kutu: {(~valid).sum()}")

# 3. Kutu alanini normalize et
clean["area_pct"] = (
    clean["area"] /
    (clean["rgb_w"] * clean["rgb_h"])
) * 100

# 4. Ozet istatistikleri yazdir
for split in ["train", "test"]:
    subset = clean[clean["split"] == split]

    print(f"\n===== {split.upper()} =====")
    print(f"Kutu sayisi: {len(subset)}")

    for column, label in [
        ("w", "Genislik (px)"),
        ("h", "Yukseklik (px)"),
        ("area_pct", "Alan (%)")
    ]:
        values = subset[column]

        print(f"\n{label}")
        print(f"Ortalama: {values.mean():.2f}")
        print(f"Medyan: {values.median():.2f}")
        print(f"Minimum: {values.min():.2f}")
        print(f"Maksimum: {values.max():.2f}")
        print(f"P5: {values.quantile(0.05):.2f}")
        print(f"P95: {values.quantile(0.95):.2f}")

# 5. Kucuk kutu kontrolu
# 1280x1024 -> 640x512 olceklenmesi varsayilir.
# 16 px yalnizca ornek bir inceleme esigidir.

small = (
    (clean["w"] / 2 < 16) |
    (clean["h"] / 2 < 16)
)

print("\n===== KUCUK KUTU KONTROLU =====")
print(f"640x512'de bir boyutu 16 px altinda: {small.sum()}")
print(f"Oran: {small.mean() * 100:.3f}%")

# 6. Karsilastirmali grafik
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

features = [
    ("w", "Bounding Box Width", "Pixels"),
    ("h", "Bounding Box Height", "Pixels"),
    ("area_pct", "Relative Bounding Box Area", "Area (%)")
]

for ax, (column, title, xlabel) in zip(axes, features):

    for split, color in [
        ("train", "royalblue"),
        ("test", "darkorange")
    ]:
        values = clean.loc[
            clean["split"] == split, column
        ]

        weights = np.ones(len(values)) * 100 / len(values)

        ax.hist(
            values,
            bins=40,
            weights=weights,
            alpha=0.55,
            color=color,
            label=split.upper()
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Percentage of Boxes (% per bin)")
    ax.legend()
    ax.grid(alpha=0.2)

plt.tight_layout()

path = FIG / "fig22_box_dimensions_en.png"
plt.savefig(path, dpi=160)
plt.close()

print(f"\nGrafik kaydedildi: {path}")