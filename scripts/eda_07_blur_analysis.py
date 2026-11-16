"""
EDA 07 - RGB VE IR GORUNTU KESKINLIGI ANALIZI

AMAC:
LLVIP veri setindeki RGB ve IR goruntulerinin
kenar ayrintisi ve keskinlik gostergelerini incelemek.
Train ve test kumelerindeki dagilimlari karsilastirmak.

YONTEM:
1. Her kumeden seed=42 ile 100 goruntu cifti secilir.
2. RGB ve IR goruntuleri gri tonlamali okunur.
3. Goruntuler ortak 640x512 boyutuna getirilir.
4. Laplacian varyansi hesaplanir.
5. Her modalite icin train/test dagilimi karsilastirilir.
6. Sayisal sonuclar ve grafik kaydedilir.

GIRDI:
outputs/logs/images_table.csv
data/raw/LLVIP_extracted/visible/
data/raw/LLVIP_extracted/infrared/

CIKTI:
outputs/figures/fig27_blur_analysis.png
outputs/logs/eda07_blur_sample.csv
Terminal: Laplacian varyans istatistikleri.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("data/raw/LLVIP_extracted")
LOG = Path("outputs/logs")
FIG = Path("outputs/figures")

FIG.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

SEED = 42
SAMPLE_PER_SPLIT = 100
SIZE = (640, 512)


def laplacian_variance(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Goruntu okunamadi: {path}")

    img = cv2.resize(
        img, SIZE, interpolation=cv2.INTER_AREA
    )

    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)

    return float(lap.var())


# 1. Goruntu kayitlarini oku
df = pd.read_csv(
    LOG / "images_table.csv",
    dtype={"stem": str}
)

df["stem"] = df["stem"].str.zfill(6)

results = []

# 2. Train ve test icin rastgele ornekleme
for split in ["train", "test"]:

    subset = df[df["split"] == split]

    sample = subset.sample(
        n=min(SAMPLE_PER_SPLIT, len(subset)),
        random_state=SEED
    )

    print(f"\n===== {split.upper()} =====")
    print(f"Secilen cift: {len(sample)}")

    for _, row in sample.iterrows():

        stem = row["stem"]

        rgb_path = (
            ROOT / "visible" / split / f"{stem}.jpg"
        )

        ir_path = (
            ROOT / "infrared" / split / f"{stem}.jpg"
        )

        rgb_var = laplacian_variance(rgb_path)
        ir_var = laplacian_variance(ir_path)

        results.append({
            "stem": stem,
            "split": split,
            "rgb_laplacian_var": rgb_var,
            "ir_laplacian_var": ir_var
        })


# 3. Sonuclari kaydet
res = pd.DataFrame(results)

csv_path = LOG / "eda07_blur_sample.csv"
res.to_csv(csv_path, index=False)

# 4. Ozet istatistikler
for split in ["train", "test"]:

    subset = res[res["split"] == split]

    for col, name in [
        ("rgb_laplacian_var", "RGB"),
        ("ir_laplacian_var", "IR")
    ]:
        values = subset[col]

        print(f"\n{split.upper()} - {name}")
        print(f"Ortalama: {values.mean():.2f}")
        print(f"Medyan: {values.median():.2f}")
        print(f"P10: {values.quantile(0.10):.2f}")
        print(f"P90: {values.quantile(0.90):.2f}")


# 5. Dagilim gorsellestirme
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, title in [
    (axes[0], "rgb_laplacian_var", "RGB"),
    (axes[1], "ir_laplacian_var", "IR")
]:

    for split, color in [
        ("train", "royalblue"),
        ("test", "darkorange")
    ]:

        values = res.loc[
            res["split"] == split, col
        ]

        # Carpik dagilim icin log donusumu
        log_values = np.log10(values + 1)

        ax.hist(
            log_values,
            bins=25,
            density=True,
            alpha=0.55,
            color=color,
            label=split.upper()
        )

    ax.set_title(f"{title} - Laplacian Varyansi")
    ax.set_xlabel("log10(1 + Laplacian varyansi)")
    ax.set_ylabel("Yogunluk")
    ax.legend()
    ax.grid(alpha=0.2)

plt.tight_layout()

fig_path = FIG / "fig27_blur_analysis.png"
plt.savefig(fig_path, dpi=160)
plt.close()

print(f"\nGrafik: {fig_path}")
print(f"CSV: {csv_path}")
print("\nEDA 07 tamamlandi.")
