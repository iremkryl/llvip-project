"""
EDA 05 - RGB VE IR YOGUNLUK ILISKISI

AMAC:
RGB ortalama parlakligi ile IR ortalama yogunlugu
arasindaki iliskiyi incelemek ve train/test
kumeleri icin karsilastirmak.

YONTEM:
1. quality_pass.csv dosyasini oku.
2. Basariyla cozulmus RGB ve IR goruntulerini sec.
3. RGB ortalama parlaklik ile IR ortalama yogunluk
   arasindaki Pearson ve Spearman korelasyonlarini hesapla.
4. Train ve test icin ayri hexbin dagilim grafigi ciz.
5. Genel dagilim ve iliski yapisini raporla.

Girdi: outputs/logs/quality_pass.csv
Cikti: outputs/figures/fig24_rgb_ir_relationship.png
Terminal: Korelasyon ve ozet istatistikler.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

LOG = Path("outputs/logs")
FIG = Path("outputs/figures")
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(LOG / "quality_pass.csv")

# Basariyla cozulmus goruntuler
df = df[(df["rgb_ok"] == True) & (df["ir_ok"] == True)].copy()

# Korelasyonlar
pearson_all = df["rgb_mean"].corr(df["ir_mean"], method="pearson")
spearman_all = df["rgb_mean"].corr(df["ir_mean"], method="spearman")

print("===== TUM VERI =====")
print(f"Goruntu sayisi: {len(df)}")
print(f"RGB mean ortalama: {df['rgb_mean'].mean():.2f}")
print(f"IR mean ortalama: {df['ir_mean'].mean():.2f}")
print(f"Pearson korelasyonu: {pearson_all:.4f}")
print(f"Spearman korelasyonu: {spearman_all:.4f}")

for split in ["train", "test"]:
    subset = df[df["split"] == split]

    pearson = subset["rgb_mean"].corr(subset["ir_mean"], method="pearson")
    spearman = subset["rgb_mean"].corr(subset["ir_mean"], method="spearman")

    print(f"\n===== {split.upper()} =====")
    print(f"Goruntu sayisi: {len(subset)}")
    print(f"RGB mean ortalama: {subset['rgb_mean'].mean():.2f}")
    print(f"IR mean ortalama: {subset['ir_mean'].mean():.2f}")
    print(f"Pearson korelasyonu: {pearson:.4f}")
    print(f"Spearman korelasyonu: {spearman:.4f}")

# Grafik
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

for ax, split in zip(axes, ["train", "test"]):
    subset = df[df["split"] == split]

    hb = ax.hexbin(
        subset["rgb_mean"],
        subset["ir_mean"],
        gridsize=35,
        mincnt=1,
        cmap="viridis"
    )

    pearson = subset["rgb_mean"].corr(subset["ir_mean"], method="pearson")

    ax.set_title(f"{split.upper()} - Pearson r = {pearson:.3f}")
    ax.set_xlabel("RGB ortalama parlaklik")
    ax.set_ylabel("IR ortalama yogunluk")
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 140)

    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("Goruntu sayisi")

plt.tight_layout()

path = FIG / "fig24_rgb_ir_relationship.png"
plt.savefig(path, dpi=160)
plt.close()

print(f"\nGrafik kaydedildi: {path}")