"""
EDA 01 - RGB VE IR PARLAKLIK / YOGUNLUK DAGILIMI

AMAC:
LLVIP veri setindeki 15.488 eslestirilmis RGB-IR goruntu ciftinin
ortalama piksel yogunluklarini iki modalite icin ayri ayri
incelemek ve dagilim ozelliklerini karsilastirmak.

YONTEM:
1. quality_pass.csv dosyasindan goruntu kalitesi kayitlari okunur.
2. Basariyla cozulmus RGB ve IR goruntuleri secilir.
3. RGB ve IR ortalama piksel yogunluklarinin histogramlari cizilir.
4. Ortalama, medyan, minimum ve maksimum degerleri hesaplanir.
5. Grafik, her iki modaliteyi yan yana gosterecek sekilde kaydedilir.

Girdi: outputs/logs/quality_pass.csv
Cikti: outputs/figures/fig20_rgb_ir_brightness.png
Terminalde her iki modaliteye ait ozet istatistikler.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

LOG = Path("outputs/logs")
FIG = Path("outputs/figures")
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(LOG / "quality_pass.csv")

# Basariyla okunan goruntuler
df = df[(df["rgb_ok"] == True) & (df["ir_ok"] == True)]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, column, title, color in [
    (axes[0], "rgb_mean", "RGB Mean Brightness", "royalblue"),
    (axes[1], "ir_mean", "IR Mean Intensity", "darkorange")
]:
    values = df[column].dropna()

    ax.hist(values, bins=50, color=color,
            alpha=0.75, edgecolor="white")

    median = values.median()
    ax.axvline(median, color="black",
               linestyle="--",
               label=f"Median: {median:.2f}")

    ax.set_title(title)
    ax.set_xlabel("Mean Pixel Intensity (0-255)")
    ax.set_ylabel("Number of Images")
    ax.set_xlim(0, 255)
    ax.legend()
    ax.grid(alpha=0.2)

    print(f"\n{title}")
    print(f"Goruntu sayisi: {len(values)}")
    print(f"Ortalama: {values.mean():.2f}")
    print(f"Medyan: {median:.2f}")
    print(f"Minimum: {values.min():.2f}")
    print(f"Maksimum: {values.max():.2f}")

plt.tight_layout()

path = FIG / "fig20_rgb_ir_brightness_en.png"
plt.savefig(path, dpi=160)
plt.close()

print(f"\nGrafik kaydedildi: {path}")
