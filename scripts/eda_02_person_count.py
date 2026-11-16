"""
EDA 02 - GORUNTU BASINA INSAN SAYISI DAGILIMI

AMAC:
LLVIP veri setindeki goruntulerde bulunan insan
etiketi sayilarini incelemek ve train/test
kumelerinin dagilimlarini karsilastirmak.

YONTEM:
1. images_table.csv dosyasindan kayitlar okunur.
2. Her goruntunun n_boxes degeri elde edilir.
3. Train ve test icin insan sayisi dagilimlari hesaplanir.
4. Farkli kume buyuklukleri nedeniyle yuzdelik oranlar
   kullanilarak karsilastirma grafigi olusturulur.
5. Ortalama, medyan, ceyreklik ve uc degerler raporlanir.

Girdi: outputs/logs/images_table.csv
Cikti: outputs/figures/fig21_person_count_by_split.png
Terminalde Train/test insan sayisi istatistikleri görünür
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG = Path("outputs/logs")
FIG = Path("outputs/figures")
FIG.mkdir(parents=True, exist_ok=True)

# 1. Goruntu tablosunu oku
df = pd.read_csv(LOG / "images_table.csv")

df["n_boxes"] = pd.to_numeric(
    df["n_boxes"], errors="raise"
)

# 2. Ozet istatistikleri hesapla
for split in ["train", "test"]:
    values = df.loc[
        df["split"] == split, "n_boxes"
    ]

    print(f"\n===== {split.upper()} =====")
    print(f"Goruntu sayisi: {len(values)}")
    print(f"Toplam kutu: {values.sum()}")
    print(f"Ortalama: {values.mean():.2f}")
    print(f"Medyan: {values.median():.2f}")
    print(f"Q1: {values.quantile(0.25):.2f}")
    print(f"Q3: {values.quantile(0.75):.2f}")
    print(f"Minimum: {values.min()}")
    print(f"Maksimum: {values.max()}")
    print(f"Kutusuz goruntu: {(values == 0).sum()}")

# 3. Dagilim grafigini olustur
max_count = int(df["n_boxes"].max())
x = np.arange(max_count + 1)

fig, ax = plt.subplots(figsize=(12, 6))

width = 0.38

for split, offset in [
    ("train", -width / 2),
    ("test", width / 2)
]:
    values = df.loc[
        df["split"] == split, "n_boxes"
    ]

    frequencies = (
        values.value_counts()
        .reindex(x, fill_value=0)
        .sort_index()
    )

    percentages = frequencies / len(values) * 100

    ax.bar(
        x + offset,
        percentages,
        width=width,
        label=f"{split.upper()} ({len(values)} goruntu)",
        alpha=0.8
    )

ax.set_title(
    "Goruntu Basina Insan Sayisi - Train/Test",
    fontsize=14
)

ax.set_xlabel("Goruntudeki insan etiketi sayisi")
ax.set_ylabel("Goruntulerin orani (%)")
ax.set_xticks(x)
ax.legend()
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

path = FIG / "fig21_person_count_by_split.png"
plt.savefig(path, dpi=160)
plt.close()

print(f"\nGrafik kaydedildi: {path}")
