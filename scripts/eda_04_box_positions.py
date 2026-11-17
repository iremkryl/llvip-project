"""
EDA 04 - BOUNDING BOX KONUM DAGILIMI

AMAC:
LLVIP veri setindeki insan kutularinin goruntu icindeki
konum dagilimini incelemek ve train/test kumelerini
karsilastirmak.

YONTEM:
1. boxes_table.csv dosyasini oku.
2. Gecersiz kutulari analiz disinda birak.
3. Her kutu icin normalize merkez koordinatlarini hesapla.
4. Train ve test icin 2B konum yogunluk haritalari olustur.
5. Kenara degen kutularin oranini hesapla.

Girdi: outputs/logs/boxes_table.csv
Cikti: outputs/figures/fig23_box_positions.png
Terminalde konum ve kenar temas istatistikleri.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG = Path("outputs/logs")
FIG = Path("outputs/figures")
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(LOG / "boxes_table.csv")

# Gecerli kutular
valid = (
    (df["w"] > 0) &
    (df["h"] > 0) &
    (df["xmin"] >= 0) &
    (df["ymin"] >= 0) &
    (df["xmax"] <= df["rgb_w"]) &
    (df["ymax"] <= df["rgb_h"])
)

clean = df.loc[valid].copy()

# Normalize merkez koordinatlari
clean["cx_norm"] = ((clean["xmin"] + clean["xmax"]) / 2) / clean["rgb_w"]
clean["cy_norm"] = ((clean["ymin"] + clean["ymax"]) / 2) / clean["rgb_h"]

# Kenara temas
clean["touches_edge"] = (
    (clean["xmin"] <= 0) |
    (clean["ymin"] <= 0) |
    (clean["xmax"] >= clean["rgb_w"]) |
    (clean["ymax"] >= clean["rgb_h"])
)

for split in ["train", "test"]:
    subset = clean[clean["split"] == split]

    print(f"\n===== {split.upper()} =====")
    print(f"Kutu sayisi: {len(subset)}")
    print(f"Merkez x ortalama: {subset['cx_norm'].mean():.3f}")
    print(f"Merkez y ortalama: {subset['cy_norm'].mean():.3f}")
    print(f"Kenar temasi olan kutu: {subset['touches_edge'].sum()}")
    print(f"Kenar temasi orani: {subset['touches_edge'].mean() * 100:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, split in zip(axes, ["train", "test"]):
    subset = clean[clean["split"] == split]

    h = ax.hist2d(
        subset["cx_norm"],
        subset["cy_norm"],
        bins=40,
        range=[[0, 1], [0, 1]],
        cmap="viridis"
    )

    ax.set_title(f"{split.upper()} - Bounding Box Centers")
    ax.set_xlabel("Normalized X Center")
    ax.set_ylabel("Normalized Y Center")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # ust taraf yukarida gorunsun

    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label("Number of Boxes")

plt.tight_layout()

path = FIG / "fig23_box_positions_en.png"
plt.savefig(path, dpi=160)
plt.close()

print(f"\nGrafik kaydedildi: {path}")