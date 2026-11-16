
"""
EDA 06 - RGB/IR ORNEKLERI VE PIKSEL DAGILIMI

AMAC:
LLVIP veri setindeki eslestirilmis RGB ve IR goruntulerini
farkli parlaklik seviyelerinde karsilastirmak.
RGB renk kanallari ve IR yogunluk dagilimlarini incelemek.

YONTEM:
1. quality_pass.csv dosyasindan goruntu kayitlari okunur.
2. Train/test icin dusuk, orta ve yuksek RGB parlakligina
   sahip temsili eslestirilmis goruntuler secilir.
3. RGB ve IR goruntuleri yan yana gorsellestirilir.
4. Her kumeden seed=42 ile 50 rastgele goruntu cifti secilir.
5. RGB kanallari ve IR piksel histogramlari hesaplanir.
6. Iki ayri analiz gorseli kaydedilir.

Girdi: outputs/logs/quality_pass.csv
       data/raw/LLVIP_extracted/visible/
       data/raw/LLVIP_extracted/infrared/
Cikti: outputs/figures/fig25_paired_rgb_ir_samples.png
       outputs/figures/fig26_pixel_distributions.png
Terminal: Orneklem ve gorsel bilgileri.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("data/raw/LLVIP_extracted")
LOG = Path("outputs/logs")
FIG = Path("outputs/figures")
FIG.mkdir(parents=True, exist_ok=True)

SEED = 42
SAMPLE_PER_SPLIT = 50

# 1. Kayitlari oku
df = pd.read_csv(
    LOG / "quality_pass.csv",
    dtype={"stem": str}
)

df["stem"] = df["stem"].str.zfill(6)

df = df[
    df["rgb_ok"].astype(str).str.lower().isin(["true", "1"])
    & df["ir_ok"].astype(str).str.lower().isin(["true", "1"])
].copy()

df["rgb_mean"] = pd.to_numeric(df["rgb_mean"])

print(f"Toplam gecerli cift: {len(df)}")


def image_paths(row):
    split = row["split"]
    stem = row["stem"]

    rgb = ROOT / "visible" / split / f"{stem}.jpg"
    ir = ROOT / "infrared" / split / f"{stem}.jpg"

    return rgb, ir


# 2. Farkli parlakliklardan ornek sec
selected = []

for split in ["train", "test"]:
    subset = df[df["split"] == split]
    used = set()

    for q, label in [
        (0.10, "Dusuk"),
        (0.50, "Orta"),
        (0.90, "Yuksek")
    ]:
        target = subset["rgb_mean"].quantile(q)

        candidates = subset.loc[
            (subset["rgb_mean"] - target)
            .abs()
            .sort_values(kind="stable")
            .index
        ]

        row = next(
            r for _, r in candidates.iterrows()
            if r["stem"] not in used
        )

        used.add(row["stem"])
        selected.append((row, split, label))

# 3. Eslesmis RGB ve IR ornekleri
fig, axes = plt.subplots(
    2, 6, figsize=(20, 7)
)

for j, (row, split, label) in enumerate(selected):

    rgb_path, ir_path = image_paths(row)

    with Image.open(rgb_path) as im:
        rgb = im.convert("RGB")
        rgb.thumbnail((640, 512))

    with Image.open(ir_path) as im:
        ir = im.convert("L")
        ir.thumbnail((640, 512))

    axes[0, j].imshow(rgb)
    axes[1, j].imshow(ir, cmap="gray", vmin=0, vmax=255)

    axes[0, j].set_title(
        f"{split.upper()} / {label}\n{row['stem']}",
        fontsize=10
    )

    for i in range(2):
        axes[i, j].axis("off")

axes[0, 0].set_ylabel("RGB", fontsize=12)
axes[1, 0].set_ylabel("IR", fontsize=12)

plt.tight_layout()

path1 = FIG / "fig25_paired_rgb_ir_samples.png"
plt.savefig(path1, dpi=150)
plt.close()

print(f"Eslesmis ornekler: {path1}")


# 4. Piksel histogramlari
bins = np.linspace(0, 256, 65)
centers = (bins[:-1] + bins[1:]) / 2

fig, axes = plt.subplots(
    1, 2, figsize=(14, 5)
)

for ax, split in zip(axes, ["train", "test"]):

    subset = df[df["split"] == split]

    sample = subset.sample(
        n=min(SAMPLE_PER_SPLIT, len(subset)),
        random_state=SEED
    )

    rgb_hist = np.zeros((3, 64), dtype=np.float64)
    ir_hist = np.zeros(64, dtype=np.float64)

    for _, row in sample.iterrows():

        rgb_path, ir_path = image_paths(row)

        with Image.open(rgb_path) as im:
            rgb = np.asarray(
                im.convert("RGB").resize((320, 256))
            )

        with Image.open(ir_path) as im:
            ir = np.asarray(
                im.convert("L").resize((320, 256))
            )

        for channel in range(3):
            rgb_hist[channel] += np.histogram(
                rgb[:, :, channel],
                bins=bins
            )[0]

        ir_hist += np.histogram(
            ir, bins=bins
        )[0]

    for channel, color, label in [
        (0, "red", "RGB - R"),
        (1, "green", "RGB - G"),
        (2, "blue", "RGB - B")
    ]:
        values = rgb_hist[channel]
        values = values / values.sum() * 100

        ax.plot(
            centers, values,
            color=color, label=label,
            linewidth=1.5
        )

    ir_values = ir_hist / ir_hist.sum() * 100

    ax.plot(
        centers, ir_values,
        color="black",
        label="IR",
        linewidth=2,
        linestyle="--"
    )

    ax.set_title(
        f"{split.upper()} - {len(sample)} Cift"
    )
    ax.set_xlabel("Piksel yogunlugu (0-255)")
    ax.set_ylabel("Piksellerin orani (% / aralik)")
    ax.set_xlim(0, 255)
    ax.legend()
    ax.grid(alpha=0.2)

    print(
        f"{split.upper()}: "
        f"{len(sample)} cift analiz edildi."
    )

plt.tight_layout()

path2 = FIG / "fig26_pixel_distributions.png"
plt.savefig(path2, dpi=150)
plt.close()

print(f"Piksel histogramlari: {path2}")
print("\nEDA 06 tamamlandi.")