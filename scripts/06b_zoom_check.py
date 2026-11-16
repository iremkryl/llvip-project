"""06b_zoom_check.py - Aday eksik etiketler icin yakin kirpma (zoom) kontrolu.

Not: RGB kirpmalarina yalnizca GOSTERIM icin gama (0.5) uygulanir; veri degismez.
Bolge koordinatlari goruntuden tahmin edildi.
"""
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("data/raw/LLVIP_extracted")
LOG = Path("outputs/logs")
FIG = Path("outputs/figures")
GAMMA = 0.5

# (stem, split, (x0, y0, x1, y1)) - tam cozunurluk (1280x1024) koordinatlari
CASES = [
    ("100030", "train", (150, 330, 650, 750)),
    ("100033", "train", (150, 330, 650, 750)),
    ("160363", "train", (400, 650, 950, 1024)),
    ("160365", "train", (400, 650, 950, 1024)),
]

boxes = pd.read_csv(LOG / "boxes_table.csv", dtype={"stem": str})


def draw(ax, folder, split, stem, region, title):
    x0, y0, x1, y1 = region
    with Image.open(ROOT / folder / split / f"{stem}.jpg") as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    crop = arr[y0:y1, x0:x1]
    if folder == "visible":
        crop = np.power(crop, GAMMA)  # yalnizca gosterim
    ax.imshow(crop, extent=(x0, x1, y1, y0))
    b = boxes[boxes["stem"] == stem]
    for _, r in b.iterrows():
        ax.add_patch(patches.Rectangle((r.xmin, r.ymin), r.xmax - r.xmin, r.ymax - r.ymin,
                                       fill=False, edgecolor="red", linewidth=1.5))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_title(title, fontsize=9)
    return b


fig, axes = plt.subplots(2, len(CASES), figsize=(5 * len(CASES), 8), squeeze=False)
for j, (stem, split, region) in enumerate(CASES):
    for i, folder in enumerate(["visible", "infrared"]):
        b = draw(axes[i, j], folder, split, stem, region,
                 f"{'RGB (gama 0.5)' if i == 0 else 'IR'} {stem}")
    x0, y0, x1, y1 = region
    inside = b[(b.xmax > x0) & (b.xmin < x1) & (b.ymax > y0) & (b.ymin < y1)]
    print(f"{stem}: toplam kutu={len(b)}, bu kirpma bolgesiyle kesisen kutu={len(inside)}")

plt.tight_layout()
FIG.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG / "fig05_zoom_candidates.png", dpi=150)
print("Kaydedildi: outputs/figures/fig05_zoom_candidates.png")