"""
ADIM 15B - GERCEK GORUNTU BOYUTLANDIRMA ANALIZI

AMAC:
LLVIP RGB ve IR goruntulerini 1280x1024 boyutundan
640x512 boyutuna kucultmenin goruntuler ve insan
kutulari uzerindeki etkisini incelemek.

YONTEM:
1. Temizlenmis anotasyonlar okunur.
2. Tum gecerli kutularin yeni boyutlari hesaplanir.
3. Train/test icin dusuk isik ve kucuk hedef
   iceren ikiser ornek secilir.
4. RGB ve IR goruntuleri LANCZOS ile gercekten
   yeniden boyutlandirilir.
5. Ornekler data/processed altinda PNG kaydedilir.
6. Once/sonra ve kucuk hedef yakinlastirma
   gorselleri olusturulur.
7. Kucultme ve geri buyutme sonrasi piksel
   farklari ornekler uzerinde olculur.

GIRDI:
data/processed/boxes_cleaned.csv
outputs/logs/images_table.csv
outputs/logs/quality_pass.csv
data/raw/LLVIP_extracted/

CIKTI:
data/processed/boxes_scaled_640x512.csv
data/processed/resize_demo/
outputs/logs/step15b_resize_samples.csv
outputs/figures/fig32_resize_before_after.png
outputs/figures/fig33_small_target_zoom.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib.patches import Rectangle

ROOT = Path("data/raw/LLVIP_extracted")
PROCESSED = Path("data/processed")
LOG = Path("outputs/logs")
FIG = Path("outputs/figures")

FIG.mkdir(parents=True, exist_ok=True)

ORIGINAL_SIZE = (1280, 1024)
TARGET_SIZE = (640, 512)

SX = TARGET_SIZE[0] / ORIGINAL_SIZE[0]
SY = TARGET_SIZE[1] / ORIGINAL_SIZE[1]


# 1. Tablolari oku
boxes = pd.read_csv(
    PROCESSED / "boxes_cleaned.csv",
    dtype={"stem": str}
)

images = pd.read_csv(
    LOG / "images_table.csv",
    dtype={"stem": str}
)

quality = pd.read_csv(
    LOG / "quality_pass.csv",
    dtype={"stem": str}
)

for df in [boxes, images, quality]:
    df["stem"] = df["stem"].str.zfill(6)

images = images.merge(
    quality[["stem", "split", "rgb_mean"]],
    on=["stem", "split"],
    how="inner",
    validate="one_to_one"
)


# 2. Tum gecerli kutularin olceklenmesi
scaled = boxes.copy()

for column in ["xmin", "xmax", "w"]:
    scaled[f"scaled_{column}"] = (
        scaled[column] * SX
    )

for column in ["ymin", "ymax", "h"]:
    scaled[f"scaled_{column}"] = (
        scaled[column] * SY
    )

scaled["scaled_area"] = (
    scaled["scaled_w"] * scaled["scaled_h"]
)

scaled["target_width"] = TARGET_SIZE[0]
scaled["target_height"] = TARGET_SIZE[1]

scaled_path = (
    PROCESSED / "boxes_scaled_640x512.csv"
)

scaled.to_csv(scaled_path, index=False)

print("===== KUTU OLCEKLEME =====")
print(f"Gecerli kutu: {len(scaled)}")

for threshold in [8, 16, 32]:

    small = (
        (scaled["scaled_w"] < threshold) |
        (scaled["scaled_h"] < threshold)
    )

    print(
        f"Bir boyutu {threshold} px altinda: "
        f"{small.sum()} "
        f"(%{small.mean() * 100:.3f})"
    )


# 3. Train/test icin ornekleri sec
selected = []

for split in ["train", "test"]:

    available = images[
        (images["split"] == split) &
        (images["stem"].isin(
            boxes.loc[
                boxes["split"] == split, "stem"
            ]
        ))
    ].copy()

    # RGB parlakligi %10 dilimine yakin ornek
    target = available["rgb_mean"].quantile(0.10)

    dark = available.loc[
        (available["rgb_mean"] - target)
        .abs().idxmin()
    ]

    # En kucuk alanli kutuyu iceren farkli ornek:
    candidates = boxes[
        (boxes["split"] == split) &
        (boxes["stem"] != dark["stem"])
    ].copy()

    candidates["cx"] = (
        candidates["xmin"] + candidates["xmax"]
    ) / 2

    candidates["cy"] = (
        candidates["ymin"] + candidates["ymax"]
    ) / 2

    candidates = candidates[
        (candidates["cx"].between(128, 1152)) &
        (candidates["cy"].between(128, 896)) &
        (candidates["xmin"] > 1) &
        (candidates["ymin"] > 1) &
        (candidates["xmax"] < candidates["rgb_w"]) &
        (candidates["ymax"] < candidates["rgb_h"]) &
        (candidates["w"] >= 20) &
        (candidates["h"] >= 40)
    ].copy()

    if candidates.empty:
        raise ValueError(
            f"{split} icin uygun kucuk hedef bulunamadi."
        )

    candidates["box_area"] = (
        candidates["w"] * candidates["h"]
    )

    tiny = candidates.sort_values(
        ["box_area", "stem"]
    ).iloc[0]


    selected.extend([
        (split, dark["stem"], "Dusuk isik"),
        (split, tiny["stem"], "Kucuk hedef")
    ])


# 4. Gercek goruntuleri yeniden boyutlandir
examples = []
sample_rows = []


def reconstruction_mae(original, resized):
    """
    Kucultulen goruntuyu tekrar orijinal boyuta
    getirerek ortalama mutlak piksel farkini hesapla.
    """
    restored = resized.resize(
        original.size,
        Image.Resampling.BICUBIC
    )

    a = np.asarray(original).astype(np.float32)
    b = np.asarray(restored).astype(np.float32)

    return float(np.abs(a - b).mean())


for split, stem, category in selected:

    rgb_path = (
        ROOT / "visible" / split / f"{stem}.jpg"
    )

    ir_path = (
        ROOT / "infrared" / split / f"{stem}.jpg"
    )

    with Image.open(rgb_path) as im:
        rgb_original = im.convert("RGB")

    with Image.open(ir_path) as im:
        ir_original = im.convert("L")

    if (
        rgb_original.size != ORIGINAL_SIZE or
        ir_original.size != ORIGINAL_SIZE
    ):
        raise ValueError(
            f"Beklenmeyen goruntu boyutu: {stem}"
        )

    rgb_small = rgb_original.resize(
        TARGET_SIZE,
        Image.Resampling.LANCZOS
    )

    ir_small = ir_original.resize(
        TARGET_SIZE,
        Image.Resampling.LANCZOS
    )

    # Ham veriye dokunmadan ornekleri kaydet
    base = PROCESSED / "resize_demo" / split

    (base / "rgb").mkdir(
        parents=True, exist_ok=True
    )

    (base / "ir").mkdir(
        parents=True, exist_ok=True
    )

    rgb_small.save(
        base / "rgb" / f"{stem}.png"
    )

    ir_small.save(
        base / "ir" / f"{stem}.png"
    )

    ann = boxes[
        (boxes["split"] == split) &
        (boxes["stem"] == stem)
    ].copy()

    ann["box_area"] = ann["w"] * ann["h"]

    # Kucuk hedefte en kucuk kutuyu,
    # digerinde en buyuk kutuyu isaretle.
    if category == "Kucuk hedef":
        target_box = ann.sort_values(
            "box_area"
        ).iloc[0]
    else:
        target_box = ann.sort_values(
            "box_area"
        ).iloc[-1]

    rgb_mae = reconstruction_mae(
        rgb_original, rgb_small
    )

    ir_mae = reconstruction_mae(
        ir_original, ir_small
    )

    examples.append({
        "stem": stem,
        "split": split,
        "category": category,
        "rgb_original": np.asarray(rgb_original),
        "ir_original": np.asarray(ir_original),
        "rgb_small": np.asarray(rgb_small),
        "ir_small": np.asarray(ir_small),
        "box": target_box
    })

    sample_rows.append({
        "stem": stem,
        "split": split,
        "category": category,
        "box_w_original": target_box["w"],
        "box_h_original": target_box["h"],
        "box_w_scaled": target_box["w"] * SX,
        "box_h_scaled": target_box["h"] * SY,
        "rgb_reconstruction_mae": rgb_mae,
        "ir_reconstruction_mae": ir_mae
    })

    print(
        f"{split.upper()} / {category}: {stem} "
        f"| RGB MAE={rgb_mae:.2f} "
        f"| IR MAE={ir_mae:.2f}"
    )


sample_df = pd.DataFrame(sample_rows)

sample_path = (
    LOG / "step15b_resize_samples.csv"
)

sample_df.to_csv(sample_path, index=False)


# 5. Kutu cizme fonksiyonu
def draw_box(ax, box, scale=1.0, origin=(0, 0)):

    x = box["xmin"] * scale - origin[0]
    y = box["ymin"] * scale - origin[1]

    w = box["w"] * scale
    h = box["h"] * scale

    ax.add_patch(
        Rectangle(
            (x, y), w, h,
            fill=False,
            edgecolor="lime",
            linewidth=2
        )
    )


# 6. Once-sonra tam goruntu karsilastirmasi
fig, axes = plt.subplots(
    4, 4, figsize=(17, 15)
)

column_titles = [
    "RGB - 1280x1024",
    "RGB - 640x512",
    "IR - 1280x1024",
    "IR - 640x512"
]

for i, ex in enumerate(examples):

    pictures = [
        ex["rgb_original"],
        ex["rgb_small"],
        ex["ir_original"],
        ex["ir_small"]
    ]

    for j, picture in enumerate(pictures):

        ax = axes[i, j]

        if picture.ndim == 2:
            ax.imshow(
                picture, cmap="gray",
                vmin=0, vmax=255,
                interpolation="nearest"
            )
        else:
            ax.imshow(
                picture,
                interpolation="nearest"
            )

        scale = 1.0 if j % 2 == 0 else 0.5

        draw_box(
            ax, ex["box"], scale=scale
        )

        height, width = picture.shape[:2]

        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")

        if i == 0:
            ax.set_title(
                column_titles[j], fontsize=11
            )

    axes[i, 0].set_ylabel(
        f"{ex['split'].upper()}\n"
        f"{ex['category']}\n"
        f"{ex['stem']}",
        fontsize=10
    )

fig.suptitle(
    "Gercek Boyutlandirma: Once ve Sonra",
    fontsize=15
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

fig32 = FIG / "fig32_resize_before_after.png"

plt.savefig(fig32, dpi=160)
plt.close()


# 7. Kucuk hedeflerin yakinlastirilmis karsilastirmasi
tiny_examples = [
    ex for ex in examples
    if ex["category"] == "Kucuk hedef"
]

fig, axes = plt.subplots(
    len(tiny_examples), 4,
    figsize=(15, 7),
    squeeze=False
)

for i, ex in enumerate(tiny_examples):

    box = ex["box"]

    cx = (box["xmin"] + box["xmax"]) / 2
    cy = (box["ymin"] + box["ymax"]) / 2

    # 256x256 orijinal bolge
    # Esdegeri 128x128 kucultulmus bolgedir.
    x0 = int(
        np.clip(
            2 * round((cx - 128) / 2),
            0, ORIGINAL_SIZE[0] - 256
        )
    )

    y0 = int(
        np.clip(
            2 * round((cy - 128) / 2),
            0, ORIGINAL_SIZE[1] - 256
        )
    )

    regions = [
        (
            ex["rgb_original"][
                y0:y0+256, x0:x0+256
            ],
            1.0, (x0, y0)
        ),
        (
            ex["rgb_small"][
                y0//2:(y0+256)//2,
                x0//2:(x0+256)//2
            ],
            0.5, (x0/2, y0/2)
        ),
        (
            ex["ir_original"][
                y0:y0+256, x0:x0+256
            ],
            1.0, (x0, y0)
        ),
        (
            ex["ir_small"][
                y0//2:(y0+256)//2,
                x0//2:(x0+256)//2
            ],
            0.5, (x0/2, y0/2)
        )
    ]

    for j, (picture, scale, origin) in enumerate(regions):

        ax = axes[i, j]

        if picture.ndim == 2:
            ax.imshow(
                picture, cmap="gray",
                vmin=0, vmax=255,
                interpolation="nearest"
            )
        else:
            ax.imshow(
                picture,
                interpolation="nearest"
            )

        draw_box(
            ax, box,
            scale=scale,
            origin=origin
        )

        height, width = picture.shape[:2]

        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")

        if i == 0:
            ax.set_title(
                column_titles[j], fontsize=10
            )

    axes[i, 0].set_ylabel(
        f"{ex['split'].upper()}\n{ex['stem']}",
        fontsize=10
    )

fig.suptitle(
    "Kucuk Hedeflerde Boyutlandirma Etkisi",
    fontsize=14
)

plt.tight_layout(rect=[0, 0, 1, 0.94])

fig33 = FIG / "fig33_small_target_zoom.png"

plt.savefig(fig33, dpi=160)
plt.close()


print("\n===== ADIM 15B TAMAMLANDI =====")
print(f"Olceklenmis kutular: {scaled_path}")
print(f"Orneklem CSV: {sample_path}")
print(f"Once/sonra gorseli: {fig32}")
print(f"Kucuk hedef gorseli: {fig33}")
print(f"Gercekten boyutlandirilan cift: {len(examples)}")
