
"""
EDA 08 - RGB VE IR NESNE KONTRAST ANALIZI

AMAC:
LLVIP veri setindeki insan kutularinin yakin cevrelerine
gore yerel yogunluk farklarini RGB ve IR icin olcmek.

YONTEM:
1. Train ve testten seed=42 ile 100'er goruntu secilir.
2. RGB ve IR goruntuleri gri tonlamali okunur.
3. Her gecerli insan kutusunun icindeki medyan hesaplanir.
4. Kutunun disinda %25 genisletilmis cevre incelenir.
5. Diger insan kutulari cevre hesabindan cikarilir.
6. Simetrik normalize yogunluk farklari hesaplanir.
7. RGB ve IR olcumleri ayni kutular uzerinde karsilastirilir.

GIRDI:
outputs/logs/images_table.csv
outputs/logs/boxes_table.csv
data/raw/LLVIP_extracted/visible/
data/raw/LLVIP_extracted/infrared/

CIKTI:
outputs/logs/eda08_object_contrast.csv
outputs/figures/fig28_object_contrast.png
Terminal: RGB ve IR kontrast istatistikleri.
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

SEED = 42
SAMPLE_SIZE = 100

images = pd.read_csv(
    LOG / "images_table.csv", dtype={"stem": str}
)
boxes = pd.read_csv(
    LOG / "boxes_table.csv", dtype={"stem": str}
)

images["stem"] = images["stem"].str.zfill(6)
boxes["stem"] = boxes["stem"].str.zfill(6)

# Gecersiz kutulari analiz disinda tut.
boxes = boxes[
    (boxes["w"] > 0) &
    (boxes["h"] > 0) &
    (boxes["xmin"] >= 0) &
    (boxes["ymin"] >= 0) &
    (boxes["xmax"] <= boxes["rgb_w"]) &
    (boxes["ymax"] <= boxes["rgb_h"])
].copy()


def local_contrast(img, box, all_boxes, box_index):
    height, width = img.shape

    x1 = max(0, int(np.floor(box.xmin)))
    y1 = max(0, int(np.floor(box.ymin)))
    x2 = min(width, int(np.ceil(box.xmax)))
    y2 = min(height, int(np.ceil(box.ymax)))

    if x2 <= x1 or y2 <= y1:
        return None

    # Kutunun disina %25 cevre ekle.
    pad_x = max(8, int((x2 - x1) * 0.25))
    pad_y = max(8, int((y2 - y1) * 0.25))

    a = max(0, x1 - pad_x)
    b = max(0, y1 - pad_y)
    c = min(width, x2 + pad_x)
    d = min(height, y2 + pad_y)

    patch = img[b:d, a:c]

    # Yalnizca kutunun disindaki bolgeyi sec.
    background_mask = np.ones(
        patch.shape, dtype=bool
    )

    background_mask[
        y1-b:y2-b, x1-a:x2-a
    ] = False

    # Diger etiketli insan kutularini maskele.
    for j, other in enumerate(all_boxes):
        if j == box_index:
            continue

        ox1 = max(a, int(np.floor(other.xmin)))
        oy1 = max(b, int(np.floor(other.ymin)))
        ox2 = min(c, int(np.ceil(other.xmax)))
        oy2 = min(d, int(np.ceil(other.ymax)))

        if ox2 > ox1 and oy2 > oy1:
            background_mask[
                oy1-b:oy2-b, ox1-a:ox2-a
            ] = False

    inside = img[y1:y2, x1:x2]
    outside = patch[background_mask]

    if inside.size < 20 or outside.size < 20:
        return None

    inside_med = float(np.median(inside))
    outside_med = float(np.median(outside))

    # Simetrik normalize yerel yogunluk farki.
    score = abs(inside_med - outside_med) / (
        inside_med + outside_med + 1e-6
    )

    return inside_med, outside_med, score


results = []

for split in ["train", "test"]:
    pool = images[
        (images["split"] == split) &
        (images["n_boxes"] > 0)
    ]

    sample = pool.sample(
        n=min(SAMPLE_SIZE, len(pool)),
        random_state=SEED
    )

    print(f"\n{split.upper()}: {len(sample)} cift secildi")

    for frame in sample.itertuples(index=False):
        stem = frame.stem

        ann = boxes[
            (boxes["stem"] == stem) &
            (boxes["split"] == split)
        ].reset_index(drop=True)

        rgb_path = ROOT / "visible" / split / f"{stem}.jpg"
        ir_path = ROOT / "infrared" / split / f"{stem}.jpg"

        rgb = cv2.imread(
            str(rgb_path), cv2.IMREAD_GRAYSCALE
        )
        ir = cv2.imread(
            str(ir_path), cv2.IMREAD_GRAYSCALE
        )

        if rgb is None or ir is None:
            raise FileNotFoundError(
                f"Goruntu okunamadi: {stem}"
            )

        if rgb.shape != ir.shape:
            raise ValueError(
                f"Modalite boyutlari farkli: {stem}"
            )

        objects = list(ann.itertuples(index=False))

        for j, box in enumerate(objects):
            rgb_result = local_contrast(
                rgb, box, objects, j
            )
            ir_result = local_contrast(
                ir, box, objects, j
            )

            if rgb_result is None or ir_result is None:
                continue

            
            results.append({
                "stem": stem,
                "split": split,

                # Hangi insan kutusunun olculdugu
                "box_index": j,
                "xmin": box.xmin,
                "ymin": box.ymin,
                "xmax": box.xmax,
                "ymax": box.ymax,

                # RGB olcumleri
                "rgb_inside": rgb_result[0],
                "rgb_background": rgb_result[1],
                "rgb_contrast": rgb_result[2],

                # IR olcumleri
                "ir_inside": ir_result[0],
                "ir_background": ir_result[1],
                "ir_contrast": ir_result[2]
            })


res = pd.DataFrame(results)

if res.empty:
    raise RuntimeError("Olculebilir kutu bulunamadi.")

csv_path = LOG / "eda08_object_contrast.csv"
res.to_csv(csv_path, index=False)

# Sayisal ozet
for split in ["train", "test"]:
    sub = res[res["split"] == split]

    print(f"\n===== {split.upper()} =====")
    print(f"Olculen kutu: {len(sub)}")
    print(f"RGB medyan kontrast: {sub['rgb_contrast'].median():.4f}")
    print(f"IR medyan kontrast: {sub['ir_contrast'].median():.4f}")

    stronger = (
        sub["ir_contrast"] > sub["rgb_contrast"]
    ).mean() * 100

    print(f"IR kontrasti daha yuksek kutu: %{stronger:.2f}")

# Eslestirilmis kutu kontrastlari
fig, axes = plt.subplots(
    1, 2, figsize=(13, 6)
)

for ax, split in zip(axes, ["train", "test"]):
    sub = res[res["split"] == split]

    ax.scatter(
        sub["rgb_contrast"],
        sub["ir_contrast"],
        alpha=0.35,
        s=15
    )

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color="gray",
        label="Esit kontrast"
    )

    ax.set_title(
        f"{split.upper()} - {len(sub)} Kutu"
    )
    ax.set_xlabel("RGB yerel kontrast")
    ax.set_ylabel("IR yerel kontrast")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend()

plt.tight_layout()

fig_path = FIG / "fig28_object_contrast.png"
plt.savefig(fig_path, dpi=160)
plt.close()

print(f"\nCSV: {csv_path}")
print(f"Grafik: {fig_path}")
print("\nEDA 08 tamamlandi.")
