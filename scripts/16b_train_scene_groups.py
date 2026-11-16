"""
AMAC:
Validation bolmesi oncesinde train icindeki
04-07, 08-09 ve 14-15 sahne gruplarini incelemek.

YONTEM:
1. Her gruptan zamana yayilmis IR ornekleri alinir.
2. Normalize edilmis goruntu benzerligi hesaplanir.
3. Her grup cifti icin iki farkli aday secilir.
4. IR ve RGB goruntuler yan yana gosterilir.
5. Kesin grup birlestirme karari verilmez.
6. Ham veri ve mevcut gruplar degistirilmez.

GIRDI:
data/raw/LLVIP_extracted/infrared/train/
data/raw/LLVIP_extracted/visible/train/

CIKTI:
outputs/logs/step16b_train_group_candidates.csv
outputs/figures/fig38_train_scene_groups_en.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path("data/raw/LLVIP_extracted")
LOG = Path("outputs/logs")
FIG = Path("outputs/figures")

LOG.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

GROUP_PAIRS = [
    ("04", "07"),
    ("08", "09"),
    ("14", "15")
]

MAX_IMAGES = 140
FEATURE_SIZE = (80, 64)
THUMB_SIZE = (260, 208)
CANDIDATES_PER_PAIR = 2


def spread_sample(paths, maximum):
    if not paths:
        raise ValueError("Bos goruntu grubu bulundu.")

    count = min(len(paths), maximum)

    indices = np.unique(
        np.linspace(
            0, len(paths) - 1, count
        ).astype(int)
    )

    return [paths[i] for i in indices]


def get_feature(path):
    with Image.open(path) as im:
        gray = im.convert("L")
        gray = gray.resize(
            FEATURE_SIZE,
            Image.Resampling.BILINEAR
        )
        gray = gray.filter(
            ImageFilter.GaussianBlur(radius=1)
        )
        arr = np.asarray(
            gray, dtype=np.float32
        ).ravel()

    arr -= arr.mean()

    norm = np.linalg.norm(arr)

    if norm < 1e-8:
        return np.zeros_like(arr)

    return arr / norm


def get_font(size):
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass

    return ImageFont.load_default()


def load_thumbnail(prefix, stem, modality):
    path = (
        ROOT / modality / "train" / f"{stem}.jpg"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with Image.open(path) as im:
        return im.convert("RGB").resize(
            THUMB_SIZE,
            Image.Resampling.LANCZOS
        )


records = []

print("\n===== ADIM 16B - TRAIN SAHNE GRUPLARI =====")

for prefix_a, prefix_b in GROUP_PAIRS:

    files_a = sorted(
        (ROOT / "infrared" / "train")
        .glob(f"{prefix_a}*.jpg")
    )

    files_b = sorted(
        (ROOT / "infrared" / "train")
        .glob(f"{prefix_b}*.jpg")
    )

    if not files_a or not files_b:
        raise FileNotFoundError(
            f"Eksik IR grubu: {prefix_a} veya {prefix_b}"
        )

    sample_a = spread_sample(
        files_a, MAX_IMAGES
    )
    sample_b = spread_sample(
        files_b, MAX_IMAGES
    )

    features_a = np.stack([
        get_feature(p) for p in sample_a
    ])

    features_b = np.stack([
        get_feature(p) for p in sample_b
    ])

    similarity = features_a @ features_b.T

    ranking = np.argsort(
        similarity.ravel()
    )[::-1]

    selected_indices = []

    # Secilen iki adayin hemen hemen ayni
    # karelerden olusmasini engellemeye calis.
    min_distance_a = max(
        5, len(sample_a) // 12
    )

    min_distance_b = max(
        5, len(sample_b) // 12
    )

    for flat_index in ranking:

        i, j = np.unravel_index(
            flat_index,
            similarity.shape
        )

        too_close = any(
            abs(i - old_i) < min_distance_a
            or abs(j - old_j) < min_distance_b
            for old_i, old_j in selected_indices
        )

        if too_close:
            continue

        selected_indices.append((i, j))

        if (
            len(selected_indices)
            == CANDIDATES_PER_PAIR
        ):
            break

    if len(selected_indices) < CANDIDATES_PER_PAIR:
        raise ValueError(
            f"{prefix_a}-{prefix_b} icin yeterli "
            "farkli aday bulunamadi."
        )

    print(
        f"\nGrup {prefix_a} - {prefix_b}: "
        f"{len(files_a)} / {len(files_b)} goruntu"
    )

    for rank, (i, j) in enumerate(
        selected_indices, start=1
    ):

        record = {
            "group_a": prefix_a,
            "group_b": prefix_b,
            "candidate_rank": rank,
            "stem_a": sample_a[i].stem,
            "stem_b": sample_b[j].stem,
            "ir_similarity": float(
                similarity[i, j]
            )
        }

        records.append(record)

        print(
            f"  Aday {rank}: "
            f"{record['stem_a']} <-> "
            f"{record['stem_b']} | "
            f"Skor: {record['ir_similarity']:.4f}"
        )


# Sonuclari kaydet
result = pd.DataFrame(records)

csv_path = (
    LOG / "step16b_train_group_candidates.csv"
)

result.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)


# Ingilizce gorsel paneli
title_font = get_font(21)
label_font = get_font(15)
small_font = get_font(13)

image_w, image_h = THUMB_SIZE
left_margin = 20
column_gap = 14

row_height = 270
top_margin = 83

sheet_width = (
    2 * left_margin
    + 4 * image_w
    + 3 * column_gap
)

sheet_height = (
    top_margin
    + len(records) * row_height
    + 25
)

sheet = Image.new(
    "RGB",
    (sheet_width, sheet_height),
    "white"
)

draw = ImageDraw.Draw(sheet)

draw.text(
    (left_margin, 12),
    "Potential Shared Scenes Within the Training Set",
    fill="black",
    font=title_font
)

draw.text(
    (left_margin, 44),
    "IR similarity ranks candidates; static landmarks "
    "require visual verification",
    fill="#555555",
    font=small_font
)


for row_number, record in result.iterrows():

    y = top_margin + row_number * row_height

    group_a = record["group_a"]
    group_b = record["group_b"]

    stem_a = record["stem_a"]
    stem_b = record["stem_b"]

    score = record["ir_similarity"]

    heading = (
        f"Train {group_a} vs Train {group_b} | "
        f"Candidate {int(record['candidate_rank'])} | "
        f"IR similarity: {score:.4f}"
    )

    draw.text(
        (left_margin, y),
        heading,
        fill="black",
        font=label_font
    )

    panels = [
        (f"Group {group_a} - IR",
         group_a, stem_a, "infrared"),
        (f"Group {group_b} - IR",
         group_b, stem_b, "infrared"),
        (f"Group {group_a} - RGB",
         group_a, stem_a, "visible"),
        (f"Group {group_b} - RGB",
         group_b, stem_b, "visible")
    ]

    for column, (label, prefix, stem, modality) in (
        enumerate(panels)
    ):

        x = (
            left_margin
            + column * (image_w + column_gap)
        )

        thumbnail = load_thumbnail(
            prefix, stem, modality
        )

        sheet.paste(
            thumbnail, (x, y + 24)
        )

        draw.text(
            (x, y + 238),
            f"{label} | {stem}",
            fill="black",
            font=small_font
        )

    draw.line(
        (
            left_margin,
            y + row_height - 5,
            sheet_width - left_margin,
            y + row_height - 5
        ),
        fill="#dddddd",
        width=2
    )


fig_path = (
    FIG / "fig38_train_scene_groups_en.png"
)

sheet.save(fig_path)


print("\n===== ADIM 16B - DOSYALAR =====")
print(f"Toplam aday cifti: {len(result)}")
print(f"CSV: {csv_path}")
print(f"Gorsel: {fig_path}")

print(
    "\nNot: Yuksek benzerlik skoru tek basina "
    "grup birlestirme gerekcesi degildir."
)

print("\nADIM 16B TAMAMLANDI.")