"""
AMAC:
Validation ve train arasinda ayni fiziksel sahnenin
bulunma riskini sistematik olarak incelemek.

YONTEM:
1. Nihai veri bolmesinden train/val gruplari okunur.
2. Her gruptan zamana yayilmis IR ornekleri secilir.
3. Tum validation-train grup ciftleri karsilastirilir.
4. Her grup cifti icin en benzer kareler bulunur.
5. Yedi oncelikli cift ve grup 11'in en yakin
   train adayi gorsel inceleme icin secilir.
6. RGB ve IR goruntuleri Ingilizce panelde gosterilir.
7. Veri bolmesi veya anotasyonlar degistirilmez.

GIRDI:
data/processed/final_split.csv
data/raw/LLVIP_extracted/visible/train/
data/raw/LLVIP_extracted/infrared/train/

CIKTI:
outputs/logs/step16f_all_group_comparisons.csv
outputs/logs/step16f_priority_comparisons.csv
outputs/figures/fig42_validation_train_scene_audit_en.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter


ROOT = Path("data/raw/LLVIP_extracted")
PROCESSED = Path("data/processed")
LOG = Path("outputs/logs")
FIG = Path("outputs/figures")

LOG.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SAMPLE_PER_GROUP = 60
FEATURE_SIZE = (80, 64)
THUMB_SIZE = (270, 216)

VALIDATION_GROUPS = {"06", "11", "13", "16"}

PRIORITY_PAIRS = [
    ("06", "05"),
    ("16", "05"),
    ("16", "10"),
    ("16", "17"),
    ("16", "18"),
    ("13", "10"),
    ("13", "14_15")
]


# 1. Veri bolmesini oku

df = pd.read_csv(
    PROCESSED / "final_split.csv",
    dtype=str,
    keep_default_na=False
)

required = {
    "stem",
    "original_split",
    "final_split",
    "validation_group"
}

if not required.issubset(df.columns):
    raise ValueError(
        f"Eksik sutun: {sorted(required - set(df.columns))}"
    )

df["stem"] = df["stem"].str.zfill(6)

original_train = df[
    df["original_split"] == "train"
].copy()

if len(original_train) != 12025:
    raise ValueError(
        "Orijinal train 12025 cift icermiyor."
    )

if original_train[
    "validation_group"
].eq("").any():
    raise ValueError(
        "Bazi train goruntulerinin grubu eksik."
    )

if original_train.duplicated("stem").any():
    raise ValueError(
        "Tekrarlanan train stem bulundu."
    )

val_groups = set(
    original_train.loc[
        original_train["final_split"] == "val",
        "validation_group"
    ]
)

if val_groups != VALIDATION_GROUPS:
    raise ValueError(
        f"Validation gruplari beklenenden farkli: "
        f"{val_groups}"
    )

train_groups = set(
    original_train.loc[
        original_train["final_split"] == "train",
        "validation_group"
    ]
)

if val_groups & train_groups:
    raise ValueError(
        "Ayni grup train ve validation'da bulunuyor."
    )

if len(train_groups) != 12:
    raise ValueError(
        f"12 train grubu bekleniyor: {len(train_groups)}"
    )


# 2. Zamana yayilmis goruntuler sec

def spread_sample(stems, maximum):

    stems = sorted(stems)

    n = min(
        len(stems),
        maximum
    )

    indices = np.unique(
        np.linspace(
            0,
            len(stems) - 1,
            n
        ).astype(int)
    )

    return [
        stems[i]
        for i in indices
    ]


group_samples = {}

for group, part in original_train.groupby(
    "validation_group"
):

    group_samples[group] = spread_sample(
        part["stem"].tolist(),
        SAMPLE_PER_GROUP
    )


# 3. IR yapisal ozelliklerini hesapla

def extract_feature(stem):

    path = (
        ROOT / "infrared" / "train" /
        f"{stem}.jpg"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

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
            gray,
            dtype=np.float32
        ).ravel()

    arr = arr - arr.mean()

    norm = float(
        np.linalg.norm(arr)
    )

    if norm < 1e-8:
        return np.zeros_like(arr)

    return arr / norm


features = {}

for group, stems in group_samples.items():

    features[group] = np.stack([
        extract_feature(stem)
        for stem in stems
    ])


# 4. Tum validation-train grup ciftlerini tara

records = []

for val_group in sorted(val_groups):

    for train_group in sorted(train_groups):

        a = features[val_group]
        b = features[train_group]

        similarity = a @ b.T

        i, j = np.unravel_index(
            int(np.argmax(similarity)),
            similarity.shape
        )

        records.append({
            "validation_group": val_group,
            "train_group": train_group,
            "validation_stem": (
                group_samples[val_group][i]
            ),
            "train_stem": (
                group_samples[train_group][j]
            ),
            "ir_similarity": float(
                similarity[i, j]
            ),
            "validation_sample_count": len(a),
            "train_sample_count": len(b)
        })


result = pd.DataFrame(records)

if len(result) != 48:
    raise ValueError(
        f"48 grup karsilastirmasi bekleniyor: "
        f"{len(result)}"
    )

result = result.sort_values(
    [
        "validation_group",
        "ir_similarity"
    ],
    ascending=[True, False]
).reset_index(drop=True)

all_path = (
    LOG / "step16f_all_group_comparisons.csv"
)

result.to_csv(
    all_path,
    index=False,
    encoding="utf-8-sig"
)


# 5. Oncelikli sahne adaylarini sec

priority_rows = []

for val_group, train_group in PRIORITY_PAIRS:

    match = result[
        (result["validation_group"] == val_group) &
        (result["train_group"] == train_group)
    ]

    if len(match) != 1:
        raise ValueError(
            f"Eslesme bulunamadi: {val_group}-{train_group}"
        )

    record = match.iloc[0].to_dict()

    record["reason"] = "Previously suspected pair"

    priority_rows.append(record)


# Grup 11 icin en benzer train grubunu ekle.
group_11 = (
    result[
        result["validation_group"] == "11"
    ]
    .sort_values(
        "ir_similarity",
        ascending=False
    )
)

nearest_11 = group_11.iloc[0].to_dict()

nearest_11["reason"] = (
    "Closest training group to validation group 11"
)

priority_rows.append(nearest_11)

priority = pd.DataFrame(priority_rows)

priority.insert(
    0,
    "review_id",
    range(1, len(priority) + 1)
)

priority_path = (
    LOG / "step16f_priority_comparisons.csv"
)

priority.to_csv(
    priority_path,
    index=False,
    encoding="utf-8-sig"
)


# 6. Gorsel panel yardimcilari

def get_font(size):

    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue

    return ImageFont.load_default()


TITLE_FONT = get_font(21)
LABEL_FONT = get_font(15)
SMALL_FONT = get_font(13)


def thumbnail(stem, modality):

    path = (
        ROOT / modality / "train" /
        f"{stem}.jpg"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    with Image.open(path) as im:

        image = im.convert("RGB")

        return image.resize(
            THUMB_SIZE,
            Image.Resampling.LANCZOS
        )


# 7. Ingilizce gorsel paneli

image_w, image_h = THUMB_SIZE

margin = 20
gap = 15
top = 90
row_height = 285

sheet_width = (
    margin * 2
    + image_w * 4
    + gap * 3
)

sheet_height = (
    top
    + len(priority) * row_height
    + 20
)

sheet = Image.new(
    "RGB",
    (sheet_width, sheet_height),
    "white"
)

draw = ImageDraw.Draw(sheet)

draw.text(
    (margin, 12),
    "Potential Scene Overlap Between Validation and Training",
    fill="black",
    font=TITLE_FONT
)

draw.text(
    (margin, 43),
    "IR-based screening of 48 group pairs; "
    "visual assessment required",
    fill="#555555",
    font=SMALL_FONT
)

for i, row in priority.iterrows():

    y = top + i * row_height

    val_group = row["validation_group"]
    train_group = row["train_group"]

    val_stem = row["validation_stem"]
    train_stem = row["train_stem"]

    score = row["ir_similarity"]

    draw.text(
        (margin, y),
        (
            f"Validation {val_group} vs Training "
            f"{train_group} | IR score: {score:.4f}"
        ),
        fill="black",
        font=LABEL_FONT
    )

    panels = [
        ("Validation - IR",
         val_stem, "infrared"),
        ("Training - IR",
         train_stem, "infrared"),
        ("Validation - RGB",
         val_stem, "visible"),
        ("Training - RGB",
         train_stem, "visible")
    ]

    for j, (label, stem, modality) in enumerate(
        panels
    ):

        x = (
            margin +
            j * (image_w + gap)
        )

        image = thumbnail(
            stem, modality
        )

        sheet.paste(
            image,
            (x, y + 25)
        )

        draw.text(
            (x, y + 246),
            f"{label} | {stem}",
            fill="black",
            font=SMALL_FONT
        )

    draw.line(
        (
            margin,
            y + row_height - 6,
            sheet_width - margin,
            y + row_height - 6
        ),
        fill="#dddddd",
        width=2
    )


fig_path = (
    FIG / "fig42_validation_train_scene_audit_en.png"
)

sheet.save(fig_path)


# 8. Terminal ozeti

print("\n===== ADIM 16F - VALIDATION SAHNE DENETIMI =====")

print(
    f"Validation sahne grubu: {len(val_groups)}"
)

print(
    f"Train sahne grubu: {len(train_groups)}"
)

print(
    f"Karsilastirilan grup cifti: {len(result)}"
)

print(
    f"Grup basina en fazla ornek: {SAMPLE_PER_GROUP}"
)

print("\n===== VALIDATION GRUPLARININ EN YAKIN ADAYLARI =====")

for group in sorted(val_groups):

    subset = result[
        result["validation_group"] == group
    ].head(3)

    print(f"\nValidation {group}:")

    for row in subset.itertuples(index=False):

        print(
            f"  Train {row.train_group}: "
            f"{row.ir_similarity:.4f} | "
            f"{row.validation_stem} <-> "
            f"{row.train_stem}"
        )


print("\n===== ONCELIKLI GORSEL INCELEME =====")

for row in priority.itertuples(index=False):

    print(
        f"{row.review_id}. "
        f"Val {row.validation_group} - "
        f"Train {row.train_group} | "
        f"Skor: {row.ir_similarity:.4f}"
    )


print("\n===== OLUSTURULAN DOSYALAR =====")

print(f"CSV: {all_path}")
print(f"CSV: {priority_path}")
print(f"Gorsel: {fig_path}")

print(
    "\nNot: Yuksek piksel benzerligi, "
    "ayni fiziksel sahnenin kesin kaniti degildir."
)

print(
    "Mevcut final_split.csv ve YOLO etiketleri "
    "degistirilmedi."
)

print("\nADIM 16F TAMAMLANDI.")