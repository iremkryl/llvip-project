
"""
AMAC:
Validation 16-Train 10 ve Validation 13-Train 14/15
arasinda olasi ortak fiziksel sahneleri incelemek.

YONTEM:
1. Nihai bolme ve orijinal sahne gruplari okunur.
2. Her gruptan zamana yayilmis en fazla 160 kare secilir.
3. RGB ve IR icin ayri benzerlik matrisleri hesaplanir.
4. Her grup cifti icin iki IR ve bir RGB adayi secilir.
5. Adaylar RGB ve IR olarak yan yana gorsellestirilir.
6. Mevcut veri bolmesi ve etiketler degistirilmez.
7. Benzerlik skorundan kesin sahne karari verilmez.

GIRDI:
data/processed/final_split.csv
data/raw/LLVIP_extracted/visible/train/
data/raw/LLVIP_extracted/infrared/train/

CIKTI:
outputs/logs/step16g_targeted_scene_candidates.csv
outputs/figures/fig43_targeted_scene_review_en.png
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

GROUP_PAIRS = [
    ("16", "10"),
    ("13", "14"),
    ("13", "15")
]

MAX_IMAGES = 160
FEATURE_SIZE = (80, 64)
THUMB_SIZE = (260, 208)


# 1. Nihai veri bolmesini oku

df = pd.read_csv(
    PROCESSED / "final_split.csv",
    dtype=str,
    keep_default_na=False
)

required = [
    "stem",
    "original_split",
    "final_split",
    "site_group"
]

missing = set(required) - set(df.columns)

if missing:
    raise ValueError(
        f"Eksik sutunlar: {sorted(missing)}"
    )

df["stem"] = df["stem"].str.zfill(6)

if len(df) != 15488:
    raise ValueError(
        "Nihai veri bolmesinde 15488 cift bekleniyor."
    )


# 2. Gruplardan zamana yayilmis ornekler al

def select_stems(group, expected_split):

    part = df[
        (df["original_split"] == "train") &
        (df["site_group"] == group) &
        (df["final_split"] == expected_split)
    ]

    if part.empty:
        raise ValueError(
            f"Grup bulunamadi: {group} / {expected_split}"
        )

    stems = sorted(part["stem"].tolist())

    n = min(len(stems), MAX_IMAGES)

    indices = np.unique(
        np.linspace(
            0,
            len(stems) - 1,
            n
        ).astype(int)
    )

    return [stems[i] for i in indices]


groups_needed = {
    group
    for pair in GROUP_PAIRS
    for group in pair
}

group_stems = {}

for group in sorted(groups_needed):

    role = (
        "val" if group in {"13", "16"}
        else "train"
    )

    group_stems[group] = select_stems(
        group, role
    )


# 3. RGB ve IR ozelliklerini ayri hesapla

def feature(stem, modality):

    path = (
        ROOT / modality / "train" /
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

        values = np.asarray(
            gray, dtype=np.float32
        ).ravel()

    values -= values.mean()

    norm = np.linalg.norm(values)

    if norm < 1e-8:
        return np.zeros_like(values)

    return values / norm


features = {}

for group, stems in group_stems.items():

    for name, folder in [
        ("IR", "infrared"),
        ("RGB", "visible")
    ]:

        features[(group, name)] = np.stack([
            feature(stem, folder)
            for stem in stems
        ])


# 4. Her karsilastirma icin farkli adaylar sec

records = []

for val_group, train_group in GROUP_PAIRS:

    val_stems = group_stems[val_group]
    train_stems = group_stems[train_group]

    already_selected = []

    print(
        f"\n===== VAL {val_group} - TRAIN "
        f"{train_group} ====="
    )

    for modality, needed in [
        ("IR", 2),
        ("RGB", 1)
    ]:

        a = features[(val_group, modality)]
        b = features[(train_group, modality)]

        similarity = a @ b.T

        ranking = np.argsort(
            similarity.ravel()
        )[::-1]

        chosen = 0

        for flat_index in ranking:

            i, j = np.unravel_index(
                flat_index,
                similarity.shape
            )

            # Ayni veya cok yakin kareleri
            # tekrar tekrar gosterme.
            too_close = any(
                (
                    abs(i - old_i) < 10
                    and abs(j - old_j) < 10
                )
                for old_i, old_j in already_selected
            )

            if too_close:
                continue

            already_selected.append((i, j))

            record = {
                "validation_group": val_group,
                "train_group": train_group,
                "selection_modality": modality,
                "candidate_rank": chosen + 1,
                "validation_stem": val_stems[i],
                "train_stem": train_stems[j],
                "similarity_score": float(
                    similarity[i, j]
                )
            }

            records.append(record)

            print(
                f"{modality} aday {chosen + 1}: "
                f"{record['validation_stem']} <-> "
                f"{record['train_stem']} | "
                f"Skor: {record['similarity_score']:.4f}"
            )

            chosen += 1

            if chosen == needed:
                break

        if chosen != needed:
            raise ValueError(
                f"Yeterli aday bulunamadi: "
                f"{val_group}-{train_group}/{modality}"
            )


result = pd.DataFrame(records)

if len(result) != 9:
    raise ValueError(
        f"9 aday bekleniyor: {len(result)}"
    )


# 5. CSV olustur

csv_path = (
    LOG / "step16g_targeted_scene_candidates.csv"
)

result.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)


# 6. Gorsel yardimcilari

def get_font(size):

    for name in ["arial.ttf", "DejaVuSans.ttf"]:

        try:
            return ImageFont.truetype(
                name, size
            )

        except OSError:
            continue

    return ImageFont.load_default()


TITLE_FONT = get_font(21)
LABEL_FONT = get_font(15)
SMALL_FONT = get_font(13)


def load_thumb(stem, modality):

    folder = (
        "infrared" if modality == "IR"
        else "visible"
    )

    path = (
        ROOT / folder / "train" /
        f"{stem}.jpg"
    )

    with Image.open(path) as im:

        return im.convert("RGB").resize(
            THUMB_SIZE,
            Image.Resampling.LANCZOS
        )


# 7. Ingilizce gorsel paneli

image_w, image_h = THUMB_SIZE

margin = 20
gap = 15
top = 85
row_height = 280

sheet_width = (
    margin * 2
    + image_w * 4
    + gap * 3
)

sheet_height = (
    top
    + len(result) * row_height
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
    "Targeted Validation-Training Scene Review",
    fill="black",
    font=TITLE_FONT
)

draw.text(
    (margin, 43),
    "Candidates ranked separately using IR and RGB; "
    "visual verification required",
    fill="#555555",
    font=SMALL_FONT
)


for index, row in result.iterrows():

    y = top + index * row_height

    val_stem = row["validation_stem"]
    train_stem = row["train_stem"]

    heading = (
        f"Validation {row['validation_group']} vs "
        f"Training {row['train_group']} | "
        f"Selected by {row['selection_modality']} | "
        f"Score: {row['similarity_score']:.4f}"
    )

    draw.text(
        (margin, y),
        heading,
        fill="black",
        font=LABEL_FONT
    )

    panels = [
        ("Validation IR", val_stem, "IR"),
        ("Training IR", train_stem, "IR"),
        ("Validation RGB", val_stem, "RGB"),
        ("Training RGB", train_stem, "RGB")
    ]

    for column, (
        label, stem, modality
    ) in enumerate(panels):

        x = margin + column * (
            image_w + gap
        )

        thumbnail = load_thumb(
            stem, modality
        )

        sheet.paste(
            thumbnail,
            (x, y + 25)
        )

        draw.text(
            (x, y + 240),
            f"{label} | {stem}",
            fill="black",
            font=SMALL_FONT
        )

    draw.line(
        (
            margin,
            y + row_height - 5,
            sheet_width - margin,
            y + row_height - 5
        ),
        fill="#dddddd",
        width=2
    )


fig_path = (
    FIG / "fig43_targeted_scene_review_en.png"
)

sheet.save(fig_path)


# 8. Terminal ozeti

print("\n===== ADIM 16G - HEDEFLI SAHNE KONTROLU =====")

print(
    f"Incelenen grup cifti: {len(GROUP_PAIRS)}"
)

print(
    f"Toplam aday karsilastirmasi: {len(result)}"
)

print(
    f"Grup basina maksimum ornek: {MAX_IMAGES}"
)

print("\n===== OLUSTURULAN DOSYALAR =====")
print(f"CSV: {csv_path}")
print(f"Gorsel: {fig_path}")

print(
    "\nNot: RGB ve IR skorlarinin dogrudan "
    "karsilastirilmasi uygun degildir."
)

print(
    "Gorsel benzerlik, sahne kimligini "
    "tek basina kesinlestirmez."
)

print(
    "Nihai veri bolmesi ve YOLO "
    "etiketleri degistirilmedi."
)

print("\nADIM 16G TAMAMLANDI.")
