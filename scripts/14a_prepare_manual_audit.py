
"""
ADIM 14A - MANUEL ETIKET DENETIMI ICIN ORNEKLEM HAZIRLIGI

AMAC:
LLVIP veri setinden 100 RGB-IR goruntu ciftini
tekrarlanabilir bicimde secerek manuel etiket
kalitesi denetimine hazirlamak.

YONTEM:
1. Goruntu envanteri ve temiz anotasyonlar okunur.
2. Seed=42 ile tum veri setinden 100 cift secilir.
3. Split, sahne grubu ve kalite bayraklari eklenir.
4. RGB ve IR goruntulerine insan kutulari cizilir.
5. Her sayfada 10 cift bulunan Ingilizce paneller uretilir.
6. Manuel denetim alanlari bos birakilir.
7. Ham goruntuler ve anotasyonlar degistirilmez.

GIRDI:
outputs/logs/images_table.csv
outputs/logs/frame_site_groups.csv
data/processed/boxes_cleaned.csv
data/processed/quality_flags.csv
data/raw/LLVIP_extracted/

CIKTI:
outputs/logs/manual_audit_100.csv
outputs/figures/manual_audit/contact_01.png ... contact_10.png
Terminal: Orneklem ve sahne grubu dagilimlari.
"""

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("data/raw/LLVIP_extracted")
LOG = Path("outputs/logs")
PROCESSED = Path("data/processed")
OUT = Path("outputs/figures/manual_audit")

OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
SAMPLE_SIZE = 100

THUMB_SIZE = (320, 256)
CARD_WIDTH = 700
CARD_HEIGHT = 340
SHEET_COLUMNS = 2
SHEET_ROWS = 5
PER_SHEET = SHEET_COLUMNS * SHEET_ROWS


# 1. Yazilar icin font
def get_font(size):
    for font_name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue

    return ImageFont.load_default()


FONT = get_font(17)
SMALL_FONT = get_font(15)


# 2. Kayitlari oku
images = pd.read_csv(
    LOG / "images_table.csv",
    dtype={"stem": str, "prefix": str}
)

groups = pd.read_csv(
    LOG / "frame_site_groups.csv",
    dtype={"stem": str, "site_group": str}
)

boxes = pd.read_csv(
    PROCESSED / "boxes_cleaned.csv",
    dtype={"stem": str}
)

flags = pd.read_csv(
    PROCESSED / "quality_flags.csv",
    dtype={"stem": str}
)

for table in [images, groups, boxes, flags]:
    table["stem"] = table["stem"].str.zfill(6)


# 3. Veri tablolari birlestir
df = images.merge(
    groups[["stem", "split", "site_group"]],
    on=["stem", "split"],
    how="left",
    validate="one_to_one"
)

flag_cols = [
    "stem",
    "split",
    "flag_any_review",
    "flag_low_rgb",
    "flag_invalid_box_removed",
    "flag_duplicate_rgb",
    "flag_duplicate_ir"
]

df = df.merge(
    flags[flag_cols],
    on=["stem", "split"],
    how="left",
    validate="one_to_one"
)

if df["site_group"].isna().any():
    raise ValueError("Missing scene group information.")

if df["flag_any_review"].isna().any():
    raise ValueError("Missing quality flag information.")


# 4. Seed=42 ile 100 goruntu cifti sec
sample = df.sample(
    n=SAMPLE_SIZE,
    random_state=SEED
).copy()

sample = sample.sort_values(
    ["split", "stem"]
).reset_index(drop=True)

sample.insert(
    0, "audit_id",
    range(1, len(sample) + 1)
)


# 5. Temiz anotasyon sayilarini ekle
counts = (
    boxes.groupby(["stem", "split"])
    .size()
    .rename("clean_box_count")
    .reset_index()
)

sample = sample.merge(
    counts,
    on=["stem", "split"],
    how="left",
    validate="one_to_one"
)

sample["clean_box_count"] = (
    sample["clean_box_count"]
    .fillna(0)
    .astype(int)
)


# 6. Manuel denetim alanlari
manual_columns = [
    "review_status",
    "missing_person",
    "incorrect_box",
    "uncertain_annotation",
    "rgb_visibility",
    "ir_visibility",
    "rgb_box_alignment",
    "notes"
]

for column in manual_columns:
    sample[column] = ""

csv_path = LOG / "manual_audit_100.csv"

# Daha once doldurulmus denetim kayitlarini koru.
if csv_path.exists():
    old = pd.read_csv(
        csv_path, dtype=str, keep_default_na=False
    )

    existing_columns = [
        c for c in manual_columns if c in old.columns
    ]

    if existing_columns and old[
        existing_columns
    ].astype(str).apply(
        lambda col: col.str.strip().ne("")
    ).any().any():
        raise RuntimeError(
            "Manual audit already contains reviews. "
            "Existing annotations were not overwritten."
        )

sample.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)


# 7. RGB ve IR goruntulerini kutularla hazirla
def load_annotated_image(stem, split, modality):

    path = (
        ROOT / modality / split / f"{stem}.jpg"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with Image.open(path) as im:
        original = im.convert("RGB")

    ow, oh = original.size

    resized = original.resize(
        THUMB_SIZE,
        Image.Resampling.LANCZOS
    )

    sx = THUMB_SIZE[0] / ow
    sy = THUMB_SIZE[1] / oh

    draw = ImageDraw.Draw(resized)

    annotations = boxes[
        (boxes["stem"] == stem) &
        (boxes["split"] == split)
    ]

    for _, box in annotations.iterrows():

        rect = (
            float(box["xmin"]) * sx,
            float(box["ymin"]) * sy,
            float(box["xmax"]) * sx,
            float(box["ymax"]) * sy
        )

        draw.rectangle(
            rect,
            outline="lime",
            width=2
        )

    return resized


# 8. Ingilizce baslikli inceleme karti
def make_card(row):

    stem = row["stem"]
    split = row["split"]

    rgb = load_annotated_image(
        stem, split, "visible"
    )

    ir = load_annotated_image(
        stem, split, "infrared"
    )

    card = Image.new(
        "RGB",
        (CARD_WIDTH, CARD_HEIGHT),
        "white"
    )

    draw = ImageDraw.Draw(card)

    draw.rectangle(
        (0, 0, CARD_WIDTH - 1, CARD_HEIGHT - 1),
        outline="#cccccc",
        width=2
    )

    flagged = (
        str(row["flag_any_review"]).lower()
        == "true"
    )

    flag_text = "YES" if flagged else "NO"

    title = (
        f"#{int(row['audit_id']):03d} | "
        f"{split.upper()} | "
        f"Image ID: {stem} | "
        f"Boxes: {int(row['clean_box_count'])} | "
        f"Review flag: {flag_text}"
    )

    draw.text(
        (15, 10),
        title,
        fill="black",
        font=FONT
    )

    card.paste(rgb, (15, 42))
    card.paste(ir, (365, 42))

    draw.text(
        (15, 310),
        "RGB (Visible)",
        fill="black",
        font=SMALL_FONT
    )

    draw.text(
        (365, 310),
        "IR (Infrared)",
        fill="black",
        font=SMALL_FONT
    )

    return card


# 9. Her sayfada 10 eslestirilmis goruntu cifti
for start in range(0, len(sample), PER_SHEET):

    page = sample.iloc[
        start:start + PER_SHEET
    ]

    sheet = Image.new(
        "RGB",
        (
            CARD_WIDTH * SHEET_COLUMNS,
            CARD_HEIGHT * SHEET_ROWS
        ),
        "#eeeeee"
    )

    for index, (_, row) in enumerate(
        page.iterrows()
    ):

        card = make_card(row)

        x = (
            index % SHEET_COLUMNS
        ) * CARD_WIDTH

        y = (
            index // SHEET_COLUMNS
        ) * CARD_HEIGHT

        sheet.paste(card, (x, y))

    page_number = start // PER_SHEET + 1

    output = (
        OUT / f"contact_{page_number:02d}.png"
    )

    sheet.save(output)

    print(
        f"Contact sheet {page_number:02d} saved: "
        f"{output}"
    )


# 10. Orneklem istatistikleri
print("\n===== STEP 14A - SAMPLE SUMMARY =====")

print(f"Total selected image pairs: {len(sample)}")

print("\nTrain/Test distribution:")
print(
    sample["split"]
    .value_counts()
    .to_string()
)

print("\nScene group distribution:")
print(
    sample["site_group"]
    .value_counts()
    .sort_index()
    .to_string()
)

flagged_count = (
    sample["flag_any_review"]
    .astype(str)
    .str.lower()
    .eq("true")
    .sum()
)

print(
    f"\nImage pairs with review flags: {flagged_count}"
)

print(
    "Total valid bounding boxes: "
    f"{sample['clean_box_count'].sum()}"
)

print(f"\nAudit CSV: {csv_path}")
print(f"Contact sheets: {OUT}")

print("\nSTEP 14A COMPLETED.")
