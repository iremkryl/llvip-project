"""
AMAC:
Kaydedilen YOLO etiketlerini kaynak VOC kutulariyla
sayisal ve gorsel olarak karsilastirmak.

YONTEM:
1. Tum YOLO TXT dosyalari tekrar okunur.
2. Normalize koordinatlar piksel koordinatina cevrilir.
3. Toplam 42432 kutu kaynak kutularla karsilastirilir.
4. Iki bos etiket dosyasi kontrol edilir.
5. Train, val ve testten ornekler secilir.
6. Kaynak ve YOLO kutulari RGB/IR ustune cizilir.
7. Ingilizce gorsel ve kontrol raporu uretilir.
8. Mevcut etiketler ve ham goruntuler degistirilmez.

GIRDI:
data/processed/step17a_label_manifest.csv
data/processed/boxes_cleaned.csv
data/processed/yolo_labels/
data/raw/LLVIP_extracted/

CIKTI:
outputs/logs/step17b_roundtrip_audit.csv
outputs/logs/step17b_yolo_roundtrip_summary.json
outputs/figures/fig40_yolo_label_roundtrip_en.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROCESSED = Path("data/processed")
LOG = Path("outputs/logs")
FIG = Path("outputs/figures")

LOG.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

IMAGE_W = 1280
IMAGE_H = 1024

ERROR_TOLERANCE_PX = 0.001
THUMB_SIZE = (280, 224)
SEED = 42


# 1. Verileri oku

manifest = pd.read_csv(
    PROCESSED / "step17a_label_manifest.csv",
    dtype=str,
    keep_default_na=False
)

boxes = pd.read_csv(
    PROCESSED / "boxes_cleaned.csv",
    dtype=str,
    keep_default_na=False
)

for table in (manifest, boxes):
    table["stem"] = (
        table["stem"].str.strip().str.zfill(6)
    )

for column in ["xmin", "ymin", "xmax", "ymax"]:
    boxes[column] = pd.to_numeric(
        boxes[column],
        errors="raise"
    )

manifest["box_count"] = pd.to_numeric(
    manifest["box_count"],
    errors="raise"
).astype(int)

if len(manifest) != 15488:
    raise ValueError(
        "Manifest 15488 goruntu cifti icermiyor."
    )

if len(boxes) != 42432:
    raise ValueError(
        "Kaynak kutu sayisi 42432 degil."
    )

if manifest.duplicated(
    ["final_split", "stem"]
).any():
    raise ValueError(
        "Manifestte tekrar eden cikti dosyasi var."
    )


# 2. Kaynak kutulari goruntu bazinda grupla

source_lookup = {}

for row in boxes.itertuples(index=False):

    key = (row.split, row.stem)

    coords = [
        row.xmin,
        row.ymin,
        row.xmax,
        row.ymax
    ]

    source_lookup.setdefault(
        key, []
    ).append(coords)


# 3. Kaydedilmis YOLO etiketlerini oku

def read_yolo_labels(path):

    if not path.is_file():
        raise FileNotFoundError(
            f"YOLO etiket dosyasi bulunamadi: {path}"
        )

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    result = []

    for number, line in enumerate(
        lines, start=1
    ):

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Hatali etiket satiri: "
                f"{path}, satir {number}"
            )

        if parts[0] != "0":
            raise ValueError(
                f"Beklenmeyen sinif kimligi: {path}"
            )

        xc, yc, bw, bh = map(
            float, parts[1:]
        )

        values = np.array(
            [xc, yc, bw, bh]
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Gecersiz sayisal deger: {path}"
            )

        if not (
            0 <= xc <= 1 and
            0 <= yc <= 1 and
            0 < bw <= 1 and
            0 < bh <= 1
        ):
            raise ValueError(
                f"Normalize koordinat hatasi: {path}"
            )

        # Sinirlar ondalik yuvarlama toleransiyla
        # kontrol edilir.
        if (
            xc - bw / 2 < -1e-7 or
            yc - bh / 2 < -1e-7 or
            xc + bw / 2 > 1 + 1e-7 or
            yc + bh / 2 > 1 + 1e-7
        ):
            raise ValueError(
                f"Goruntu sinirini asan kutu: {path}"
            )

        result.append([
            (xc - bw / 2) * IMAGE_W,
            (yc - bh / 2) * IMAGE_H,
            (xc + bw / 2) * IMAGE_W,
            (yc + bh / 2) * IMAGE_H
        ])

    return np.asarray(
        result,
        dtype=np.float64
    ).reshape(-1, 4)


# 4. Tum veri setinde sayisal karsilastirma

records = []
decoded_lookup = {}

total_boxes = 0
empty_files = []

for row in manifest.itertuples(index=False):

    key = (
        row.original_split,
        row.stem
    )

    original = np.asarray(
        source_lookup.get(key, []),
        dtype=np.float64
    ).reshape(-1, 4)

    decoded = read_yolo_labels(
        Path(row.label_path)
    )

    if not (
        len(original)
        == len(decoded)
        == row.box_count
    ):
        raise ValueError(
            f"Kutu sayisi uyusmazligi: "
            f"{row.final_split}/{row.stem}"
        )

    if len(original):

        error = float(
            np.max(
                np.abs(original - decoded)
            )
        )

    else:

        error = 0.0

        empty_files.append(
            (
                row.original_split,
                row.stem
            )
        )

    if error > ERROR_TOLERANCE_PX:
        raise ValueError(
            f"YOLO geri donusum hatasi: "
            f"{row.stem}, {error:.6f} piksel"
        )

    decoded_lookup[
        (row.final_split, row.stem)
    ] = decoded

    total_boxes += len(decoded)

    records.append({
        "stem": row.stem,
        "original_split": row.original_split,
        "final_split": row.final_split,
        "box_count": len(decoded),
        "max_coordinate_error_px": error,
        "is_empty_label": len(decoded) == 0,
        "roundtrip_pass": True
    })


audit = pd.DataFrame(records)

if total_boxes != 42432:
    raise ValueError(
        "Toplam donusturulen kutu sayisi hatali."
    )

if len(empty_files) != 2:
    raise ValueError(
        "Bos etiket dosyasi sayisi 2 degil."
    )

expected_empty = {
    ("train", "100030"),
    ("train", "100033")
}

if set(empty_files) != expected_empty:
    raise ValueError(
        f"Beklenmeyen bos etiket kayitlari: "
        f"{empty_files}"
    )

audit_path = (
    LOG / "step17b_roundtrip_audit.csv"
)

audit.to_csv(
    audit_path,
    index=False,
    encoding="utf-8-sig"
)


# 5. Gorsel kontrol icin sabit ornekler sec

rng = np.random.default_rng(SEED)

selected = []

for split in ["train", "val", "test"]:

    subset = audit[
        audit["final_split"] == split
    ]

    # Bir tek-kisilik ve bir kalabalik ornek.
    for description, condition in [
        ("Single person",
         subset["box_count"] == 1),
        ("Multiple persons",
         subset["box_count"] >= 4)
    ]:

        candidates = subset.loc[
            condition, "stem"
        ].to_numpy()

        if len(candidates) == 0:
            raise ValueError(
                f"{split} icin uygun ornek bulunamadi."
            )

        stem = str(
            rng.choice(candidates)
        )

        selected.append(
            (split, stem, description)
        )


# Iki kutusuz goruntuyu de gorsellestir.
for stem in ["100030", "100033"]:
    selected.append(
        ("train", stem, "Empty annotation")
    )


# 6. Gorsel panel yardimcilari

def get_font(size):

    for name in [
        "arial.ttf",
        "DejaVuSans.ttf"
    ]:
        try:
            return ImageFont.truetype(
                name, size
            )
        except OSError:
            continue

    return ImageFont.load_default()


TITLE_FONT = get_font(21)
LABEL_FONT = get_font(16)
SMALL_FONT = get_font(14)


def annotated_thumbnail(
    path,
    coordinates,
    color
):

    with Image.open(path) as im:
        image = im.convert("RGB")

    if image.size != (
        IMAGE_W, IMAGE_H
    ):
        raise ValueError(
            f"Beklenmeyen goruntu boyutu: {path}"
        )

    image = image.resize(
        THUMB_SIZE,
        Image.Resampling.LANCZOS
    )

    draw = ImageDraw.Draw(image)

    sx = THUMB_SIZE[0] / IMAGE_W
    sy = THUMB_SIZE[1] / IMAGE_H

    for box in coordinates:

        x1, y1, x2, y2 = box

        draw.rectangle(
            (
                x1 * sx,
                y1 * sy,
                x2 * sx,
                y2 * sy
            ),
            outline=color,
            width=2
        )

    return image


# 7. Ingilizce karsilastirma gorseli

image_w, image_h = THUMB_SIZE

margin = 22
gap = 16
top = 112
row_height = 274

sheet_width = (
    2 * margin
    + 4 * image_w
    + 3 * gap
)

sheet_height = (
    top
    + len(selected) * row_height
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
    "VOC Annotations vs Reconstructed YOLO Labels",
    fill="black",
    font=TITLE_FONT
)

draw.text(
    (margin, 43),
    "Same image pairs and coordinates; "
    "comparison across RGB and IR",
    fill="#555555",
    font=SMALL_FONT
)

headings = [
    "RGB - VOC Boxes",
    "RGB - YOLO Boxes",
    "IR - VOC Boxes",
    "IR - YOLO Boxes"
]

for column, heading in enumerate(headings):

    x = margin + column * (
        image_w + gap
    )

    draw.text(
        (x, 81),
        heading,
        fill="black",
        font=LABEL_FONT
    )


manifest_lookup = {
    (row.final_split, row.stem): row
    for row in manifest.itertuples(index=False)
}

for index, (
    split, stem, description
) in enumerate(selected):

    y = top + index * row_height

    row = manifest_lookup[
        (split, stem)
    ]

    key = (
        row.original_split,
        stem
    )

    original = np.asarray(
        source_lookup.get(key, []),
        dtype=np.float64
    ).reshape(-1, 4)

    decoded = decoded_lookup[
        (split, stem)
    ]

    draw.text(
        (margin, y),
        (
            f"{split.upper()} | Image {stem} | "
            f"{description} | "
            f"{len(original)} box(es)"
        ),
        fill="black",
        font=LABEL_FONT
    )

    panels = [
        (
            row.rgb_path,
            original,
            "#00d9ff"
        ),
        (
            row.rgb_path,
            decoded,
            "#ffe500"
        ),
        (
            row.ir_path,
            original,
            "#00d9ff"
        ),
        (
            row.ir_path,
            decoded,
            "#ffe500"
        )
    ]

    for column, (
        path, coordinates, color
    ) in enumerate(panels):

        x = margin + column * (
            image_w + gap
        )

        thumb = annotated_thumbnail(
            Path(path),
            coordinates,
            color
        )

        sheet.paste(
            thumb,
            (x, y + 25)
        )

    draw.line(
        (
            margin,
            y + row_height - 8,
            sheet_width - margin,
            y + row_height - 8
        ),
        fill="#dddddd",
        width=2
    )


fig_path = (
    FIG / "fig40_yolo_label_roundtrip_en.png"
)

sheet.save(fig_path)


# 8. JSON ozetini olustur

max_error = float(
    audit["max_coordinate_error_px"].max()
)

split_summary = {}

for split in ["train", "val", "test"]:

    part = audit[
        audit["final_split"] == split
    ]

    split_summary[split] = {
        "image_pairs": int(len(part)),
        "boxes": int(part["box_count"].sum()),
        "maximum_pixel_error": float(
            part["max_coordinate_error_px"].max()
        )
    }

summary = {
    "verified_image_pairs": len(audit),
    "verified_boxes": total_boxes,
    "empty_label_files": len(empty_files),
    "empty_image_stems": [
        stem for _, stem in empty_files
    ],
    "maximum_coordinate_error_pixels": max_error,
    "accepted_error_tolerance_pixels": (
        ERROR_TOLERANCE_PX
    ),
    "all_roundtrip_checks_passed": True,
    "split_summary": split_summary,
    "visual_samples": [
        {
            "split": split,
            "stem": stem,
            "description": description
        }
        for split, stem, description in selected
    ],
    "limitations": [
        "Numerical reconstruction does not prove "
        "annotation correctness.",
        "Identical coordinates in RGB and IR "
        "do not prove spatial alignment.",
        "Visual examples are illustrative, "
        "not an independent accuracy estimate."
    ]
}

json_path = (
    LOG / "step17b_yolo_roundtrip_summary.json"
)

json_path.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# 9. Terminal sonuclari

print("\n===== ADIM 17B - YOLO GERI CIZIM KONTROLU =====")

print(
    f"Kontrol edilen goruntu cifti: {len(audit)}"
)

print(
    f"Kontrol edilen insan kutusu: {total_boxes}"
)

print(
    f"Bos etiket dosyasi: {len(empty_files)}"
)

print("\n===== VERI BOLMELERI =====")

for split in ["train", "val", "test"]:

    part = split_summary[split]

    print(
        f"{split.upper()}: "
        f"{part['image_pairs']} cift | "
        f"{part['boxes']} kutu | "
        f"Maks. hata: "
        f"{part['maximum_pixel_error']:.8f} px"
    )


print("\n===== SAYISAL DOGRULAMA =====")

print(
    "Tum kutular geri donusturuldu: EVET"
)

print(
    "Kaynak ve YOLO kutu sayilari tutarli: EVET"
)

print(
    f"Maksimum koordinat farki: "
    f"{max_error:.8f} piksel"
)

print(
    f"Kabul edilen tolerans: "
    f"{ERROR_TOLERANCE_PX:.3f} piksel"
)

print(
    "Toleransi asan kutu: 0"
)


print("\n===== GORSEL KONTROL ORNEKLERI =====")

for split, stem, description in selected:

    print(
        f"{split.upper()} / {stem} / {description}"
    )


print("\n===== OLUSTURULAN DOSYALAR =====")

print(f"CSV: {audit_path}")
print(f"JSON: {json_path}")
print(f"Gorsel: {fig_path}")

print(
    "\nNot: Sayisal eslesme kutularin gercek "
    "insanlara dogru oturdugunu kanitlamaz."
)

print("\nADIM 17B TAMAMLANDI.")
