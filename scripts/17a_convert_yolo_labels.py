"""
AMAC:
Temizlenmis LLVIP insan kutularini YOLO etiket
formatina donusturmek ve nihai veri bolmesini korumak.

YONTEM:
1. Final split ve temiz kutu tablolari okunur.
2. Her kutu YOLO'nun normalize xywh bicimine cevrilir.
3. Koordinatlar ters donusturulerek kontrol edilir.
4. Her goruntu icin ayri TXT etiket dosyasi uretilir.
5. Kutusuz goruntuler icin bos TXT dosyasi olusturulur.
6. RGB, IR ve etiket dosyalarinin yolu kaydedilir.
7. Ham veriler ve onceki sonuclar degistirilmez.

GIRDI:
data/processed/final_split.csv
data/processed/boxes_cleaned.csv
data/raw/LLVIP_extracted/

CIKTI:
data/processed/yolo_labels/train/
data/processed/yolo_labels/val/
data/processed/yolo_labels/test/
data/processed/step17a_label_manifest.csv
outputs/logs/step17a_yolo_conversion_summary.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("data/raw/LLVIP_extracted")
PROCESSED = Path("data/processed")
LOG = Path("outputs/logs")

LABEL_ROOT = PROCESSED / "yolo_labels"
MANIFEST_PATH = PROCESSED / "step17a_label_manifest.csv"
SUMMARY_PATH = LOG / "step17a_yolo_conversion_summary.json"

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 1024

CLASS_ID = 0
DECIMALS = 8

EXPECTED_IMAGES = {
    "train": 9937,
    "val": 2088,
    "test": 3463
}

EXPECTED_BOXES = {
    "train": 28215,
    "val": 5915,
    "test": 8302
}


# 1. Verileri oku

final = pd.read_csv(
    PROCESSED / "final_split.csv",
    dtype=str,
    keep_default_na=False
)

boxes = pd.read_csv(
    PROCESSED / "boxes_cleaned.csv",
    dtype=str,
    keep_default_na=False
)


# 2. Zorunlu alanlari dogrula

required_final = {
    "stem",
    "original_split",
    "final_split",
    "clean_box_count"
}

required_boxes = {
    "stem",
    "split",
    "xmin",
    "ymin",
    "xmax",
    "ymax"
}

if not required_final.issubset(final.columns):
    raise ValueError(
        "final_split.csv dosyasinda eksik sutun var."
    )

if not required_boxes.issubset(boxes.columns):
    raise ValueError(
        "boxes_cleaned.csv dosyasinda eksik sutun var."
    )

for table in (final, boxes):
    table["stem"] = (
        table["stem"].astype(str)
        .str.strip()
        .str.zfill(6)
    )

final["original_split"] = (
    final["original_split"].str.strip().str.lower()
)

final["final_split"] = (
    final["final_split"].str.strip().str.lower()
)

boxes["split"] = (
    boxes["split"].str.strip().str.lower()
)

if len(final) != 15488:
    raise ValueError(
        f"15488 goruntu bekleniyor: {len(final)}"
    )

if len(boxes) != 42432:
    raise ValueError(
        f"42432 kutu bekleniyor: {len(boxes)}"
    )

if final.duplicated(
    ["original_split", "stem"]
).any():
    raise ValueError(
        "Final split'te tekrarlanan kaynak goruntu var."
    )

if final.duplicated(
    ["final_split", "stem"]
).any():
    raise ValueError(
        "Ayni cikti bolmesinde tekrar dosya adi var."
    )

if not final["final_split"].isin(
    EXPECTED_IMAGES
).all():
    raise ValueError(
        "Bilinmeyen final split etiketi bulundu."
    )

if not final["original_split"].isin(
    ["train", "test"]
).all():
    raise ValueError(
        "Bilinmeyen original split etiketi bulundu."
    )

if not (
    final.loc[
        final["original_split"].eq("test"),
        "final_split"
    ].eq("test")
).all():
    raise ValueError(
        "Resmi test bolmesi degismis."
    )

if not (
    final.loc[
        final["original_split"].eq("train"),
        "final_split"
    ].isin(["train", "val"])
).all():
    raise ValueError(
        "Orijinal train beklenmedik bolmeye atanmis."
    )


# 3. Kutu koordinatlarini kontrol et

coordinate_columns = [
    "xmin", "ymin", "xmax", "ymax"
]

for col in coordinate_columns:
    boxes[col] = pd.to_numeric(
        boxes[col],
        errors="raise"
    )

coordinates = boxes[
    coordinate_columns
].to_numpy(dtype=np.float64)

if not np.isfinite(coordinates).all():
    raise ValueError(
        "Kutu koordinatlarinda sonsuz veya NaN deger var."
    )

x1 = coordinates[:, 0]
y1 = coordinates[:, 1]
x2 = coordinates[:, 2]
y2 = coordinates[:, 3]

valid = (
    (x1 >= 0) &
    (y1 >= 0) &
    (x2 <= IMAGE_WIDTH) &
    (y2 <= IMAGE_HEIGHT) &
    (x2 > x1) &
    (y2 > y1)
)

if not valid.all():
    raise ValueError(
        f"{int((~valid).sum())} gecersiz kutu bulundu."
    )

# Onceki EDA'daki goruntu boyutlariyla tutarlilik
if {"rgb_w", "rgb_h"}.issubset(boxes.columns):

    widths = pd.to_numeric(
        boxes["rgb_w"], errors="raise"
    )

    heights = pd.to_numeric(
        boxes["rgb_h"], errors="raise"
    )

    if not (
        widths.eq(IMAGE_WIDTH).all() and
        heights.eq(IMAGE_HEIGHT).all()
    ):
        raise ValueError(
            "Goruntu boyutlari 1280x1024 ile uyusmuyor."
        )


# 4. YOLO xywh koordinatlarini hesapla

xc = (x1 + x2) / (2 * IMAGE_WIDTH)
yc = (y1 + y2) / (2 * IMAGE_HEIGHT)

bw = (x2 - x1) / IMAGE_WIDTH
bh = (y2 - y1) / IMAGE_HEIGHT

normalized = np.column_stack([
    xc, yc, bw, bh
])

if (
    (normalized < 0).any() or
    (normalized > 1).any()
):
    raise ValueError(
        "YOLO koordinatlari 0-1 araligi disinda."
    )


# 5. Sayisal ters donusum testi

recovered = np.column_stack([
    (xc - bw / 2) * IMAGE_WIDTH,
    (yc - bh / 2) * IMAGE_HEIGHT,
    (xc + bw / 2) * IMAGE_WIDTH,
    (yc + bh / 2) * IMAGE_HEIGHT
])

max_error = float(
    np.abs(coordinates - recovered).max()
)

if max_error > 0.001:
    raise ValueError(
        "YOLO ters donusum hatasi beklenenden buyuk."
    )


# 6. Temiz kutularin envanterde bulunmasini kontrol et

known_keys = set(
    zip(
        final["original_split"],
        final["stem"]
    )
)

box_keys = set(
    zip(
        boxes["split"],
        boxes["stem"]
    )
)

unknown_keys = box_keys - known_keys

if unknown_keys:
    raise ValueError(
        "Envanterde olmayan goruntuye ait kutu bulundu."
    )

# Her kaynak goruntu icin etiket satirlarini hazirla.
label_lookup = {}

for index, row in enumerate(
    boxes.itertuples(index=False)
):

    key = (row.split, row.stem)

    values = normalized[index]

    line = (
        f"{CLASS_ID} "
        f"{values[0]:.{DECIMALS}f} "
        f"{values[1]:.{DECIMALS}f} "
        f"{values[2]:.{DECIMALS}f} "
        f"{values[3]:.{DECIMALS}f}"
    )

    label_lookup.setdefault(
        key, []
    ).append(line)


# 7. Tum goruntuler icin etiket ve manifest hazirla

manifest_rows = []
label_files = []

empty_images = 0

for row in final.itertuples(index=False):

    stem = row.stem
    source_split = row.original_split
    target_split = row.final_split

    key = (source_split, stem)

    labels = label_lookup.get(
        key, []
    )

    expected_count = int(
        row.clean_box_count
    )

    if len(labels) != expected_count:
        raise ValueError(
            f"Kutu sayisi uyusmuyor: "
            f"{source_split}/{stem} | "
            f"{len(labels)} != {expected_count}"
        )

    if not labels:
        empty_images += 1

    rgb_path = (
        ROOT / "visible" /
        source_split / f"{stem}.jpg"
    )

    ir_path = (
        ROOT / "infrared" /
        source_split / f"{stem}.jpg"
    )

    label_path = (
        LABEL_ROOT / target_split /
        f"{stem}.txt"
    )

    if not rgb_path.is_file():
        raise FileNotFoundError(
            f"RGB bulunamadi: {rgb_path}"
        )

    if not ir_path.is_file():
        raise FileNotFoundError(
            f"IR bulunamadi: {ir_path}"
        )

    content = (
        "\n".join(labels) + "\n"
        if labels else ""
    )

    label_files.append(
        (label_path, content)
    )

    manifest_rows.append({
        "stem": stem,
        "original_split": source_split,
        "final_split": target_split,
        "rgb_path": rgb_path.as_posix(),
        "ir_path": ir_path.as_posix(),
        "label_path": label_path.as_posix(),
        "class_id": CLASS_ID,
        "class_name": "person",
        "box_count": len(labels),
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT
    })


# 8. Sayilari yazmadan once dogrula

manifest = pd.DataFrame(
    manifest_rows
)

image_counts = (
    manifest["final_split"]
    .value_counts()
    .to_dict()
)

box_counts = (
    manifest.groupby(
        "final_split"
    )["box_count"]
    .sum()
    .astype(int)
    .to_dict()
)

if image_counts != EXPECTED_IMAGES:
    raise ValueError(
        f"Goruntu sayilari hatali: {image_counts}"
    )

if box_counts != EXPECTED_BOXES:
    raise ValueError(
        f"Kutu sayilari hatali: {box_counts}"
    )

if empty_images != 2:
    raise ValueError(
        f"2 kutusuz goruntu bekleniyor: {empty_images}"
    )


# 9. Dosyalari guvenli bicimde kaydet

# Var olan, farkli icerikli etiketleri koru.
for path, content in label_files:

    if path.exists():

        existing = path.read_text(
            encoding="utf-8"
        )

        if existing != content:
            raise FileExistsError(
                "Farkli icerikli mevcut etiket: "
                f"{path}"
            )

manifest_content = manifest.to_csv(
    index=False,
    lineterminator="\n"
)

if MANIFEST_PATH.exists():

    old = MANIFEST_PATH.read_text(
        encoding="utf-8-sig"
    )

    if old.strip() != manifest_content.strip():
        raise FileExistsError(
            "Mevcut manifest farkli. "
            "Uzerine yazilmadi."
        )

for path, content in label_files:

    if not path.exists():
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        path.write_text(
            content,
            encoding="utf-8"
        )

if not MANIFEST_PATH.exists():

    MANIFEST_PATH.write_text(
        manifest_content,
        encoding="utf-8-sig"
    )


# 10. Kaydedilen TXT dosyalarini yeniden dogrula

verified_boxes = {
    "train": 0,
    "val": 0,
    "test": 0
}

for row in manifest.itertuples(index=False):

    path = Path(
        row.label_path
    )

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    lines = (
        text.splitlines()
        if text else []
    )

    if len(lines) != row.box_count:
        raise ValueError(
            f"Kaydedilen etiket sayisi hatali: {path}"
        )

    for line in lines:

        parts = line.split()

        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(
                f"YOLO etiket bicimi hatali: {path}"
            )

        values = np.array(
            [float(v) for v in parts[1:]]
        )

        if not (
            np.isfinite(values).all() and
            (values >= 0).all() and
            (values <= 1).all() and
            values[2] > 0 and
            values[3] > 0
        ):
            raise ValueError(
                f"Kaydedilen YOLO kutusu gecersiz: {path}"
            )

    verified_boxes[row.final_split] += len(lines)

if verified_boxes != EXPECTED_BOXES:
    raise ValueError(
        "TXT dosyalarindaki kutu toplamlarinda hata."
    )


# 11. JSON ozetini olustur

summary = {
    "total_image_pairs": len(manifest),
    "total_valid_boxes": len(boxes),
    "class_id": CLASS_ID,
    "class_name": "person",
    "image_size": [
        IMAGE_WIDTH,
        IMAGE_HEIGHT
    ],
    "image_pairs_per_split": image_counts,
    "boxes_per_split": verified_boxes,
    "empty_label_files": empty_images,
    "coordinate_roundtrip_max_error_pixels": (
        max_error
    ),
    "label_format": (
        "class_id x_center y_center width height"
    ),
    "coordinates_normalized": True,
    "raw_images_modified": False,
    "notes": [
        "RGB and IR share the same source boxes.",
        "Image files remain in their original folders.",
        "VOC coordinate conventions and RGB-IR "
        "alignment require separate consideration.",
        "Model training has not been performed."
    ]
}

LOG.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


# 12. Terminal sonuclari

print("\n===== ADIM 17A - YOLO ETIKET DONUSUMU =====")

print(
    f"Islenen RGB-IR cifti: {len(manifest)}"
)

print(
    f"Donusturulen insan kutusu: {len(boxes)}"
)

print(
    f"Bos etiket dosyasi: {empty_images}"
)

print("\n===== VERI BOLMELERI =====")

for split in ["train", "val", "test"]:

    print(
        f"{split.upper()}: "
        f"{image_counts[split]} etiket dosyasi | "
        f"{verified_boxes[split]} insan kutusu"
    )

print("\n===== DOGRULAMA KONTROLLERI =====")
print("YOLO koordinatlari 0-1 araliginda: EVET")
print("Tum RGB/IR dosyalari bulundu: EVET")
print("Resmi test bolmesi korundu: EVET")
print("Ham goruntuler degistirilmedi: EVET")

print(
    "Koordinat ters donusum maksimum hatasi: "
    f"{max_error:.10f} piksel"
)

print("\n===== OLUSTURULAN DOSYALAR =====")

print(f"Etiket klasoru: {LABEL_ROOT}")
print(f"Manifest CSV: {MANIFEST_PATH}")
print(f"JSON: {SUMMARY_PATH}")

print(
    "\nNot: YOLO etiketleri olusturuldu. "
    "Gorsel geri cizim kontrolu henuz yapilmadi."
)

print("\nADIM 17A TAMAMLANDI.")
