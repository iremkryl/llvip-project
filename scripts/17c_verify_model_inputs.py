"""
AMAC:
RGB-only, IR-only ve RGB+IR erken fuzyon modelleri icin
ornek girdi tensorlerini ve ortak etiket yapisini dogrulamak.

YONTEM:
1. YOLO manifesti ve nihai veri bolmesi okunur.
2. Train, val ve testten tek ve cok kisili ornekler secilir.
3. RGB ve IR 640x512 boyutuna getirilir.
4. RGB 3, IR 1 ve erken fuzyon 4 kanalli hazirlanir.
5. Piksel degerleri [0,1] araligina olceklenir.
6. Eslesme, kanal, boyut ve etiketler kontrol edilir.
7. Kutusuz iki goruntu ayrica test edilir.
8. Model egitilmez ve goruntu dosyalari degistirilmez.

GIRDI:
data/processed/step17a_label_manifest.csv
data/processed/final_split.csv
data/processed/yolo_labels/
data/raw/LLVIP_extracted/

CIKTI:
outputs/logs/step17c_model_input_summary.json
Terminal: Girdi bicimleri ve dogrulama sonuclari.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROCESSED = Path("data/processed")
LOG = Path("outputs/logs")

LOG.mkdir(parents=True, exist_ok=True)

TARGET_W = 640
TARGET_H = 512

SEED = 42

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


# 1. Manifest ve nihai bolmeyi oku

manifest = pd.read_csv(
    PROCESSED / "step17a_label_manifest.csv",
    dtype={"stem": str}
)

final = pd.read_csv(
    PROCESSED / "final_split.csv",
    dtype={"stem": str}
)

for table in (manifest, final):
    table["stem"] = (
        table["stem"].astype(str).str.zfill(6)
    )

manifest["box_count"] = pd.to_numeric(
    manifest["box_count"],
    errors="raise"
).astype(int)

if len(manifest) != 15488:
    raise ValueError(
        "Manifestte 15488 goruntu cifti bulunmuyor."
    )

if len(final) != 15488:
    raise ValueError(
        "Nihai bolme tablosu eksik."
    )

if manifest.duplicated(
    ["final_split", "stem"]
).any():
    raise ValueError(
        "Manifestte tekrar eden goruntu bulundu."
    )

if final.duplicated(
    ["final_split", "stem"]
).any():
    raise ValueError(
        "Nihai bolmede tekrar eden goruntu bulundu."
    )

manifest_keys = set(
    zip(
        manifest["final_split"],
        manifest["stem"]
    )
)

final_keys = set(
    zip(
        final["final_split"],
        final["stem"]
    )
)

if manifest_keys != final_keys:
    raise ValueError(
        "Manifest ve final split eslesmiyor."
    )

actual_images = (
    manifest["final_split"]
    .value_counts()
    .to_dict()
)

actual_boxes = (
    manifest.groupby("final_split")[
        "box_count"
    ].sum().to_dict()
)

if actual_images != EXPECTED_IMAGES:
    raise ValueError(
        "Bolme goruntu sayilari uyusmuyor."
    )

if actual_boxes != EXPECTED_BOXES:
    raise ValueError(
        "Bolme kutu sayilari uyusmuyor."
    )


# 2. Gercek YOLO etiketlerini oku

def read_labels(path):

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return np.empty(
            (0, 5),
            dtype=np.float32
        )

    rows = []

    for line in text.splitlines():

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Hatali YOLO satiri: {path}"
            )

        values = [float(v) for v in parts]

        if values[0] != 0:
            raise ValueError(
                f"Beklenmeyen sinif: {path}"
            )

        coords = np.asarray(
            values[1:],
            dtype=np.float64
        )

        xc, yc, bw, bh = coords

        if not (
            np.isfinite(coords).all()
            and (coords >= 0).all()
            and (coords <= 1).all()
            and bw > 0
            and bh > 0
            and xc - bw / 2 >= -1e-7
            and yc - bh / 2 >= -1e-7
            and xc + bw / 2 <= 1 + 1e-7
            and yc + bh / 2 <= 1 + 1e-7
        ):
            raise ValueError(
                f"Gecersiz YOLO koordinati: {path}"
            )

        rows.append(values)

    return np.asarray(
        rows,
        dtype=np.float32
    )


# 3. RGB, IR ve fuzyon tensorlerini hazirla

def prepare_inputs(row):

    rgb_path = Path(
        row.rgb_path
    )

    ir_path = Path(
        row.ir_path
    )

    if not rgb_path.is_file():
        raise FileNotFoundError(rgb_path)

    if not ir_path.is_file():
        raise FileNotFoundError(ir_path)

    with Image.open(rgb_path) as im:

        original_rgb_size = im.size

        rgb_array = np.asarray(
            im.convert("RGB").resize(
                (TARGET_W, TARGET_H),
                Image.Resampling.LANCZOS
            ),
            dtype=np.uint8
        )

    with Image.open(ir_path) as im:

        original_ir_size = im.size

        # IR'nin tekrarlanan kanallarini
        # orneklem uzerinde bir kez daha kontrol et.
        original_ir = np.asarray(
            im.convert("RGB"),
            dtype=np.uint8
        )

        if not (
            np.array_equal(
                original_ir[:, :, 0],
                original_ir[:, :, 1]
            )
            and np.array_equal(
                original_ir[:, :, 1],
                original_ir[:, :, 2]
            )
        ):
            raise ValueError(
                f"IR kanallari farkli: {row.stem}"
            )

        ir_array = np.asarray(
            im.convert("L").resize(
                (TARGET_W, TARGET_H),
                Image.Resampling.LANCZOS
            ),
            dtype=np.uint8
        )

    if original_rgb_size != original_ir_size:
        raise ValueError(
            f"RGB/IR boyut uyusmazligi: {row.stem}"
        )

    if original_rgb_size != (1280, 1024):
        raise ValueError(
            f"Beklenmeyen kaynak boyut: {row.stem}"
        )

    # CHW: Channel, Height, Width
    rgb = (
        rgb_array.astype(np.float32)
        .transpose(2, 0, 1) / 255.0
    )

    ir = (
        ir_array.astype(np.float32)[None, :, :]
        / 255.0
    )

    fusion = np.concatenate(
        [rgb, ir],
        axis=0
    )

    return rgb, ir, fusion


# 4. Ornekleri sabit tohumla sec

selected = []

for split in ["train", "val", "test"]:

    subset = manifest[
        manifest["final_split"] == split
    ]

    for description, condition in [
        (
            "Single person",
            subset["box_count"] == 1
        ),
        (
            "Multiple persons",
            subset["box_count"] >= 4
        )
    ]:

        candidates = subset[condition]

        if candidates.empty:
            raise ValueError(
                f"{split} icin uygun ornek yok."
            )

        chosen = candidates.sample(
            n=1,
            random_state=SEED
        ).iloc[0]

        selected.append(
            (chosen, description)
        )


# 5. Kutusuz iki goruntuyu de sec

empty = manifest[
    manifest["box_count"] == 0
].copy()

if set(empty["stem"]) != {
    "100030", "100033"
}:
    raise ValueError(
        "Beklenen kutusuz goruntuler bulunamadi."
    )

for _, row in empty.sort_values(
    "stem"
).iterrows():

    selected.append(
        (row, "Empty annotation")
    )


# 6. Tensor ve etiket kontrolleri

sample_results = []

expected_shapes = {
    "rgb": (3, TARGET_H, TARGET_W),
    "ir": (1, TARGET_H, TARGET_W),
    "fusion": (4, TARGET_H, TARGET_W)
}

print("\n===== ADIM 17C - MODEL GIRDI KONTROLU =====")

print(
    f"Ornek girdi boyutu: "
    f"{TARGET_W}x{TARGET_H}"
)

print(
    "Piksel olcekleme: [0, 255] -> [0, 1]"
)

print("\n===== ORNEK GORUNTULER =====")

for row, description in selected:

    rgb, ir, fusion = prepare_inputs(row)

    tensors = {
        "rgb": rgb,
        "ir": ir,
        "fusion": fusion
    }

    for name, tensor in tensors.items():

        if tensor.shape != expected_shapes[name]:
            raise ValueError(
                f"Tensor boyutu hatali: "
                f"{row.stem} / {name}"
            )

        if tensor.dtype != np.float32:
            raise ValueError(
                f"Beklenmeyen veri tipi: {name}"
            )

        if not (
            np.isfinite(tensor).all()
            and tensor.min() >= 0
            and tensor.max() <= 1
        ):
            raise ValueError(
                f"Piksel araligi hatali: {row.stem}"
            )

    if not (
        np.array_equal(
            fusion[:3], rgb
        )
        and np.array_equal(
            fusion[3:4], ir
        )
    ):
        raise ValueError(
            f"Fuzyon kanal sirasi hatali: {row.stem}"
        )

    labels = read_labels(
        row.label_path
    )

    if len(labels) != int(row.box_count):
        raise ValueError(
            f"Etiket sayisi uyusmuyor: {row.stem}"
        )

    # Goruntu boyutu degisse de normalize
    # xywh kutu koordinatlari ayni kalir.
    # Sadece piksel karsiliklari degisir.
    if len(labels):

        coords = labels[:, 1:]

        pixel_boxes = np.column_stack([
            (coords[:, 0] - coords[:, 2] / 2)
            * TARGET_W,
            (coords[:, 1] - coords[:, 3] / 2)
            * TARGET_H,
            (coords[:, 0] + coords[:, 2] / 2)
            * TARGET_W,
            (coords[:, 1] + coords[:, 3] / 2)
            * TARGET_H
        ])

        if not (
            (pixel_boxes[:, 0] >= -1e-4).all()
            and (pixel_boxes[:, 1] >= -1e-4).all()
            and (
                pixel_boxes[:, 2] <= TARGET_W + 1e-4
            ).all()
            and (
                pixel_boxes[:, 3] <= TARGET_H + 1e-4
            ).all()
        ):
            raise ValueError(
                f"Olceklenen kutu siniri hatali: {row.stem}"
            )

    sample_results.append({
        "stem": row.stem,
        "split": row.final_split,
        "description": description,
        "box_count": int(row.box_count),
        "rgb_shape": list(rgb.shape),
        "ir_shape": list(ir.shape),
        "fusion_shape": list(fusion.shape),
        "labels_shape": list(labels.shape),
        "tensor_dtype": "float32",
        "range_min": float(fusion.min()),
        "range_max": float(fusion.max())
    })

    print(
        f"{row.final_split.upper()} / "
        f"{row.stem} / {description} | "
        f"{len(labels)} kutu | UYGUN"
    )


# 7. Nihai bolmenin SHA256 bilgisini kaydet

split_path = (
    PROCESSED / "final_split.csv"
)

split_sha256 = hashlib.sha256(
    split_path.read_bytes()
).hexdigest()

summary = {
    "total_image_pairs": len(manifest),
    "total_valid_boxes": int(
        manifest["box_count"].sum()
    ),
    "image_pairs_per_split": actual_images,
    "boxes_per_split": actual_boxes,
    "checked_sample_pairs": len(sample_results),
    "original_image_size": [1280, 1024],
    "resized_image_size": [TARGET_W, TARGET_H],
    "channel_order": "CHW",
    "rgb_only_shape": [
        3, TARGET_H, TARGET_W
    ],
    "ir_only_shape": [
        1, TARGET_H, TARGET_W
    ],
    "early_fusion_shape": [
        4, TARGET_H, TARGET_W
    ],
    "pixel_scale": "[0,1]",
    "ir_percentile_normalization": False,
    "final_split_sha256": split_sha256,
    "checked_samples": sample_results,
    "limitations": [
        "These are example NumPy input tensors, "
        "not a complete model training pipeline.",
        "The four-channel early-fusion input "
        "requires an adapted model input layer.",
        "Standard YOLO training configurations "
        "cannot automatically consume this "
        "separate RGB/IR manifest.",
        "No model was trained.",
        "RGB-IR geometric alignment remains "
        "a separate validation question."
    ]
}

output = (
    LOG / "step17c_model_input_summary.json"
)

output.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


# 8. Terminal ozeti

print("\n===== GIRDI TENSORLERI =====")

print(
    f"RGB-only: {expected_shapes['rgb']}"
)

print(
    f"IR-only: {expected_shapes['ir']}"
)

print(
    f"RGB+IR early fusion: "
    f"{expected_shapes['fusion']}"
)

print(
    "Veri tipi: float32"
)

print(
    "IR p1-p99 normalizasyonu: UYGULANMADI"
)

print("\n===== DOGRULAMA KONTROLLERI =====")

print(
    f"Kontrol edilen ornek cift: {len(sample_results)}"
)

print(
    "RGB ve IR boyutlari eslesiyor: EVET"
)

print(
    "IR kanallari orneklerde ozdes: EVET"
)

print(
    "Fuzyon kanal sirasi dogru: EVET"
)

print(
    "Ornek YOLO kutulari tutarli: EVET"
)

print(
    "Kutusuz goruntuler kontrol edildi: EVET"
)

print(
    "Ham goruntuler degistirilmedi: EVET"
)

print(
    "Model egitimi yapildi: HAYIR"
)

print("\n===== OLUSTURULAN DOSYALAR =====")

print(f"JSON: {output}")
print(f"Split SHA256: {split_sha256}")

print("\nADIM 17C TAMAMLANDI.")
