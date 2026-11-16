
"""
ADIM 15D - VERI KAYBI VE KALITE BAYRAKLARI

AMAC:
Temizleme sonrasinda korunan goruntu ve anotasyon
sayilarini dogrulamak, inceleme gerektiren kayitlari
kalite bayraklariyla isaretlemek.

YONTEM:
1. Ham ve temiz anotasyon tablolari okunur.
2. Her goruntu icin once/sonra kutu sayisi hesaplanir.
3. RGB/IR bayt tekrarlari isaretlenir.
4. Kutusuz ve anotasyonu temizlenmis goruntuler belirlenir.
5. Train yuzdeliklerine dayali parlaklik/yogunluk
   inceleme bayraklari olusturulur.
6. Veri kaybi ve bayrak istatistikleri kaydedilir.
7. Ham veriler degistirilmez, goruntu silinmez.

GIRDI:
outputs/logs/images_table.csv
outputs/logs/quality_pass.csv
outputs/logs/file_hashes.csv
data/processed/boxes_cleaned.csv
data/processed/removed_invalid_boxes.csv

CIKTI:
data/processed/quality_flags.csv
outputs/logs/step15d_quality_summary.json
Terminal: Veri kaybi ve kalite bayragi ozeti.
"""

from pathlib import Path
import json
import pandas as pd

LOG = Path("outputs/logs")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

images = pd.read_csv(
    LOG / "images_table.csv",
    dtype={"stem": str, "prefix": str}
)

quality = pd.read_csv(
    LOG / "quality_pass.csv",
    dtype={"stem": str}
)

hashes = pd.read_csv(
    LOG / "file_hashes.csv",
    dtype={"stem": str}
)

clean = pd.read_csv(
    PROCESSED / "boxes_cleaned.csv",
    dtype={"stem": str}
)

removed = pd.read_csv(
    PROCESSED / "removed_invalid_boxes.csv",
    dtype={"stem": str}
)

# 1. Goruntu envanterini kalite ve hash bilgileriyle birlestir
df = images[
    ["stem", "split", "prefix", "n_boxes"]
].rename(
    columns={"n_boxes": "raw_box_count"}
).copy()

df = df.merge(
    quality[
        ["stem", "split", "rgb_ok", "ir_ok",
         "rgb_mean", "ir_mean"]
    ],
    on=["stem", "split"],
    how="left",
    validate="one_to_one"
)

df = df.merge(
    hashes[
        ["stem", "split", "rgb_sha", "ir_sha"]
    ],
    on=["stem", "split"],
    how="left",
    validate="one_to_one"
)

if df[["rgb_mean", "ir_mean",
       "rgb_sha", "ir_sha"]].isna().any().any():
    raise ValueError(
        "Goruntu, kalite veya hash tablolarinda eksik eslesme."
    )

# 2. Goruntu basina temiz ve cikarilan kutu sayilari
def attach_counts(frame, table, column):
    counts = (
        table.groupby(["stem", "split"])
        .size()
        .rename(column)
        .reset_index()
    )

    return frame.merge(
        counts,
        on=["stem", "split"],
        how="left",
        validate="one_to_one"
    )


df = attach_counts(df, clean, "clean_box_count")
df = attach_counts(df, removed, "removed_box_count")

for column in ["clean_box_count", "removed_box_count"]:
    df[column] = df[column].fillna(0).astype(int)

# Envanter tutarliligi
assert (
    df["raw_box_count"]
    == df["clean_box_count"] + df["removed_box_count"]
).all(), "Ham ve temiz anotasyon sayilari uyusmuyor."

assert len(df) == len(images), (
    "Goruntu kaydi sayisi degisti."
)

# 3. Kalite ve inceleme bayraklari
df["flag_no_boxes"] = df["clean_box_count"] == 0

df["flag_invalid_box_removed"] = (
    df["removed_box_count"] > 0
)

df["flag_duplicate_rgb"] = (
    df.groupby("rgb_sha")["stem"].transform("size") > 1
)

df["flag_duplicate_ir"] = (
    df.groupby("ir_sha")["stem"].transform("size") > 1
)

rgb_ok = df["rgb_ok"].astype(str).str.lower().eq("true")
ir_ok = df["ir_ok"].astype(str).str.lower().eq("true")

df["flag_decode_problem"] = ~(rgb_ok & ir_ok)

# Esikler yalnizca train dagilimindan hesaplanir.
train = df[df["split"] == "train"]

rgb_p10 = float(train["rgb_mean"].quantile(0.10))
ir_p05 = float(train["ir_mean"].quantile(0.05))
ir_p95 = float(train["ir_mean"].quantile(0.95))

df["flag_low_rgb"] = df["rgb_mean"] < rgb_p10

df["flag_ir_intensity_tail"] = (
    (df["ir_mean"] < ir_p05) |
    (df["ir_mean"] > ir_p95)
)

flag_columns = [
    "flag_no_boxes",
    "flag_invalid_box_removed",
    "flag_duplicate_rgb",
    "flag_duplicate_ir",
    "flag_decode_problem",
    "flag_low_rgb",
    "flag_ir_intensity_tail"
]

df["flag_any_review"] = df[flag_columns].any(axis=1)

# 4. Rapor
summary = {
    "raw_image_pairs": int(len(images)),
    "retained_image_pairs": int(len(df)),
    "removed_image_pairs": 0,
    "raw_boxes": int(images["n_boxes"].sum()),
    "retained_boxes": int(len(clean)),
    "removed_boxes": int(len(removed)),
    "quality_flag_counts": {
        col: int(df[col].sum())
        for col in flag_columns
    },
    "review_candidate_images": int(
        df["flag_any_review"].sum()
    ),
    "thresholds": {
        "rgb_train_p10": rgb_p10,
        "ir_train_p05": ir_p05,
        "ir_train_p95": ir_p95
    }
}

csv_path = PROCESSED / "quality_flags.csv"
json_path = LOG / "step15d_quality_summary.json"

df.to_csv(csv_path, index=False)

json_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\n===== ADIM 15D - VERI OZETI =====")
print(f"Ham goruntu cifti: {len(images)}")
print(f"Korunan goruntu cifti: {len(df)}")
print(f"Ham kutu: {images['n_boxes'].sum()}")
print(f"Temiz kutu: {len(clean)}")
print(f"Cikarilan kutu: {len(removed)}")

print("\n===== BAYRAKLAR =====")
for col in flag_columns:
    print(f"{col}: {df[col].sum()}")

print(
    f"\nEn az bir inceleme bayragi: "
    f"{df['flag_any_review'].sum()}"
)

print("\n===== INCELEME ESIKLERI =====")
print(f"Train RGB P10: {rgb_p10:.2f}")
print(f"Train IR P05: {ir_p05:.2f}")
print(f"Train IR P95: {ir_p95:.2f}")

print(f"\nCSV: {csv_path}")
print(f"JSON: {json_path}")
print("\nADIM 15D TAMAMLANDI.")