"""
AMAC:
LLVIP icin grup temelli train/validation/test bolmesini
olusturmak ve veri butunlugunu dogrulamak.

YONTEM:
1. 16C grup atamalari ve 16D adaylari okunur.
2. Validation icin 06, 11, 13 ve 16 secilir.
3. Resmi test bolmesi aynen korunur.
4. Her goruntunun tek bir bolmede oldugu kontrol edilir.
5. Train ve validation arasinda grup cakismasi aranir.
6. Goruntu ve kutu sayilari dogrulanir.
7. CSV ve JSON ile bolme bilgileri kaydedilir.
8. Ham goruntuler ve onceki dosyalar degistirilmez.

GIRDI:
outputs/logs/frame_site_groups.csv
outputs/logs/step16c_train_validation_groups.csv
outputs/logs/step16d_validation_candidates.csv
data/processed/quality_flags.csv

CIKTI:
data/processed/final_split.csv
outputs/logs/step16e_final_split_summary.json
Terminal: Bolme sayilari ve dogrulama sonuclari.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd


LOG = Path("outputs/logs")
PROCESSED = Path("data/processed")

LOG.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

VALIDATION_GROUPS = {
    "06", "11", "13", "16"
}

EXPECTED = {
    "train": 9937,
    "val": 2088,
    "test": 3463
}

EXPECTED_BOXES = {
    "train": 28215,
    "val": 5915,
    "test": 8302
}


# 1. Tablolari oku
def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dosya bulunamadi: {path}"
        )

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False
    )


sites = read_csv(
    LOG / "frame_site_groups.csv"
)

train_groups = read_csv(
    LOG / "step16c_train_validation_groups.csv"
)

candidates = read_csv(
    LOG / "step16d_validation_candidates.csv"
)

quality = read_csv(
    PROCESSED / "quality_flags.csv"
)


# 2. Zorunlu sutunlari kontrol et
required = [
    (
        sites,
        ["stem", "split", "prefix",
         "site_group", "assigned_by"]
    ),
    (
        train_groups,
        ["stem", "split", "site_group",
         "validation_group", "clean_box_count"]
    ),
    (
        candidates,
        ["candidate_id", "validation_groups",
         "validation_pairs"]
    ),
    (
        quality,
        ["stem", "split", "clean_box_count",
         "rgb_mean", "ir_mean", "flag_low_rgb"]
    )
]

for table, columns in required:
    missing = set(columns) - set(table.columns)

    if missing:
        raise ValueError(
            f"Eksik sutunlar: {sorted(missing)}"
        )


# 3. Goruntu kimliklerini standartlastir
for table in (sites, train_groups, quality):

    table["stem"] = (
        table["stem"].str.strip().str.zfill(6)
    )

    table["split"] = (
        table["split"].str.strip().str.lower()
    )

if sites.duplicated(
    ["split", "stem"]
).any():
    raise ValueError(
        "Sahne tablosunda tekrar goruntu var."
    )

if train_groups.duplicated(
    ["split", "stem"]
).any():
    raise ValueError(
        "16C tablosunda tekrar goruntu var."
    )

if quality.duplicated(
    ["split", "stem"]
).any():
    raise ValueError(
        "Kalite tablosunda tekrar goruntu var."
    )


# 4. Secimin 16D'nin ilk adayi oldugunu dogrula
best = candidates[
    candidates["candidate_id"] == "1"
]

if len(best) != 1:
    raise ValueError(
        "16D birinci adayi bulunamadi."
    )

best_groups = set(
    g.strip()
    for g in best.iloc[0][
        "validation_groups"
    ].split(",")
)

if best_groups != VALIDATION_GROUPS:
    raise ValueError(
        "16D birinci aday gruplari ile "
        "secili validation gruplari farkli."
    )

if int(
    best.iloc[0]["validation_pairs"]
) != EXPECTED["val"]:
    raise ValueError(
        "16D validation goruntu sayisi degismis."
    )


# 5. Tum goruntu envanterini kontrol et
if len(sites) != 15488:
    raise ValueError(
        f"15488 cift bekleniyor: {len(sites)}"
    )

if len(quality) != 15488:
    raise ValueError(
        "Kalite tablosu eksik veya fazla kayitli."
    )

source_counts = (
    sites["split"].value_counts().to_dict()
)

if source_counts != {
    "train": 12025,
    "test": 3463
}:
    raise ValueError(
        f"Orijinal bolme sayilari hatali: "
        f"{source_counts}"
    )

if len(train_groups) != 12025:
    raise ValueError(
        "16C dosyasi 12025 train kaydi icermeli."
    )

if not train_groups["split"].eq("train").all():
    raise ValueError(
        "16C tablosunda train disi kayit var."
    )


# 6. Train gruplarini envantere ekle
train_metadata = train_groups[
    [
        "stem",
        "split",
        "site_group",
        "validation_group",
        "clean_box_count"
    ]
].rename(
    columns={
        "site_group": "site_group_16c",
        "clean_box_count": "box_count_16c"
    }
)

df = sites.merge(
    train_metadata,
    on=["stem", "split"],
    how="left",
    validate="one_to_one",
    indicator="train_merge"
)

source_train = df["split"].eq("train")
source_test = df["split"].eq("test")

if not df.loc[
    source_train, "train_merge"
].eq("both").all():
    raise ValueError(
        "Bazi train goruntuleri 16C'de eksik."
    )

if not df.loc[
    source_test, "train_merge"
].eq("left_only").all():
    raise ValueError(
        "16C beklenmedik test kayitlari iceriyor."
    )

if not (
    df.loc[source_train, "site_group"]
    ==
    df.loc[source_train, "site_group_16c"]
).all():
    raise ValueError(
        "16C sahne gruplari kaynakla uyusmuyor."
    )

df = df.drop(
    columns=["train_merge", "site_group_16c"]
)


# 7. Temiz kutu ve kalite istatistiklerini ekle
quality_columns = [
    "stem", "split", "clean_box_count",
    "rgb_mean", "ir_mean", "flag_low_rgb"
]

quality_data = quality[
    quality_columns
].rename(
    columns={
        "clean_box_count": "box_count_quality"
    }
)

df = df.merge(
    quality_data,
    on=["stem", "split"],
    how="left",
    validate="one_to_one",
    indicator="quality_merge"
)

if not df["quality_merge"].eq("both").all():
    raise ValueError(
        "Bazi goruntulerin kalite kaydi eksik."
    )

df = df.drop(
    columns=["quality_merge"]
)

df["clean_box_count"] = pd.to_numeric(
    df["box_count_quality"],
    errors="raise"
).astype(int)

train_count_16c = pd.to_numeric(
    df.loc[source_train, "box_count_16c"],
    errors="raise"
).astype(int)

if not train_count_16c.reset_index(
    drop=True
).equals(
    df.loc[source_train, "clean_box_count"]
    .reset_index(drop=True)
):
    raise ValueError(
        "16C ve kalite tablosunun kutu "
        "sayilari uyusmuyor."
    )

if int(
    df["clean_box_count"].sum()
) != 42432:
    raise ValueError(
        "Toplam temiz kutu sayisi 42432 degil."
    )

df["validation_group"] = (
    df["validation_group"].fillna("")
)


# 8. Nihai bolme etiketlerini ata
df["original_split"] = df["split"]

df["final_split"] = "train"

df.loc[
    source_train &
    df["validation_group"].isin(
        VALIDATION_GROUPS
    ),
    "final_split"
] = "val"

df.loc[
    source_test,
    "final_split"
] = "test"


# 9. Gruplarin bolunmedigini dogrula
train_part = df[
    df["original_split"] == "train"
].copy()

group_splits = (
    train_part.groupby(
        "validation_group"
    )["final_split"]
    .nunique()
)

if not group_splits.eq(1).all():
    raise ValueError(
        "Bir validation grubu birden fazla "
        "bolmeye ayrilmis."
    )

actual_val_groups = set(
    train_part.loc[
        train_part["final_split"] == "val",
        "validation_group"
    ]
)

if actual_val_groups != VALIDATION_GROUPS:
    raise ValueError(
        "Validation gruplari beklenenden farkli."
    )

# Grup adi cakismasi kontroludur.
# Fiziksel sahne sizintisinin kesin testi degildir.
train_group_names = set(
    train_part["validation_group"]
)

test_group_names = set(
    df.loc[
        source_test, "site_group"
    ]
)

name_overlap = (
    train_group_names & test_group_names
)

if name_overlap:
    raise ValueError(
        "Train ve test sahne grubu adlari "
        f"cakisiyor: {sorted(name_overlap)}"
    )


# 10. Goruntu ve kutu sayilarini kontrol et
split_counts = (
    df["final_split"]
    .value_counts()
    .to_dict()
)

box_counts = (
    df.groupby("final_split")[
        "clean_box_count"
    ].sum()
    .astype(int)
    .to_dict()
)

if split_counts != EXPECTED:
    raise ValueError(
        f"Goruntu sayilari beklenenden farkli: "
        f"{split_counts}"
    )

if box_counts != EXPECTED_BOXES:
    raise ValueError(
        f"Kutu sayilari beklenenden farkli: "
        f"{box_counts}"
    )

if not df.loc[
    source_test, "final_split"
].eq("test").all():
    raise ValueError(
        "Resmi test bolmesi degistirilmis."
    )


# 11. Nihai CSV tablosunu hazirla
output_columns = [
    "stem",
    "original_split",
    "final_split",
    "prefix",
    "site_group",
    "validation_group",
    "assigned_by",
    "clean_box_count"
]

final = df[
    output_columns
].copy()

final = pd.concat(
    [
        final[
            final["original_split"] == "train"
        ].sort_values("stem"),
        final[
            final["original_split"] == "test"
        ].sort_values("stem")
    ],
    ignore_index=True
)

if final.duplicated(
    ["original_split", "stem"]
).any():
    raise ValueError(
        "Final tabloda tekrar goruntu var."
    )


# 12. Farkli final split ile ustune yazmayi engelle
csv_path = (
    PROCESSED / "final_split.csv"
)

csv_content = final.to_csv(
    index=False,
    lineterminator="\n"
)

if csv_path.exists():

    old_content = csv_path.read_text(
        encoding="utf-8-sig"
    )

    if old_content.strip() != csv_content.strip():
        raise FileExistsError(
            "final_split.csv zaten var ve "
            "icerigi farkli. Otomatik olarak "
            "uzerine yazilmadi."
        )

    print(
        "\nMevcut final_split.csv dogrulandi; "
        "ayni bolme korunuyor."
    )

else:
    csv_path.write_text(
        csv_content,
        encoding="utf-8-sig"
    )


# 13. Dosya ozeti ve SHA256
checksum = hashlib.sha256(
    csv_path.read_bytes()
).hexdigest()

summary = {
    "validation_groups": sorted(
        VALIDATION_GROUPS
    ),
    "validation_ratio_of_original_train": (
        EXPECTED["val"] / 12025
    ),
    "image_pairs": EXPECTED,
    "valid_boxes": EXPECTED_BOXES,
    "total_image_pairs": 15488,
    "total_valid_boxes": 42432,
    "official_test_preserved": True,
    "train_validation_group_overlap": False,
    "test_group_name_overlap": sorted(
        name_overlap
    ),
    "final_split_sha256": checksum,
    "selection_method": (
        "Group-constrained validation selection "
        "using training-set descriptive features."
    ),
    "limitations": [
        "Group labels do not prove the absence "
        "of physical scene overlap.",
        "Validation selection used heuristic "
        "distribution-balance weights.",
        "Original LLVIP test split is preserved.",
        "No model performance was used to "
        "select validation groups."
    ]
}

json_path = (
    LOG / "step16e_final_split_summary.json"
)

json_path.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# 14. Terminal ozeti
print("\n===== ADIM 16E - NIHAI VERI BOLMESI =====")

for name in ["train", "val", "test"]:

    print(
        f"{name.upper()}: "
        f"{split_counts[name]} goruntu cifti | "
        f"{box_counts[name]} insan kutusu"
    )

print(
    f"\nToplam goruntu cifti: "
    f"{sum(split_counts.values())}"
)

print(
    f"Toplam temiz kutu: "
    f"{sum(box_counts.values())}"
)

print("\n===== VALIDATION GRUPLARI =====")

for group in sorted(VALIDATION_GROUPS):

    part = train_part[
        train_part["validation_group"] == group
    ]

    print(
        f"{group}: "
        f"{len(part)} goruntu | "
        f"{int(part['clean_box_count'].sum())} kutu"
    )

print("\n===== DOGRULAMA KONTROLLERI =====")
print("Resmi test korundu: EVET")
print("Train/validation grup cakismasi: YOK")
print("Train/test grup adi cakismasi: YOK")
print("Goruntu kaybi: YOK")
print("Kutu sayisi tutarli: EVET")

print("\n===== OLUSTURULAN DOSYALAR =====")
print(f"CSV: {csv_path}")
print(f"JSON: {json_path}")
print(f"SHA256: {checksum}")

print(
    "\nNot: Bu kontroller sahne grubu "
    "etiketlerini dogrular; tum fiziksel "
    "sahne sizintilarini dislamaz."
)

print("\nADIM 16E TAMAMLANDI.")
