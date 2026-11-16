"""
AMAC:
Train bolmesi icin validation gruplarini
korumaci sahne birlestirmeleri ile kesinlestirmek.

YONTEM:
1. Mevcut frame_site_groups dosyasi okunur.
2. Sadece train goruntuleri secilir.
3. Gorsel inceleme ile desteklenen birlestirmeler uygulanir:
   04+07, 08+09, 14+15.
4. Temiz kutu sayisi her goruntuye eklenir.
5. Sonuclar CSV ve JSON olarak kaydedilir.
6. Ingilizce bir ozet grafik uretilir.
7. Official test split degistirilmez.

GIRDI:
outputs/logs/frame_site_groups.csv
data/processed/boxes_cleaned.csv

CIKTI:
outputs/logs/step16c_train_validation_groups.csv
outputs/logs/step16c_train_validation_summary.json
outputs/figures/fig39_train_validation_groups_en.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG = Path("outputs/logs")
FIG = Path("outputs/figures")
DATA = Path("data/processed")

LOG.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

GROUPS_PATH = LOG / "frame_site_groups.csv"
BOXES_PATH = DATA / "boxes_cleaned.csv"

MERGE_MAP = {
    "04": "04_07",
    "07": "04_07",
    "08": "08_09",
    "09": "08_09",
    "14": "14_15",
    "15": "14_15",
}

def normalize_code(x):
    x = str(x).strip()
    if x.isdigit():
        return x.zfill(2)
    return x

def group_sort_key(x):
    x = str(x)

    if "_" in x:
        left = x.split("_")[0]
        if left.isdigit():
            return (int(left), 1, x)

    if x.endswith("D") and x[:-1].isdigit():
        return (int(x[:-1]), 2, x)

    if x.isdigit():
        return (int(x), 0, x)

    return (999, 9, x)

if not GROUPS_PATH.exists():
    raise FileNotFoundError(f"Bulunamadi: {GROUPS_PATH}")

if not BOXES_PATH.exists():
    raise FileNotFoundError(f"Bulunamadi: {BOXES_PATH}")

groups = pd.read_csv(GROUPS_PATH, dtype=str, keep_default_na=False)
boxes = pd.read_csv(BOXES_PATH, dtype={"stem": str, "split": str}, keep_default_na=False)

required_group_cols = ["stem", "split", "prefix", "site_group", "assigned_by"]
required_box_cols = ["stem", "split"]

for col in required_group_cols:
    if col not in groups.columns:
        raise ValueError(f"frame_site_groups.csv icinde eksik sutun: {col}")

for col in required_box_cols:
    if col not in boxes.columns:
        raise ValueError(f"boxes_cleaned.csv icinde eksik sutun: {col}")

groups["stem"] = groups["stem"].astype(str).str.strip()
groups["split"] = groups["split"].astype(str).str.strip().str.lower()
groups["prefix"] = groups["prefix"].astype(str).map(normalize_code)
groups["site_group"] = groups["site_group"].astype(str).map(normalize_code)
groups["assigned_by"] = groups["assigned_by"].astype(str).str.strip()

boxes["stem"] = boxes["stem"].astype(str).str.strip()
boxes["split"] = boxes["split"].astype(str).str.strip().str.lower()

train = groups[groups["split"] == "train"].copy()

if train.empty:
    raise ValueError("Train goruntuleri bulunamadi.")

# Temiz kutu sayisini goruntu bazinda hesapla
box_counts = (
    boxes[boxes["split"] == "train"]
    .groupby("stem")
    .size()
    .rename("clean_box_count")
    .reset_index()
)

train = train.merge(box_counts, on="stem", how="left")
train["clean_box_count"] = train["clean_box_count"].fillna(0).astype(int)

# Validation grouping uygula
train["validation_group"] = train["site_group"].map(MERGE_MAP).fillna(train["site_group"])

# Ozet tablo
summary = (
    train.groupby("validation_group", dropna=False)
    .agg(
        image_count=("stem", "size"),
        box_count=("clean_box_count", "sum"),
        source_site_groups=("site_group", lambda s: ",".join(sorted(set(s), key=group_sort_key))),
        prefixes=("prefix", lambda s: ",".join(sorted(set(s), key=group_sort_key))),
    )
    .reset_index()
)

summary["sort_key"] = summary["validation_group"].map(group_sort_key)
summary = summary.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)

# Cikti tablosu
out_df = train[
    [
        "stem",
        "split",
        "prefix",
        "site_group",
        "validation_group",
        "assigned_by",
        "clean_box_count",
    ]
].copy()

out_df["sort_key"] = out_df["validation_group"].map(group_sort_key)
out_df = out_df.sort_values(["sort_key", "stem"]).drop(columns="sort_key").reset_index(drop=True)

csv_path = LOG / "step16c_train_validation_groups.csv"
out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# JSON ozet
report = {
    "train_image_count": int(len(train)),
    "validation_group_count": int(len(summary)),
    "official_test_split_changed": False,
    "manual_scene_merges": {
        "04_07": ["04", "07"],
        "08_09": ["08", "09"],
        "14_15": ["14", "15"],
    },
    "notes": [
        "05 ve 22 gruplari birlestirilmedi.",
        "Bu cikti sadece train icindeki validation tasarimi icindir.",
        "Official test split degistirilmemistir.",
        "08 grubunun ic heterojenligi olasi olsa da, 08 ve 09 korumaci olarak ayni validation grubuna alinmistir.",
    ],
    "group_summary": summary.to_dict(orient="records"),
}

json_path = LOG / "step16c_train_validation_summary.json"
json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

# Grafik (INGILIZCE)
plot_df = summary.copy()

fig, ax = plt.subplots(figsize=(11, 7))
y = np.arange(len(plot_df))

bars = ax.barh(y, plot_df["image_count"])

ax.set_yticks(y)
ax.set_yticklabels(plot_df["validation_group"])
ax.invert_yaxis()
ax.set_xlabel("Number of training image pairs")
ax.set_ylabel("Validation group")
ax.set_title("Training Validation Groups After Conservative Scene Merging")

max_val = max(plot_df["image_count"].max(), 1)

for yi, img_n, box_n in zip(y, plot_df["image_count"], plot_df["box_count"]):
    ax.text(
        img_n + max_val * 0.01,
        yi,
        f"{img_n} images | {box_n} boxes",
        va="center",
        fontsize=10,
    )

fig.text(
    0.5,
    0.01,
    "Official test split is unchanged; this grouping is only for validation design.",
    ha="center",
    fontsize=10,
)

plt.tight_layout(rect=[0, 0.03, 1, 1])

fig_path = FIG / "fig39_train_validation_groups_en.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()

# Terminal ozeti
print("\n===== ADIM 16C - VALIDATION GRUPLARI =====")
print(f"Train goruntu sayisi: {len(train)}")
print(f"Nihai validation grup sayisi: {len(summary)}")

print("\n===== UYGULANAN KORUMACI BIRLESTIRMELER =====")
print("04 + 07 -> 04_07")
print("08 + 09 -> 08_09")
print("14 + 15 -> 14_15")
print("05 ve 22 AYRI birakildi.")

print("\n===== GRUP OZETI =====")
for row in summary.itertuples(index=False):
    print(
        f"{row.validation_group}: "
        f"{row.image_count} goruntu | "
        f"{row.box_count} kutu | "
        f"Kaynak gruplar: {row.source_site_groups}"
    )

print("\n===== OLUSTURULAN DOSYALAR =====")
print(f"CSV: {csv_path}")
print(f"JSON: {json_path}")
print(f"Gorsel: {fig_path}")

print("\nNot: Bu adim official test split'i degistirmez.")
print("Validation tasarimi icin train tarafindaki grup yapisini dondurur.")

print("\nADIM 16C TAMAMLANDI.")