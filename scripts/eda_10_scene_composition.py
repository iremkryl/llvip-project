"""
AMAC:
LLVIP veri setinde sahne gruplarina gore goruntu,
kutu, RGB/IR yogunluk ve dusuk isik dagilimlarini
inceleyerek temsil farkliliklarini sayisallastirmak.

YONTEM:
1. Nihai bolme ve kalite tablolari birlestirilir.
2. Sahne gruplari ve bolmeler kontrol edilir.
3. Her grup icin goruntu, kutu ve yogunluk ozeti alinir.
4. RGB, IR ve kutu sayisi icin eta-kare hesaplanir.
5. Train, validation ve test sahne dagilimi cizilir.
6. Ingilizce grafik ve sayisal rapor kaydedilir.
7. Mevcut veri bolmeleri ve ham dosyalar korunur.

GIRDI:
data/processed/final_split.csv
data/processed/quality_flags.csv

CIKTI:
outputs/logs/eda10_scene_group_metrics.csv
outputs/logs/eda10_scene_composition_summary.json
outputs/figures/fig41_scene_composition_en.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROCESSED = Path("data/processed")
LOG = Path("outputs/logs")
FIG = Path("outputs/figures")

LOG.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# 1. Veri dosyalarini oku

final = pd.read_csv(
    PROCESSED / "final_split.csv",
    dtype=str,
    keep_default_na=False
)

quality = pd.read_csv(
    PROCESSED / "quality_flags.csv",
    dtype=str,
    keep_default_na=False
)

for frame in [final, quality]:
    frame["stem"] = (
        frame["stem"].str.strip().str.zfill(6)
    )

required_final = [
    "stem", "original_split", "final_split",
    "site_group", "validation_group",
    "clean_box_count"
]

required_quality = [
    "stem", "split", "rgb_mean",
    "ir_mean", "flag_low_rgb"
]

for frame, required in [
    (final, required_final),
    (quality, required_quality)
]:
    missing = set(required) - set(frame.columns)

    if missing:
        raise ValueError(
            f"Eksik sutunlar: {sorted(missing)}"
        )


# 2. Kayit ve bolme kontrolleri

if len(final) != 15488 or len(quality) != 15488:
    raise ValueError(
        "15488 goruntu cifti bekleniyor."
    )

if final.duplicated(
    ["original_split", "stem"]
).any():
    raise ValueError(
        "Nihai bolmede tekrar kayit var."
    )

if quality.duplicated(
    ["split", "stem"]
).any():
    raise ValueError(
        "Kalite tablosunda tekrar kayit var."
    )

df = final.merge(
    quality[required_quality],
    left_on=["original_split", "stem"],
    right_on=["split", "stem"],
    how="left",
    validate="one_to_one",
    indicator=True
)

if not df["_merge"].eq("both").all():
    raise ValueError(
        "Kalite ve bolme kayitlari eslesmedi."
    )

df = df.drop(
    columns=["_merge", "split"]
)

for column in [
    "rgb_mean", "ir_mean", "clean_box_count"
]:
    df[column] = pd.to_numeric(
        df[column],
        errors="raise"
    )

if df[
    ["rgb_mean", "ir_mean", "clean_box_count"]
].isna().any().any():
    raise ValueError(
        "Sayisal degerlerde eksiklik var."
    )

df["low_rgb"] = (
    df["flag_low_rgb"]
    .str.lower()
    .isin(["true", "1"])
)

# Orijinal train'deki birlesik gruplar;
# orijinal test'teki mevcut sahne gruplari.
df["analysis_group"] = np.where(
    df["original_split"].eq("train"),
    df["validation_group"],
    df["site_group"]
)

if df["analysis_group"].eq("").any():
    raise ValueError(
        "Bos sahne grubu bulundu."
    )

expected_images = {
    "train": 9937,
    "val": 2088,
    "test": 3463
}

actual_images = (
    df["final_split"].value_counts().to_dict()
)

if actual_images != expected_images:
    raise ValueError(
        f"Bolme sayilari hatali: {actual_images}"
    )

if int(df["clean_box_count"].sum()) != 42432:
    raise ValueError(
        "Temiz kutu toplami 42432 degil."
    )

# Bir sahne grubu train/val arasinda bolunmemeli.
original_train = df[
    df["original_split"] == "train"
]

group_splits = (
    original_train.groupby("analysis_group")[
        "final_split"
    ].nunique()
)

if not group_splits.eq(1).all():
    raise ValueError(
        "Bir sahne grubu train ve val'a bolunmus."
    )


# 3. Grup bazinda istatistikler

metrics = (
    df.groupby(
        ["final_split", "analysis_group"],
        as_index=False
    )
    .agg(
        image_pairs=("stem", "size"),
        person_boxes=("clean_box_count", "sum"),
        mean_rgb=("rgb_mean", "mean"),
        mean_ir=("ir_mean", "mean"),
        low_rgb_pairs=("low_rgb", "sum"),
        boxes_per_image=("clean_box_count", "mean")
    )
)

metrics["low_rgb_pct"] = (
    100
    * metrics["low_rgb_pairs"]
    / metrics["image_pairs"]
)

metrics = metrics.sort_values(
    ["final_split", "analysis_group"]
).reset_index(drop=True)

csv_path = (
    LOG / "eda10_scene_group_metrics.csv"
)

metrics.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)


# 4. Sahne grubuyla iliskili degiskenlik: eta-kare

def eta_squared(frame, target, group_col):

    values = frame[target].to_numpy(
        dtype=np.float64
    )

    overall_mean = float(values.mean())

    total_ss = float(
        np.sum(
            (values - overall_mean) ** 2
        )
    )

    if total_ss <= 0:
        return None

    grouped = frame.groupby(group_col)[target]

    between_ss = float(
        sum(
            len(part)
            * (float(part.mean()) - overall_mean) ** 2
            for _, part in grouped
        )
    )

    return float(
        np.clip(between_ss / total_ss, 0, 1)
    )


eta_results = {}

# Validation, orijinal train'in alt kumesi oldugu
# icin once orijinal train ve testi ayri inceliyoruz.
for source_split in ["train", "test"]:

    subset = df[
        df["original_split"] == source_split
    ]

    eta_results[source_split] = {
        "scene_groups": int(
            subset["analysis_group"].nunique()
        ),
        "rgb_mean_eta2": eta_squared(
            subset, "rgb_mean", "analysis_group"
        ),
        "ir_mean_eta2": eta_squared(
            subset, "ir_mean", "analysis_group"
        ),
        "boxes_per_image_eta2": eta_squared(
            subset,
            "clean_box_count",
            "analysis_group"
        )
    }


# 5. Nihai bolme ozeti

split_summary = {}

for split in ["train", "val", "test"]:

    subset = df[
        df["final_split"] == split
    ]

    split_summary[split] = {
        "image_pairs": int(len(subset)),
        "person_boxes": int(
            subset["clean_box_count"].sum()
        ),
        "scene_groups": int(
            subset["analysis_group"].nunique()
        ),
        "mean_rgb": float(
            subset["rgb_mean"].mean()
        ),
        "mean_ir": float(
            subset["ir_mean"].mean()
        ),
        "boxes_per_image": float(
            subset["clean_box_count"].mean()
        ),
        "low_rgb_pct": float(
            100 * subset["low_rgb"].mean()
        )
    }


# 6. Ingilizce sahne dagilimi grafigi

fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 9),
    sharex=True
)

fig.suptitle(
    "Scene Group Composition by Final Data Split",
    fontsize=15
)

for ax, split in zip(
    axes, ["train", "val", "test"]
):

    part = metrics[
        metrics["final_split"] == split
    ].copy()

    part = part.sort_values(
        "image_pairs",
        ascending=True
    )

    positions = np.arange(len(part))

    bars = ax.barh(
        positions,
        part["image_pairs"]
    )

    ax.set_yticks(
        positions,
        part["analysis_group"]
    )

    ax.set_xlabel(
        "Number of RGB-IR image pairs"
    )

    ax.set_title(
        f"{split.upper()} "
        f"(n={part['image_pairs'].sum():,})"
    )

    ax.grid(
        axis="x",
        alpha=0.2
    )

    ax.set_axisbelow(True)

    ax.bar_label(
        bars,
        padding=3,
        fontsize=9
    )

    ax.set_xlim(
        0,
        max(
            metrics["image_pairs"].max() * 1.2,
            1
        )
    )

axes[0].set_ylabel(
    "Scene group"
)

fig.text(
    0.5,
    0.02,
    "Conservative scene groups; labels are "
    "proxies and do not prove physical scene identity.",
    ha="center",
    fontsize=10
)

fig.tight_layout(
    rect=[0, 0.05, 1, 0.94]
)

fig_path = (
    FIG / "fig41_scene_composition_en.png"
)

fig.savefig(
    fig_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)


# 7. JSON raporu

summary = {
    "total_image_pairs": int(len(df)),
    "total_person_boxes": int(
        df["clean_box_count"].sum()
    ),
    "split_summary": split_summary,
    "scene_variance_eta_squared": eta_results,
    "scene_group_definition": (
        "Conservative merged groups for the "
        "original training data; supplied site "
        "groups for the official test data."
    ),
    "limitations": [
        "Eta-squared is descriptive, not causal.",
        "Scene groups are provisional proxies "
        "for physical recording sites.",
        "Train and validation come from the "
        "same original training population.",
        "Low-RGB flags are based on the "
        "original training brightness threshold."
    ]
}

json_path = (
    LOG / "eda10_scene_composition_summary.json"
)

json_path.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# 8. Terminal sonuclari

print("\n===== EDA 10 - SAHNE DAGILIMI =====")

for split in ["train", "val", "test"]:

    result = split_summary[split]

    print(f"\n{split.upper()}:")
    print(
        f"  Goruntu cifti: {result['image_pairs']}"
    )
    print(
        f"  Sahne grubu: {result['scene_groups']}"
    )
    print(
        f"  Insan kutusu: {result['person_boxes']}"
    )
    print(
        f"  Ortalama kutu: "
        f"{result['boxes_per_image']:.2f}"
    )
    print(
        f"  RGB / IR ortalama: "
        f"{result['mean_rgb']:.2f} / "
        f"{result['mean_ir']:.2f}"
    )
    print(
        f"  Dusuk RGB bayragi: "
        f"%{result['low_rgb_pct']:.2f}"
    )


print("\n===== SAHNE ETKISI - ETA KARE =====")

for split, results in eta_results.items():

    print(f"\nOrijinal {split.upper()}:")

    print(
        f"  Grup sayisi: {results['scene_groups']}"
    )

    for key, title in [
        ("rgb_mean_eta2", "RGB parlakligi"),
        ("ir_mean_eta2", "IR yogunlugu"),
        ("boxes_per_image_eta2", "Kutu sayisi")
    ]:

        value = results[key]

        text = (
            f"{value:.4f}"
            if value is not None
            else "HESAPLANAMADI"
        )

        print(f"  {title}: {text}")


print("\n===== OLUSTURULAN DOSYALAR =====")
print(f"CSV: {csv_path}")
print(f"JSON: {json_path}")
print(f"Gorsel: {fig_path}")

print(
    "\nNot: Eta-kare, sahne grubu ile iliskili "
    "degiskenligi ozetler; nedensellik "
    "veya sifir sizinti kaniti degildir."
)

print("\nEDA 10 TAMAMLANDI.")