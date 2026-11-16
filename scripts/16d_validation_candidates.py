"""
AMAC:
Train verisinden sahne gruplarini bolmeden,
dengeli validation kumesi adaylarini belirlemek.

YONTEM:
1. 16C sahne gruplari ve kalite verileri birlestirilir.
2. Train istatistikleri ve grup ozellikleri hesaplanir.
3. 15-20% validation orani hedeflenir.
4. 3-6 gruptan olusan kombinasyonlar taranir.
5. RGB, IR, kutu ve dusuk isik dagilimlari
   bakimindan adaylar puanlanir.
6. En iyi 10 aday ayri bir CSV'ye kaydedilir.
7. Test verisi kullanilmaz ve final split olusturulmaz.

GIRDI:
outputs/logs/step16c_train_validation_groups.csv
data/processed/quality_flags.csv

CIKTI:
outputs/logs/step16d_validation_candidates.csv
outputs/logs/step16d_group_metrics.csv
Terminal: En iyi validation adaylari ve istatistikleri.
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


LOG = Path("outputs/logs")
DATA = Path("data/processed")

TARGET_RATIO = 0.175
MIN_RATIO = 0.15
MAX_RATIO = 0.20

MIN_GROUPS = 3
MAX_GROUPS = 6
TOP_K = 10

# Gozlem sayisi az olan 25D simdilik
# modelin egitim tarafinda korunuyor.
PROTECTED_TRAIN_GROUPS = {"25D"}


# 1. Verileri oku

groups = pd.read_csv(
    LOG / "step16c_train_validation_groups.csv",
    dtype={"stem": str, "split": str,
           "validation_group": str}
)

quality = pd.read_csv(
    DATA / "quality_flags.csv",
    dtype={"stem": str, "split": str}
)

for frame in (groups, quality):
    frame["stem"] = frame["stem"].str.zfill(6)
    frame["split"] = (
        frame["split"].str.strip().str.lower()
    )

if not groups["split"].eq("train").all():
    raise ValueError(
        "16C dosyasi yalnizca train icermeli."
    )

if groups.duplicated(["split", "stem"]).any():
    raise ValueError(
        "16C tablosunda tekrar goruntu bulundu."
    )

if quality.duplicated(["split", "stem"]).any():
    raise ValueError(
        "Kalite tablosunda tekrar goruntu bulundu."
    )

needed = [
    "stem", "split", "rgb_mean",
    "ir_mean", "flag_low_rgb"
]

missing = set(needed) - set(quality.columns)

if missing:
    raise ValueError(
        f"Kalite tablosunda eksik sutun: {missing}"
    )

df = groups.merge(
    quality[needed],
    on=["stem", "split"],
    how="left",
    validate="one_to_one",
    indicator=True
)

if not df["_merge"].eq("both").all():
    raise ValueError(
        "16C ve kalite tablolarinda eksik eslesme."
    )

df = df.drop(columns="_merge")

for col in [
    "rgb_mean", "ir_mean", "clean_box_count"
]:
    df[col] = pd.to_numeric(
        df[col], errors="raise"
    )

if df[
    ["rgb_mean", "ir_mean", "clean_box_count"]
].isna().any().any():
    raise ValueError(
        "Sayisal ozelliklerde eksik deger bulundu."
    )

df["low_rgb"] = (
    df["flag_low_rgb"]
    .astype(str)
    .str.lower()
    .isin(["true", "1"])
)

if len(df) != 12025:
    raise ValueError(
        f"12025 train cifti bekleniyor: {len(df)}"
    )

if int(df["clean_box_count"].sum()) != 34130:
    raise ValueError(
        "Train kutu sayisi 34130 ile uyusmuyor."
    )


# 2. Train genel dagilimini hesapla

total_n = len(df)

overall_rgb = float(df["rgb_mean"].mean())
overall_ir = float(df["ir_mean"].mean())
overall_boxes = float(
    df["clean_box_count"].mean()
)
overall_low = float(df["low_rgb"].mean())

rgb_std = max(
    float(df["rgb_mean"].std()), 1.0
)

ir_std = max(
    float(df["ir_mean"].std()), 1.0
)

# Ortak RGB sinirlari tum train uzerinden belirlenir.
rgb_limits = np.quantile(
    df["rgb_mean"],
    [0.10, 0.50, 0.90]
)

df["rgb_bin"] = np.digitize(
    df["rgb_mean"].to_numpy(),
    rgb_limits
)

# 0, 1, 2, 3, 4 veya daha fazla kutu.
df["box_bin"] = np.minimum(
    df["clean_box_count"].astype(int),
    4
)

overall_rgb_hist = (
    np.bincount(
        df["rgb_bin"], minlength=4
    ) / total_n
)

overall_box_hist = (
    np.bincount(
        df["box_bin"], minlength=5
    ) / total_n
)


# 3. Grup bazinda sayisal ozetler

stats = {}

for group, part in df.groupby(
    "validation_group", sort=True
):
    stats[group] = {
        "n": int(len(part)),
        "box_sum": int(
            part["clean_box_count"].sum()
        ),
        "rgb_sum": float(
            part["rgb_mean"].sum()
        ),
        "ir_sum": float(
            part["ir_mean"].sum()
        ),
        "low_sum": int(
            part["low_rgb"].sum()
        ),
        "rgb_hist": np.bincount(
            part["rgb_bin"],
            minlength=4
        ),
        "box_hist": np.bincount(
            part["box_bin"],
            minlength=5
        )
    }

group_rows = []

for group, values in stats.items():
    n = values["n"]

    group_rows.append({
        "validation_group": group,
        "image_pairs": n,
        "person_boxes": values["box_sum"],
        "boxes_per_image": values["box_sum"] / n,
        "rgb_mean": values["rgb_sum"] / n,
        "ir_mean": values["ir_sum"] / n,
        "low_rgb_pct": 100 * values["low_sum"] / n,
        "image_share_pct": 100 * n / total_n,
        "protected_train": (
            group in PROTECTED_TRAIN_GROUPS
        )
    })

group_metrics = pd.DataFrame(group_rows)

group_path = (
    LOG / "step16d_group_metrics.csv"
)

group_metrics.to_csv(
    group_path,
    index=False,
    encoding="utf-8-sig"
)


# 4. Aday kombinasyonlarini tara

available = sorted(
    set(stats) - PROTECTED_TRAIN_GROUPS
)

candidates = []
tested = 0

for group_count in range(
    MIN_GROUPS, MAX_GROUPS + 1
):

    for selected in combinations(
        available, group_count
    ):

        n_val = sum(
            stats[g]["n"] for g in selected
        )

        val_ratio = n_val / total_n

        if not (
            MIN_RATIO <= val_ratio <= MAX_RATIO
        ):
            continue

        tested += 1

        box_sum = sum(
            stats[g]["box_sum"] for g in selected
        )

        rgb_sum = sum(
            stats[g]["rgb_sum"] for g in selected
        )

        ir_sum = sum(
            stats[g]["ir_sum"] for g in selected
        )

        low_sum = sum(
            stats[g]["low_sum"] for g in selected
        )

        rgb_hist = sum(
            (stats[g]["rgb_hist"] for g in selected),
            np.zeros(4, dtype=np.int64)
        )

        box_hist = sum(
            (stats[g]["box_hist"] for g in selected),
            np.zeros(5, dtype=np.int64)
        )

        val_rgb_mean = rgb_sum / n_val
        val_ir_mean = ir_sum / n_val
        val_box_mean = box_sum / n_val
        val_low_ratio = low_sum / n_val

        val_rgb_hist = rgb_hist / n_val
        val_box_hist = box_hist / n_val

        # Basit, aciklanabilir denge cezasi.
        # Kucuk skor daha iyi dengeyi gosterir.
        penalties = {
            "ratio": (
                abs(val_ratio - TARGET_RATIO)
                / (MAX_RATIO - TARGET_RATIO)
            ),
            "rgb_mean": (
                abs(val_rgb_mean - overall_rgb)
                / rgb_std
            ),
            "ir_mean": (
                abs(val_ir_mean - overall_ir)
                / ir_std
            ),
            "box_mean": (
                abs(val_box_mean - overall_boxes)
                / max(overall_boxes, 1)
            ),
            "low_rgb": (
                abs(val_low_ratio - overall_low)
                / max(overall_low, 0.05)
            ),
            "rgb_hist": (
                0.5 * np.abs(
                    val_rgb_hist - overall_rgb_hist
                ).sum()
            ),
            "box_hist": (
                0.5 * np.abs(
                    val_box_hist - overall_box_hist
                ).sum()
            )
        }

        # Bunlar model basarisi degil,
        # validation tasarimi icin sezgisel agirliklar.
        score = (
            0.9 * penalties["ratio"]
            + 1.2 * penalties["rgb_mean"]
            + 1.0 * penalties["ir_mean"]
            + 1.0 * penalties["box_mean"]
            + 0.5 * penalties["low_rgb"]
            + 1.0 * penalties["rgb_hist"]
            + 1.0 * penalties["box_hist"]
        )

        candidates.append({
            "validation_groups": ",".join(selected),
            "group_count": len(selected),
            "validation_pairs": n_val,
            "validation_pct": 100 * val_ratio,
            "validation_boxes": box_sum,
            "validation_boxes_per_image": val_box_mean,
            "validation_rgb_mean": val_rgb_mean,
            "validation_ir_mean": val_ir_mean,
            "validation_low_rgb_pct": (
                100 * val_low_ratio
            ),
            "includes_14_15": (
                "14_15" in selected
            ),
            "balance_score": float(score)
        })


if not candidates:
    raise ValueError(
        "Belirtilen oranlarda uygun grup "
        "kombinasyonu bulunamadi."
    )


# 5. En iyi 10 adayi kaydet

ranking = pd.DataFrame(candidates)

ranking = ranking.sort_values(
    ["balance_score", "validation_pct",
     "validation_groups"]
).reset_index(drop=True)

top = ranking.head(TOP_K).copy()

top.insert(
    0, "candidate_id",
    range(1, len(top) + 1)
)

out_path = (
    LOG / "step16d_validation_candidates.csv"
)

top.to_csv(
    out_path,
    index=False,
    encoding="utf-8-sig"
)


# 6. Terminal sonuclari

print("\n===== ADIM 16D - VALIDATION ADAYLARI =====")
print(f"Toplam train goruntu cifti: {total_n}")
print(f"Toplam sahne grubu: {len(stats)}")
print(
    f"Hedef validation orani: %{TARGET_RATIO * 100:.1f}"
)
print(
    f"Izin verilen aralik: "
    f"%{MIN_RATIO * 100:.0f} - "
    f"%{MAX_RATIO * 100:.0f}"
)
print(
    f"Uygun kombinasyon sayisi: {tested}"
)

print("\n===== EN IYI 10 ADAY =====")

for row in top.itertuples(index=False):
    print(
        f"\nAday {row.candidate_id}: "
        f"{row.validation_groups}"
    )

    print(
        f"  Validation: {row.validation_pairs} cift "
        f"(%{row.validation_pct:.2f})"
    )

    print(
        f"  Ortalama kutu: "
        f"{row.validation_boxes_per_image:.2f}"
    )

    print(
        f"  RGB / IR ortalama: "
        f"{row.validation_rgb_mean:.2f} / "
        f"{row.validation_ir_mean:.2f}"
    )

    print(
        f"  Dusuk RGB orani: "
        f"%{row.validation_low_rgb_pct:.2f}"
    )

    print(
        f"  Denge skoru: "
        f"{row.balance_score:.4f}"
    )

print("\n===== TRAIN GENEL REFERANS =====")
print(f"Ortalama kutu: {overall_boxes:.2f}")
print(
    f"RGB / IR ortalama: "
    f"{overall_rgb:.2f} / {overall_ir:.2f}"
)
print(
    f"Dusuk RGB orani: %{overall_low * 100:.2f}"
)

print("\n===== OLUSTURULAN DOSYALAR =====")
print(f"CSV: {out_path}")
print(f"CSV: {group_path}")

print(
    "\nNot: Bunlar aday kombinasyonlardir. "
    "Final split henuz olusturulmadi."
)

print("\nADIM 16D TAMAMLANDI.")