"""
AMAC:
40 goruntu ve 46 kutunun manuel inceleme sonuclarini raporlamak.

YONTEM:
CSV tutarliligi kontrol edilir; gorunurluk, hizalama ve
etiket kararlari ozetlenir. Iki Ingilizce grafik cizilir.

GIRDI:
outputs/logs/manual_audit_40_image_reviews.csv
outputs/logs/manual_audit_40_box_reviews.csv

CIKTI:
outputs/logs/manual_audit_40_summary.json
outputs/logs/manual_audit_40_rgb_nonclear_cases.csv
outputs/logs/manual_audit_40_review_warnings.csv
outputs/figures/fig35_manual_visibility_en.png
outputs/figures/fig36_manual_visibility_by_sampling_en.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LOG = Path("outputs/logs")
FIG = Path("outputs/figures")
FIG.mkdir(parents=True, exist_ok=True)


# 1. Manuel denetim dosyalarini oku

images = pd.read_csv(
    LOG / "manual_audit_40_image_reviews.csv",
    dtype=str,
    keep_default_na=False
)

boxes = pd.read_csv(
    LOG / "manual_audit_40_box_reviews.csv",
    dtype=str,
    keep_default_na=False
)


# 2. Sutun ve kayit butunlugunu kontrol et

required_images = [
    "audit_id",
    "split",
    "stem",
    "selection_group",
    "missing_person",
    "review_status"
]

required_boxes = [
    "audit_id",
    "split",
    "stem",
    "selection_group",
    "box_index",
    "ir_box_correct",
    "rgb_alignment",
    "rgb_visibility",
    "ir_visibility"
]

for name, table, columns in [
    ("image", images, required_images),
    ("box", boxes, required_boxes)
]:
    missing = set(columns) - set(table.columns)

    if missing:
        raise ValueError(
            f"Missing {name} columns: {sorted(missing)}"
        )

    if table[columns].eq("").any().any():
        raise ValueError(
            f"Unfilled required fields in {name} CSV."
        )

if len(images) != 40 or len(boxes) != 46:
    raise ValueError(
        f"Expected 40 images and 46 boxes; "
        f"got {len(images)} and {len(boxes)}."
    )

if not images["review_status"].eq("complete").all():
    raise ValueError(
        "Some image reviews are incomplete."
    )

if images.duplicated(
    ["audit_id", "split", "stem"]
).any():
    raise ValueError(
        "Duplicate image audit entries."
    )

if boxes.duplicated(
    ["audit_id", "split", "stem", "box_index"]
).any():
    raise ValueError(
        "Duplicate box audit entries."
    )


# 3. Goruntu ve kutu kayitlarini eslestir

keys = [
    "audit_id",
    "split",
    "stem",
    "selection_group"
]

merged = boxes.merge(
    images[keys],
    on=keys,
    how="left",
    validate="many_to_one",
    indicator=True
)

if not merged["_merge"].eq("both").all():
    raise ValueError(
        "Some box records do not match the image audit."
    )


# 4. Gorunurluk kategorilerini dogrula

categories = [
    "Clear",
    "Difficult",
    "Not visible"
]

for modality in [
    "rgb_visibility",
    "ir_visibility"
]:
    invalid = (
        set(boxes[modality]) - set(categories)
    )

    if invalid:
        raise ValueError(
            f"Unexpected {modality}: {sorted(invalid)}"
        )


# 5. RGB ve IR gorunurluk sayilari

counts = {
    "RGB": {
        c: int(
            boxes["rgb_visibility"].eq(c).sum()
        )
        for c in categories
    },
    "IR": {
        c: int(
            boxes["ir_visibility"].eq(c).sum()
        )
        for c in categories
    }
}


# 6. Ornekleme gruplarini ayri ozetle

group_order = [
    "random",
    "quality_flagged"
]

groups = {}

for group in group_order:

    part = boxes[
        boxes["selection_group"].eq(group)
    ]

    groups[group] = {
        "images": int(
            images["selection_group"]
            .eq(group)
            .sum()
        ),
        "boxes": int(len(part)),
        "rgb_visibility": {
            c: int(
                part["rgb_visibility"].eq(c).sum()
            )
            for c in categories
        }
    }

if sum(
    g["images"] for g in groups.values()
) != len(images):
    raise ValueError(
        "Unexpected selection groups in image CSV."
    )

if sum(
    g["boxes"] for g in groups.values()
) != len(boxes):
    raise ValueError(
        "Unexpected selection groups in box CSV."
    )


# 7. RGB'de zor gorulen veya gorunmeyen kutular

nonclear = boxes[
    boxes["rgb_visibility"].ne("Clear")
].copy()

nonclear.to_csv(
    LOG / "manual_audit_40_rgb_nonclear_cases.csv",
    index=False,
    encoding="utf-8-sig"
)


# 8. Hizalamasi belirsiz olabilecek kayitlar

warnings = boxes[
    boxes["rgb_visibility"].eq("Not visible")
    &
    boxes["rgb_alignment"].eq("Aligned")
].copy()

warnings["warning"] = (
    "RGB person not visible; alignment cannot "
    "be confidently verified visually."
)

warnings.to_csv(
    LOG / "manual_audit_40_review_warnings.csv",
    index=False,
    encoding="utf-8-sig"
)


# 9. Sayisal ozeti olustur

images_with_nonclear_rgb = int(
    nonclear["audit_id"].nunique()
)

summary = {
    "images_reviewed": int(len(images)),
    "boxes_reviewed": int(len(boxes)),
    "complete_image_reviews": int(
        images["review_status"]
        .eq("complete")
        .sum()
    ),
    "images_with_no_missing_person_reported": int(
        images["missing_person"].eq("No").sum()
    ),
    "ir_boxes_rated_correct": int(
        boxes["ir_box_correct"].eq("Yes").sum()
    ),
    "rgb_boxes_rated_aligned": int(
        boxes["rgb_alignment"].eq("Aligned").sum()
    ),
    "images_with_nonclear_rgb": (
        images_with_nonclear_rgb
    ),
    "visibility_counts": counts,
    "selection_groups": groups,
    "alignment_review_warnings": int(
        len(warnings)
    ),
    "limitation": (
        "Selection is constrained by bounding-box "
        "count and includes targeted cases; "
        "results are descriptive, not representative "
        "of the full LLVIP dataset."
    )
}

summary_path = (
    LOG / "manual_audit_40_summary.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


# 10. FIGURE 35
# Modalitelere gore manuel gorunurluk karsilastirmasi

x = np.arange(len(categories))
width = 0.36

fig, ax = plt.subplots(
    figsize=(10, 6)
)

for offset, modality in [
    (-width / 2, "RGB"),
    (width / 2, "IR")
]:
    values = [
        counts[modality][c]
        for c in categories
    ]

    bars = ax.bar(
        x + offset,
        values,
        width=width,
        label=modality
    )

    ax.bar_label(
        bars,
        padding=3
    )

ax.set_xticks(
    x,
    categories
)

ax.set_ylim(0, 53)

ax.set_ylabel(
    "Annotated person boxes (count)"
)

ax.set_title(
    "Manual Person Visibility Ratings: RGB vs IR"
)

ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

fig.text(
    0.5,
    0.015,
    "Reviewer judgments on 46 boxes from 40 "
    "selected image pairs; not model performance.",
    ha="center",
    fontsize=9
)

fig.tight_layout(
    rect=[0, 0.05, 1, 1]
)

fig.savefig(
    FIG / "fig35_manual_visibility_en.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)


# 11. FIGURE 36
# Ornekleme gruplarina gore RGB gorunurlugu

labels = [
    "Box-count-constrained random (36 boxes)",
    "Quality-flagged (10 boxes)"
]

fig, ax = plt.subplots(
    figsize=(11, 5.7)
)

left = np.zeros(2)

for category in categories:

    values = np.array([
        100
        * groups[group]["rgb_visibility"][category]
        / groups[group]["boxes"]
        for group in group_order
    ])

    ax.barh(
        labels,
        values,
        left=left,
        label=category
    )

    for index, value in enumerate(values):

        if value >= 8:
            ax.text(
                left[index] + value / 2,
                index,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=10
            )

    left += values

ax.set_xlim(0, 100)
ax.invert_yaxis()

ax.set_xlabel(
    "Share of reviewed boxes (%)"
)

ax.set_title(
    "RGB Visibility by Manual Audit Selection Group"
)

ax.legend(
    title="Reviewer rating",
    loc="lower center",
    bbox_to_anchor=(0.5, -0.28),
    ncol=3,
    frameon=False
)

fig.text(
    0.5,
    0.02,
    "Descriptive only: selected image pairs "
    "are not representative of the dataset.",
    ha="center",
    fontsize=9
)

fig.subplots_adjust(
    left=0.32,
    right=0.97,
    top=0.86,
    bottom=0.29
)

fig.savefig(
    FIG / "fig36_manual_visibility_by_sampling_en.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)



# 12. Terminal sonuclari

print("\n===== ADIM 14C - MANUEL DENETIM OZETI =====")
print(f"Incelenen goruntu cifti: {len(images)}")
print(f"Incelenen insan kutusu: {len(boxes)}")
print(
    "Tamamlanan goruntu incelemesi: "
    f"{summary['complete_image_reviews']}"
)

print("\n===== ETIKET VE HIZALAMA KONTROLU =====")
print(
    "Eksik insan etiketi bildirilmeyen goruntu: "
    f"{summary['images_with_no_missing_person_reported']}"
)
print(
    "IR'de dogru olarak degerlendirilen kutu: "
    f"{summary['ir_boxes_rated_correct']}"
)
print(
    "RGB'de hizali olarak degerlendirilen kutu: "
    f"{summary['rgb_boxes_rated_aligned']}"
)
print(
    f"Hizalama inceleme uyarisi: {len(warnings)}"
)

print("\n===== RGB GORUNURLUK DAGILIMI =====")
print(f"Acik gorulebilen: {counts['RGB']['Clear']}")
print(f"Zor gorulebilen: {counts['RGB']['Difficult']}")
print(f"Gorulemeyen: {counts['RGB']['Not visible']}")

print("\n===== IR GORUNURLUK DAGILIMI =====")
print(f"Acik gorulebilen: {counts['IR']['Clear']}")
print(f"Zor gorulebilen: {counts['IR']['Difficult']}")
print(f"Gorulemeyen: {counts['IR']['Not visible']}")

print("\n===== ORNEKLEM DAGILIMI =====")
print(
    "Rastgele secilen goruntu: "
    f"{groups['random']['images']}"
)
print(
    "Rastgele secilen kutu: "
    f"{groups['random']['boxes']}"
)
print(
    "Kalite bayrakli secilen goruntu: "
    f"{groups['quality_flagged']['images']}"
)
print(
    "Kalite bayrakli secilen kutu: "
    f"{groups['quality_flagged']['boxes']}"
)

print(
    "\nRGB'de zor gorulen veya gorulemeyen "
    f"kutu: {len(nonclear)}"
)
print(
    "RGB'de en az bir kisinin zor goruldugu "
    "veya gorulemedigi goruntu: "
    f"{images_with_nonclear_rgb}"
)

print("\n===== OLUSTURULAN DOSYALAR =====")
print(f"JSON: {summary_path}")
print(
    "CSV: "
    f"{LOG / 'manual_audit_40_rgb_nonclear_cases.csv'}"
)
print(
    "CSV: "
    f"{LOG / 'manual_audit_40_review_warnings.csv'}"
)
print(
    "Grafik: "
    f"{FIG / 'fig35_manual_visibility_en.png'}"
)
print(
    "Grafik: "
    f"{FIG / 'fig36_manual_visibility_by_sampling_en.png'}"
)

print("\nADIM 14C TAMAMLANDI.")