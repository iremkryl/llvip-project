"""
EDA 08b - KONTRAST METRIGI DENETIMI

AMAC:
EDA 08'de hesaplanan yerel kontrast degerlerinin
uc durumlarini ve olcum disinda kalan kutulari incelemek.

YONTEM:
1. EDA 08 sonuclari okunur.
2. Sifir ve 0.99 uzeri kontrast sayilari hesaplanir.
3. Kutu ici ve arka plan medyanlari incelenir.
4. Secilen goruntulerdeki beklenen kutu sayilariyla
   gercekte olculen kutu sayilari karsilastirilir.
5. EDA 09'daki kritik orneklerin olcumleri yazdirilir.

GIRDI:
outputs/logs/eda08_object_contrast.csv
outputs/logs/images_table.csv
outputs/logs/boxes_table.csv

CIKTI:
Terminal: Kontrast anomalileri ve eksik olcum raporu.
"""

from pathlib import Path
import pandas as pd

LOG = Path("outputs/logs")

res = pd.read_csv(
    LOG / "eda08_object_contrast.csv",
    dtype={"stem": str}
)

images = pd.read_csv(
    LOG / "images_table.csv",
    dtype={"stem": str}
)

boxes = pd.read_csv(
    LOG / "boxes_table.csv",
    dtype={"stem": str}
)

for df in [res, images, boxes]:
    df["stem"] = df["stem"].str.zfill(6)

# EDA 08 ile ayni kutu filtreleme kurali
valid = (
    (boxes["w"] > 0) &
    (boxes["h"] > 0) &
    (boxes["xmin"] >= 0) &
    (boxes["ymin"] >= 0) &
    (boxes["xmax"] <= boxes["rgb_w"]) &
    (boxes["ymax"] <= boxes["rgb_h"])
)

valid_boxes = boxes.loc[valid].copy()

# Her goruntudeki gecerli kutu sayisi
box_counts = valid_boxes.groupby(
    ["split", "stem"]
).size().to_dict()

print("===== KONTRAST METRIGI DENETIMI =====")

for split in ["train", "test"]:

    sub = res[res["split"] == split].copy()

    print(f"\n===== {split.upper()} =====")
    print(f"Olculen kutu: {len(sub)}")

    for modality in ["rgb", "ir"]:

        c = sub[f"{modality}_contrast"]

        zero = (c == 0).sum()
        high = (c >= 0.99).sum()

        print(f"\n{modality.upper()}")
        print(f"Tam sifir: {zero} (%{zero / len(c) * 100:.2f})")
        print(f"0.99 ve uzeri: {high}")
        print(f"Medyan: {c.median():.4f}")

        inside = sub[f"{modality}_inside"]
        background = sub[f"{modality}_background"]

        print(
            "Ic ve dis medyani esit: "
            f"{(inside == background).sum()}"
        )

        print(
            "Arka plan medyani <= 1: "
            f"{(background <= 1).sum()}"
        )

    equal = (
        (sub["rgb_contrast"] - sub["ir_contrast"])
        .abs() < 1e-10
    ).sum()

    print(f"\nRGB ve IR kontrasti esit: {equal}")

    # EDA 08 ile ayni rastgele goruntuleri sec
    pool = images[
        (images["split"] == split) &
        (images["n_boxes"] > 0)
    ]

    sample = pool.sample(
        n=min(100, len(pool)),
        random_state=42
    )

    expected = set()

    for stem in sample["stem"]:
        n = box_counts.get((split, stem), 0)

        for j in range(n):
            expected.add((stem, j))

    observed = set(
        zip(
            sub["stem"],
            sub["box_index"].astype(int)
        )
    )

    missing = sorted(expected - observed)

    print(f"\nBeklenen gecerli kutu: {len(expected)}")
    print(f"Olculen kutu kimligi: {len(observed)}")
    print(f"Eksik olcum sayisi: {len(missing)}")
    print(f"Eksik kutular: {missing}")

# EDA 09'da dikkat ceken ornekler
print("\n===== KRITIK ORNEKLER =====")

cases = [
    ("010219", 0),
    ("081314", 0),
    ("210320", 2),
    ("260535", 1)
]

for stem, box_index in cases:

    sub = res[
        (res["stem"] == stem) &
        (res["box_index"] == box_index)
    ]

    if sub.empty:
        print(f"\n{stem} / Kutu {box_index}: bulunamadi")
        continue

    r = sub.iloc[0]

    print(f"\n{stem} / Kutu {box_index}")

    for modality in ["rgb", "ir"]:
        print(
            f"{modality.upper()}: "
            f"ic={r[f'{modality}_inside']:.2f}, "
            f"cevre={r[f'{modality}_background']:.2f}, "
            f"kontrast={r[f'{modality}_contrast']:.4f}"
        )

print("\nDenetim tamamlandi.")
