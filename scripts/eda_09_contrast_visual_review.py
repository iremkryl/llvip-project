"""
EDA 09 - RGB/IR KONTRAST SONUCLARININ GORSEL INCELENMESI

AMAC:
EDA 08'de hesaplanan RGB ve IR yerel kontrast degerlerini
gercek goruntuler uzerinde incelemek. Insan kutularinin
gorunurlugunu, cevre kosullarini ve olasi etiket/hizalama
sorunlarini nitel olarak degerlendirmek.

YONTEM:
1. EDA 08 kontrast sonuclari okunur.
2. Train ve test icin dort farkli durum secilir:
   - IR kontrast avantaji
   - RGB kontrast avantaji
   - Benzer kontrast
   - Iki modalitede dusuk kontrast
3. Ayni goruntunun tekrar secilmesi mumkun oldugunca onlenir.
4. RGB ve IR goruntuleri ortak kutu koordinatlariyla acilir.
5. Hedef kutu ve %25 genisletilmis cevre isaretlenir.
6. Tam goruntuler ve yakinlastirilmis bolgeler gosterilir.
7. Secim bilgileri CSV, gorseller PNG olarak kaydedilir.

GIRDI:
outputs/logs/eda08_object_contrast.csv
data/raw/LLVIP_extracted/visible/
data/raw/LLVIP_extracted/infrared/

CIKTI:
outputs/figures/fig29_contrast_cases_train.png
outputs/figures/fig30_contrast_cases_test.png
outputs/logs/eda09_selected_cases.csv
Terminal: Secilen orneklerin kontrast sonuclari.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path("data/raw/LLVIP_extracted")
LOG = Path("outputs/logs")
FIG = Path("outputs/figures")

FIG.mkdir(parents=True, exist_ok=True)

# 1. EDA 08 sonuclarini oku
csv_path = LOG / "eda08_object_contrast.csv"

df = pd.read_csv(
    csv_path,
    dtype={"stem": str}
)

required = [
    "stem", "split", "box_index",
    "xmin", "ymin", "xmax", "ymax",
    "rgb_contrast", "ir_contrast"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        "EDA 08 CSV dosyasinda eksik sutunlar: "
        + ", ".join(missing)
        + "\nOnce guncel EDA 08 betigini tekrar calistir."
    )

df["stem"] = df["stem"].str.zfill(6)

numeric_cols = [
    "xmin", "ymin", "xmax", "ymax",
    "rgb_contrast", "ir_contrast"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="raise")

df = df.dropna(subset=numeric_cols).copy()

# Pozitif fark: IR kontrasti daha yuksek.
df["difference"] = (
    df["ir_contrast"] - df["rgb_contrast"]
)

df["abs_difference"] = df["difference"].abs()
df["max_contrast"] = df[
    ["rgb_contrast", "ir_contrast"]
].max(axis=1)


# 2. Dort farkli durum icin ornek sec
def select_cases(subset):
    selected = []
    used_stems = set()

    scenarios = [
        (
            "IR avantaji",
            subset[subset["difference"] > 0].sort_values(
                "difference", ascending=False
            )
        ),
        (
            "RGB avantaji",
            subset[subset["difference"] < 0].sort_values(
                "difference", ascending=True
            )
        ),
        (
            "Benzer kontrast",
            subset[
                (subset["rgb_contrast"] >= 0.05) &
                (subset["ir_contrast"] >= 0.05)
            ].sort_values("abs_difference")
        ),
        (
            "Ikisi de dusuk",
            subset.sort_values("max_contrast")
        )
    ]

    for label, candidates in scenarios:
        chosen = None

        for _, row in candidates.iterrows():
            if row["stem"] not in used_stems:
                chosen = row.copy()
                break

        if chosen is None:
            print(f"Uygun ornek bulunamadi: {label}")
            continue

        used_stems.add(chosen["stem"])
        chosen["scenario"] = label
        selected.append(chosen)

    return selected


# 3. Goruntu dosyalarini oku
def read_pair(split, stem):
    rgb_path = (
        ROOT / "visible" / split / f"{stem}.jpg"
    )
    ir_path = (
        ROOT / "infrared" / split / f"{stem}.jpg"
    )

    if not rgb_path.exists() or not ir_path.exists():
        raise FileNotFoundError(
            f"RGB veya IR goruntusu bulunamadi: {stem}"
        )

    with Image.open(rgb_path) as img:
        rgb = np.asarray(img.convert("RGB"))

    with Image.open(ir_path) as img:
        ir = np.asarray(img.convert("L"))

    if rgb.shape[:2] != ir.shape:
        raise ValueError(
            f"RGB ve IR boyutlari uyusmuyor: {stem}"
        )

    return rgb, ir


# 4. Kutu ve cevre koordinatlari
def get_regions(row, image_shape):
    height, width = image_shape[:2]

    x1 = max(0, int(np.floor(row["xmin"])))
    y1 = max(0, int(np.floor(row["ymin"])))
    x2 = min(width, int(np.ceil(row["xmax"])))
    y2 = min(height, int(np.ceil(row["ymax"])))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Gecersiz kutu koordinatlari.")

    pad_x = max(8, int((x2 - x1) * 0.25))
    pad_y = max(8, int((y2 - y1) * 0.25))

    a = max(0, x1 - pad_x)
    b = max(0, y1 - pad_y)
    c = min(width, x2 + pad_x)
    d = min(height, y2 + pad_y)

    return (x1, y1, x2, y2), (a, b, c, d)


def draw_box(ax, box, offset=(0, 0),
             color="lime", linestyle="-"):

    x1, y1, x2, y2 = box
    ox, oy = offset

    ax.add_patch(
        Rectangle(
            (x1 - ox, y1 - oy),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            linestyle=linestyle
        )
    )


# 5. Gorsel panel olustur
def create_figure(split, cases):
    fig, axes = plt.subplots(
        4, 4, figsize=(19, 14)
    )

    for j in range(4):

        if j >= len(cases):
            for i in range(4):
                axes[i, j].axis("off")
            continue

        row = cases[j]
        stem = row["stem"]

        rgb, ir = read_pair(split, stem)

        box, region = get_regions(
            row, rgb.shape
        )

        a, b, c, d = region

        images = [
            (rgb, "RGB - Tam goruntu", False),
            (ir, "IR - Tam goruntu", False),
            (rgb[b:d, a:c], "RGB - Yakin", True),
            (ir[b:d, a:c], "IR - Yakin", True)
        ]

        for i, (img, label, cropped) in enumerate(images):

            ax = axes[i, j]

            if img.ndim == 2:
                ax.imshow(
                    img, cmap="gray",
                    vmin=0, vmax=255
                )
            else:
                ax.imshow(img)

            if cropped:
                draw_box(
                    ax, box, offset=(a, b)
                )
            else:
                draw_box(ax, box)
                draw_box(
                    ax, region,
                    color="yellow",
                    linestyle="--"
                )

            ax.axis("off")

            ax.text(
                0.02, 0.02,
                label,
                transform=ax.transAxes,
                fontsize=9,
                color="white",
                bbox={
                    "facecolor": "black",
                    "alpha": 0.65,
                    "pad": 3
                }
            )

        axes[0, j].set_title(
            f"{row['scenario']}\n"
            f"Stem: {stem} | Kutu: {row['box_index']}\n"
            f"RGB={row['rgb_contrast']:.3f}  "
            f"IR={row['ir_contrast']:.3f}",
            fontsize=10
        )

    fig.suptitle(
        f"{split.upper()} - RGB/IR Kontrast Ornekleri\n"
        "Yesil: insan kutusu | Sari kesikli: analiz cevresi",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    filename = (
        "fig29_contrast_cases_train.png"
        if split == "train"
        else "fig30_contrast_cases_test.png"
    )

    output = FIG / filename
    plt.savefig(output, dpi=150)
    plt.close()

    print(f"Gorsel kaydedildi: {output}")


# 6. Train/test icin secim ve raporlama
all_selected = []

for split in ["train", "test"]:

    subset = df[df["split"] == split].copy()

    cases = select_cases(subset)

    print(f"\n===== {split.upper()} =====")

    for row in cases:
        print(
            f"{row['scenario']}: "
            f"{row['stem']} | "
            f"RGB={row['rgb_contrast']:.4f} | "
            f"IR={row['ir_contrast']:.4f} | "
            f"Fark={row['difference']:+.4f}"
        )

        all_selected.append(row.to_dict())

    create_figure(split, cases)


# 7. Secilen ornekleri CSV'ye kaydet
selected_df = pd.DataFrame(all_selected)

selected_path = LOG / "eda09_selected_cases.csv"
selected_df.to_csv(selected_path, index=False)

print("\n===== EDA 09 TAMAMLANDI =====")
print(f"Secilen toplam kutu: {len(selected_df)}")
print(f"Secim tablosu: {selected_path}")
