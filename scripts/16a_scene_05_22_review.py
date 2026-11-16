"""
AMAC:
Train 05 ve test 22 on eklerine ait goruntulerin
ayni fiziksel sahneden gelme olasiligini incelemek.

YONTEM:
1. IR goruntulerinden zamana yayilmis ornekler secilir.
2. Kucultulmus gri tonlu goruntuler normalize edilir.
3. IR yapisal benzerlik adaylari siralanir.
4. Tekrarsiz en yuksek 6 aday secilir.
5. Her aday RGB ve IR olarak yan yana gosterilir.
6. Yalnizca inceleme gorseli ve CSV olusturulur.
7. Kesin sahne veya sizinti karari verilmez.

GIRDI:
data/raw/LLVIP_extracted/infrared/train/05*.jpg
data/raw/LLVIP_extracted/infrared/test/22*.jpg
data/raw/LLVIP_extracted/visible/

CIKTI:
outputs/logs/step16a_05_22_candidates.csv
outputs/figures/fig37_05_22_scene_candidates_en.png
"""

from pathlib import Path

import numpy as np
import pandas as pd

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageFilter
)


ROOT = Path("data/raw/LLVIP_extracted")
LOG = Path("outputs/logs")
FIG = Path("outputs/figures")

LOG.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MAX_PER_GROUP = 180
TOP_K = 6

FEATURE_SIZE = (80, 64)
THUMB_SIZE = (280, 224)


# 1. Goruntu dosyalarini bul
train_files = sorted(
    (ROOT / "infrared" / "train").glob("05*.jpg")
)

test_files = sorted(
    (ROOT / "infrared" / "test").glob("22*.jpg")
)

if not train_files or not test_files:
    raise FileNotFoundError(
        "Train 05 veya test 22 IR goruntuleri "
        "bulunamadi. Veri klasoru yolunu kontrol et."
    )


# 2. Zamana yayilmis orneklem sec
def spread_sample(files, maximum):

    n = min(len(files), maximum)

    indices = np.linspace(
        0,
        len(files) - 1,
        n
    ).astype(int)

    indices = np.unique(indices)

    return [files[i] for i in indices]


train_sample = spread_sample(
    train_files, MAX_PER_GROUP
)

test_sample = spread_sample(
    test_files, MAX_PER_GROUP
)


# 3. IR goruntusunden basit yapi ozelligi cikar
def extract_feature(path):

    with Image.open(path) as im:

        gray = im.convert("L")

        gray = gray.resize(
            FEATURE_SIZE,
            Image.Resampling.BILINEAR
        )

        gray = gray.filter(
            ImageFilter.GaussianBlur(radius=1.0)
        )

        arr = np.asarray(
            gray,
            dtype=np.float32
        ).flatten()

    # Ortalama yogunluk farkini azalt.
    arr = arr - arr.mean()

    norm = np.linalg.norm(arr)

    if norm < 1e-8:
        return np.zeros_like(arr)

    return arr / norm


train_features = np.stack([
    extract_feature(p)
    for p in train_sample
])

test_features = np.stack([
    extract_feature(p)
    for p in test_sample
])


# 4. Normalize edilmis piksel korelasyonu
# Bu skor yalnizca aday siralamak icindir.
similarity = (
    train_features @ test_features.T
)

ranking = np.argsort(
    similarity.ravel()
)[::-1]

selected = []

used_train = set()
used_test = set()

for flat_index in ranking:

    i, j = np.unravel_index(
        flat_index,
        similarity.shape
    )

    if i in used_train or j in used_test:
        continue

    selected.append({
        "rank": len(selected) + 1,
        "train_prefix": "05",
        "test_prefix": "22",
        "train_stem": train_sample[i].stem,
        "test_stem": test_sample[j].stem,
        "ir_similarity_score": float(
            similarity[i, j]
        )
    })

    used_train.add(i)
    used_test.add(j)

    if len(selected) == TOP_K:
        break

if len(selected) < TOP_K:
    raise ValueError(
        "Yeterli sayida benzersiz aday bulunamadi."
    )


# 5. Adaylari CSV'ye kaydet
result = pd.DataFrame(selected)

csv_path = (
    LOG / "step16a_05_22_candidates.csv"
)

result.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)


# 6. Ingilizce gorsel paneli olustur
def get_font(size):

    for name in (
        "arial.ttf",
        "DejaVuSans.ttf"
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass

    return ImageFont.load_default()


TITLE_FONT = get_font(20)
LABEL_FONT = get_font(16)
SMALL_FONT = get_font(14)

IMAGE_W, IMAGE_H = THUMB_SIZE

LEFT = 18
COL_GAP = 15
TOP = 65
ROW_H = 290

SHEET_W = (
    LEFT * 2 +
    IMAGE_W * 4 +
    COL_GAP * 3
)

SHEET_H = TOP + TOP_K * ROW_H + 25

sheet = Image.new(
    "RGB",
    (SHEET_W, SHEET_H),
    "white"
)

draw = ImageDraw.Draw(sheet)

draw.text(
    (LEFT, 12),
    "Potential Cross-Split Scene Similarity: Train 05 vs Test 22",
    fill="black",
    font=TITLE_FONT
)

draw.text(
    (LEFT, 40),
    "IR-based candidate ranking; visual verification required",
    fill="#555555",
    font=SMALL_FONT
)


def load_thumb(folder, split, stem):

    path = ROOT / folder / split / f"{stem}.jpg"

    if not path.exists():
        raise FileNotFoundError(path)

    with Image.open(path) as im:
        return im.convert("RGB").resize(
            THUMB_SIZE,
            Image.Resampling.LANCZOS
        )


for i, row in result.iterrows():

    y = TOP + i * ROW_H

    train_id = row["train_stem"]
    test_id = row["test_stem"]

    score = row["ir_similarity_score"]

    draw.text(
        (LEFT, y),
        (
            f"Candidate {i+1} | "
            f"IR similarity: {score:.4f}"
        ),
        fill="black",
        font=LABEL_FONT
    )

    panels = [
        (
            "Train 05 - IR",
            "infrared",
            "train",
            train_id
        ),
        (
            "Test 22 - IR",
            "infrared",
            "test",
            test_id
        ),
        (
            "Train 05 - RGB",
            "visible",
            "train",
            train_id
        ),
        (
            "Test 22 - RGB",
            "visible",
            "test",
            test_id
        )
    ]

    for j, (label, folder, split, stem) in enumerate(
        panels
    ):

        x = LEFT + j * (IMAGE_W + COL_GAP)

        image = load_thumb(
            folder,
            split,
            stem
        )

        sheet.paste(
            image,
            (x, y + 24)
        )

        draw.text(
            (x, y + 252),
            f"{label} | {stem}",
            fill="black",
            font=SMALL_FONT
        )

    draw.line(
        (
            LEFT,
            y + ROW_H - 8,
            SHEET_W - LEFT,
            y + ROW_H - 8
        ),
        fill="#dddddd",
        width=2
    )


fig_path = (
    FIG / "fig37_05_22_scene_candidates_en.png"
)

sheet.save(fig_path)


# 7. Terminal ozeti
print("\n===== ADIM 16A - SAHNE BENZERLIGI =====")

print(f"Train 05 toplam IR goruntusu: {len(train_files)}")
print(f"Test 22 toplam IR goruntusu: {len(test_files)}")

print(f"Train 05 incelenen ornek: {len(train_sample)}")
print(f"Test 22 incelenen ornek: {len(test_sample)}")

print(f"Secilen aday cift: {len(result)}")

print("\n===== EN YUKSEK BENZERLIK ADAYLARI =====")

for _, row in result.iterrows():

    print(
        f"{int(row['rank'])}. "
        f"{row['train_stem']} <-> "
        f"{row['test_stem']} | "
        f"Skor: {row['ir_similarity_score']:.4f}"
    )

print("\n===== OLUSTURULAN DOSYALAR =====")
print(f"CSV: {csv_path}")
print(f"Gorsel: {fig_path}")

print(
    "\nNot: Benzerlik skoru ayni sahnenin "
    "kesin kaniti degildir."
)

print("\nADIM 16A TAMAMLANDI.")
