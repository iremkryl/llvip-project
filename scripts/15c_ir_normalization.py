"""
ADIM 15C - IR KANAL VE NORMALIZASYON ANALIZI

AMAC:
LLVIP veri setindeki IR goruntulerinin kanal yapisini
dogrulamak ve goruntu bazli yogunluk normalizasyonunun
piksel dagilimlarina etkisini incelemek.

YONTEM:
1. Train/test kumelerinden seed=42 ile 100'er IR goruntusu secilir.
2. RGB olarak saklanan IR kanallarinin ozdesligi piksel duzeyinde kontrol edilir.
3. Ozdes kanallardan tek kanalli IR elde edilir.
4. Her goruntunun p1 ve p99 yogunluklari hesaplanir.
5. p1-p99 araligi 0-255'e esnetilerek ornek normalizasyon uygulanir.
6. Ortalama, standart sapma ve kirpilma oranlari once/sonra karsilastirilir.
7. Dusuk ve yuksek yogunluklu gercek ornekler yan yana gorsellestirilir.
8. Ham veri degistirilmez.

GIRDI:
outputs/logs/images_table.csv
data/raw/LLVIP_extracted/infrared/

CIKTI:
outputs/logs/step15c_ir_sample_stats.csv
outputs/figures/fig34_ir_normalization_before_after_en.png
Terminal: Kanal ve normalizasyon istatistikleri.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("data/raw/LLVIP_extracted/infrared")
LOG = Path("outputs/logs")
FIG = Path("outputs/figures")

FIG.mkdir(parents=True, exist_ok=True)

SEED = 42
SAMPLE_SIZE = 100

# 1. Goruntu envanteri
df = pd.read_csv(
    LOG / "images_table.csv",
    dtype={"stem": str}
)

df["stem"] = df["stem"].str.zfill(6)


# 2. Yuzdelik tabanli normalizasyon
def percentile_stretch(gray):

    p1, p99 = np.percentile(gray, [1, 99])

    # Tek veya neredeyse sabit yogunluk durumu
    if p99 - p1 < 1:
        return gray.copy(), float(p1), float(p99)

    normalized = (
        gray.astype(np.float32) - p1
    ) / (p99 - p1)

    normalized = np.clip(
        normalized * 255, 0, 255
    ).astype(np.uint8)

    return normalized, float(p1), float(p99)


results = []
examples = []

# 3. Train/test orneklerini incele
for split in ["train", "test"]:

    subset = df[df["split"] == split]

    sample = subset.sample(
        n=min(SAMPLE_SIZE, len(subset)),
        random_state=SEED
    )

    split_examples = []

    print(f"\n===== {split.upper()} =====")
    print(f"Secilen IR goruntusu: {len(sample)}")

    for _, row in sample.iterrows():

        stem = row["stem"]
        path = ROOT / split / f"{stem}.jpg"

        with Image.open(path) as im:
            original_mode = im.mode
            rgb_array = np.asarray(
                im.convert("RGB")
            )

        # Uc kanal gercekten ayni mi?
        identical = (
            np.array_equal(
                rgb_array[:, :, 0],
                rgb_array[:, :, 1]
            )
            and
            np.array_equal(
                rgb_array[:, :, 1],
                rgb_array[:, :, 2]
            )
        )

        if not identical:
            raise ValueError(
                f"IR kanallari ozdes degil: {stem}"
            )

        # Bilgi kaybetmeden tek kanala indirgeme
        gray = rgb_array[:, :, 0].copy()

        # Goruntu bazli normalizasyon denemesi
        normalized, p1, p99 = percentile_stretch(
            gray
        )

        raw_clipped = np.mean(
            (gray == 0) | (gray == 255)
        ) * 100

        new_clipped = np.mean(
            (normalized == 0) |
            (normalized == 255)
        ) * 100

        result = {
            "stem": stem,
            "split": split,
            "source_mode": original_mode,
            "channels_identical": identical,
            "raw_mean": float(gray.mean()),
            "raw_std": float(gray.std()),
            "raw_p1": p1,
            "raw_p99": p99,
            "raw_range_p1_p99": p99 - p1,
            "normalized_mean": float(
                normalized.mean()
            ),
            "normalized_std": float(
                normalized.std()
            ),
            "raw_clipped_pct": raw_clipped,
            "normalized_clipped_pct": new_clipped
        }

        results.append(result)

        split_examples.append({
            "stem": stem,
            "raw": gray,
            "normalized": normalized,
            "stats": result
        })

    # Orneklemden dusuk ve yuksek ortalama
    # yogunluklu iki ayri goruntu sec.
    split_examples.sort(
        key=lambda x: x["stats"]["raw_mean"]
    )

    examples.append(
        (split, "Low IR intensity",
         split_examples[0])
    )

    examples.append(
        (split, "High IR intensity",
         split_examples[-1])
    )


# 4. Sayisal sonuclari kaydet
res = pd.DataFrame(results)

csv_path = LOG / "step15c_ir_sample_stats.csv"
res.to_csv(csv_path, index=False)

print("\n===== KANAL KONTROLU =====")
print(f"Toplam incelenen: {len(res)}")

print(
    "Kanallari ozdes olan: "
    f"{res['channels_identical'].sum()}"
)

print(
    "Tekrarlanan uc kanaldan birine geciste "
    "piksel bilgisi korunur."
)

for split in ["train", "test"]:

    sub = res[res["split"] == split]

    print(f"\n===== {split.upper()} OZET =====")

    print(
        f"RAW IR ortalama yogunluk: "
        f"{sub['raw_mean'].mean():.2f}"
    )

    print(
        f"Normalize IR ortalama yogunluk: "
        f"{sub['normalized_mean'].mean():.2f}"
    )

    print(
        f"RAW IR standart sapma medyani: "
        f"{sub['raw_std'].median():.2f}"
    )

    print(
        f"Normalize standart sapma medyani: "
        f"{sub['normalized_std'].median():.2f}"
    )

    print(
        f"RAW kirpilmis piksel orani ort.: "
        f"%{sub['raw_clipped_pct'].mean():.2f}"
    )

    print(
        f"Normalizasyon sonrasi kirpilmis "
        f"piksel orani ort.: "
        f"%{sub['normalized_clipped_pct'].mean():.2f}"
    )


# 5. Gercek once-sonra gorseli
fig, axes = plt.subplots(
    len(examples), 2,
    figsize=(12, 16)
)

for i, (split, label, ex) in enumerate(examples):

    raw = ex["raw"]
    normalized = ex["normalized"]
    stats = ex["stats"]

    axes[i, 0].imshow(
        raw, cmap="gray",
        vmin=0, vmax=255
    )

    axes[i, 1].imshow(
        normalized, cmap="gray",
        vmin=0, vmax=255
    )

    axes[i, 0].set_title(
        f"{split.upper()} / {label}\n"
        f"{ex['stem']} - RAW\n"
        f"Mean={stats['raw_mean']:.1f}",
        fontsize=10
    )

    axes[i, 1].set_title(
        f"{ex['stem']} - NORMALIZED\n"
        f"p1={stats['raw_p1']:.1f} | "
        f"p99={stats['raw_p99']:.1f}\n"
        f"Clipped="
        f"{stats['normalized_clipped_pct']:.1f}%",
        fontsize=10
    )

    for ax in axes[i]:
        ax.axis("off")

fig.suptitle(
    "IR Intensity Normalization: Before and After\n"
    "Per-image percentile stretching (p1-p99)",
    fontsize=14
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

fig_path = (
    FIG / "fig34_ir_normalization_before_after_en.png"
)

plt.savefig(fig_path, dpi=150)
plt.close()

print("\n===== ADIM 15C TAMAMLANDI =====")
print(f"CSV: {csv_path}")
print(f"Gorsel: {fig_path}")