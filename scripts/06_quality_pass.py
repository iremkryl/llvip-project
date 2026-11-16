"""06_quality_pass.py - Kalite taramasi.

Kullanim:
  python scripts/06_quality_pass.py --limit 200   (hizli deneme, sure olcmek icin)
  python scripts/06_quality_pass.py               (tum veri)

Uretilen dosyalar:
  outputs/logs/quality_pass.csv            -> goruntu cifti basina 1 satir
  outputs/logs/quality_summary.json
  outputs/logs/invalid_boxes.csv
  outputs/figures/fig03_zero_box_images.png
  outputs/figures/fig04_invalid_boxes.png
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path("data/raw/LLVIP_extracted")
LOG = Path("outputs/logs")
FIG = Path("outputs/figures")
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # RGB -> gri agirliklari


def analyze_pair(split: str, stem: str) -> dict:
    """Bir RGB+IR cifti tam cozer ve istatistik cikarir."""
    rec = {"stem": stem, "split": split, "rgb_ok": True, "ir_ok": True, "err": ""}

    # --- RGB: tam cozme (np.asarray piksel verisinin tamamini okutur) ---
    try:
        with Image.open(ROOT / "visible" / split / f"{stem}.jpg") as im:
            a = np.asarray(im.convert("RGB"), dtype=np.float32)
        gray = a @ LUMA
        rec["rgb_mean"] = float(gray.mean())
        # her 4. piksel: yuzdelik hesabi hizlansin diye
        rec["rgb_p95"] = float(np.percentile(gray[::4, ::4], 95))
    except Exception as e:
        rec["rgb_ok"] = False
        rec["err"] += f"RGB:{e!r};"

    # --- IR: tam cozme + kanal ozdeslik testi ---
    try:
        with Image.open(ROOT / "infrared" / split / f"{stem}.jpg") as im:
            b = np.asarray(im.convert("RGB"), dtype=np.int16)
        d_rg = np.abs(b[..., 0] - b[..., 1])
        d_gb = np.abs(b[..., 1] - b[..., 2])
        rec["ir_mean"] = float(b[..., 1].mean())            # yesil kanal
        rec["ir_chdiff_mean"] = float(max(d_rg.mean(), d_gb.mean()))
        rec["ir_chdiff_max"] = int(max(d_rg.max(), d_gb.max()))
    except Exception as e:
        rec["ir_ok"] = False
        rec["err"] += f"IR:{e!r};"
    return rec


def crop_box(img: Image.Image, cx: float, cy: float, half: int = 150):
    """Kutunun etrafindan kare bir kirpma dondurur (x0, y0 ile birlikte)."""
    w, h = img.size
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
    return img.crop((x0, y0, x1, y1)), x0, y0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Rastgele N cift uzerinde calis (hizli deneme).")
    args = ap.parse_args()

    LOG.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    imgs = pd.read_csv(LOG / "images_table.csv", dtype={"stem": str, "prefix": str})
    boxes = pd.read_csv(LOG / "boxes_table.csv", dtype={"stem": str, "prefix": str})

    # ------------------------------------------------------------------
    # A) Tam cozme + parlaklik + IR kanal testi
    # ------------------------------------------------------------------
    pairs = list(zip(imgs["split"], imgs["stem"]))
    suffix = ""
    if args.limit:
        random.seed(42)
        pairs = random.sample(pairs, args.limit)
        suffix = "_sample"

    rows = [analyze_pair(s, st) for s, st in tqdm(pairs, desc="Tam cozme")]
    q = pd.DataFrame(rows)
    q.to_csv(LOG / f"quality_pass{suffix}.csv", index=False)

    summary = {
        "n_pairs_checked": int(len(q)),
        "rgb_decode_failures": int((~q["rgb_ok"]).sum()),
        "ir_decode_failures": int((~q["ir_ok"]).sum()),
        "failed_stems": q.loc[~(q["rgb_ok"] & q["ir_ok"]), "stem"].tolist(),
        "rgb_mean_describe": q["rgb_mean"].describe().round(2).to_dict(),
        "ir_mean_describe": q["ir_mean"].describe().round(2).to_dict(),
        "ir_channels_exactly_identical_images": int((q["ir_chdiff_max"] == 0).sum()),
        "ir_chdiff_mean_describe": q["ir_chdiff_mean"].describe().round(4).to_dict(),
        "ir_chdiff_max_describe": q["ir_chdiff_max"].describe().round(2).to_dict(),
    }

    # ------------------------------------------------------------------
    # B) Gecersiz kutular ve kutusuz goruntuler (tum veri uzerinden)
    # ------------------------------------------------------------------
    invalid = boxes[boxes["flag_nonpositive"]].copy()
    invalid.to_csv(LOG / "invalid_boxes.csv", index=False)
    zero_imgs = imgs[imgs["n_boxes"] == 0]
    summary["invalid_boxes"] = invalid[
        ["stem", "split", "xmin", "ymin", "xmax", "ymax", "w", "h"]
    ].to_dict("records")
    summary["zero_box_images"] = zero_imgs[["stem", "split", "prefix"]].to_dict("records")

    # Bu kutularin yakinindaki "en kucuk 15 kutu" baglam icin
    smallest = boxes.nsmallest(15, "area")[
        ["stem", "split", "xmin", "ymin", "xmax", "ymax", "w", "h", "area"]
    ]
    print("\n=== EN KUCUK 15 KUTU (alana gore) ===")
    print(smallest.to_string(index=False))

    (LOG / f"quality_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\n=== OZET ===")
    print(json.dumps(summary, indent=2, default=str))

    # ------------------------------------------------------------------
    # C) Gorseller
    # ------------------------------------------------------------------
    # fig03: kutusuz goruntuler (RGB ust, IR alt)
    if len(zero_imgs):
        fig, axes = plt.subplots(2, len(zero_imgs), figsize=(6 * len(zero_imgs), 8),
                                 squeeze=False)
        for j, (_, r) in enumerate(zero_imgs.iterrows()):
            for i, folder in enumerate(["visible", "infrared"]):
                im = Image.open(ROOT / folder / r["split"] / f"{r['stem']}.jpg")
                axes[i, j].imshow(im)
                axes[i, j].set_title(f"{'RGB' if i == 0 else 'IR'} {r['split']}/{r['stem']} (0 kutu)",
                                     fontsize=9)
                axes[i, j].axis("off")
        plt.tight_layout()
        plt.savefig(FIG / "fig03_zero_box_images.png", dpi=150)
        plt.close()

    # fig04: gecersiz kutular - kutunun etrafindan kirpma
    if len(invalid):
        n = len(invalid)
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), squeeze=False)
        for j, (_, r) in enumerate(invalid.iterrows()):
            cx, cy = (r.xmin + r.xmax) / 2, (r.ymin + r.ymax) / 2
            for i, folder in enumerate(["visible", "infrared"]):
                im = Image.open(ROOT / folder / r["split"] / f"{r['stem']}.jpg")
                crop, x0, y0 = crop_box(im, cx, cy)
                ax = axes[i, j]
                ax.imshow(crop)
                ax.add_patch(patches.Rectangle(
                    (r.xmin - x0, r.ymin - y0), r.w, r.h,
                    fill=False, edgecolor="yellow", linewidth=2))
                ax.set_title(f"{'RGB' if i == 0 else 'IR'} {r['split']}/{r['stem']}\n"
                             f"w={r.w:.0f}, h={r.h:.0f}", fontsize=8)
                ax.axis("off")
        plt.tight_layout()
        plt.savefig(FIG / "fig04_invalid_boxes.png", dpi=150)
        plt.close()

    print("\nGorseller: outputs/figures/fig03_zero_box_images.png, fig04_invalid_boxes.png")


if __name__ == "__main__":
    main()