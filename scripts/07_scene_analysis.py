"""07_scene_analysis.py - Onek=sahne hipotezi, zamansal benzerlik, parlaklik.

Uretilen dosyalar:
  data/processed/thumbs_rgb.npy, thumbs_ir.npy, thumbs_index.csv   (onbellek)
  outputs/logs/prefix_brightness.csv
  outputs/logs/index_gaps.csv
  outputs/logs/scene_similarity_summary.json
  outputs/figures/fig06_brightness_by_prefix.png
  outputs/figures/fig07_similarity_groups.png
  outputs/figures/fig08_prefix_contact_sheet.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image
from tqdm import tqdm

ROOT = Path("data/raw/LLVIP_extracted")
LOG = Path("outputs/logs")
FIG = Path("outputs/figures")
CACHE = Path("data/processed")
TW, TH = 80, 64


def thumb_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im.draft("RGB", (160, 128))          # JPEG'i 1/8 olcekte cozer (hizli)
        g = im.convert("L").resize((TW, TH), Image.BILINEAR)
    return np.asarray(g, dtype=np.uint8)


def mad_pairs(T: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """Iki kucuk resim arasindaki ortalama mutlak fark (0-1 olcekli)."""
    out = []
    for s in range(0, len(a), chunk):
        x = T[a[s:s + chunk]].astype(np.int16)
        y = T[b[s:s + chunk]].astype(np.int16)
        out.append(np.abs(x - y).mean(axis=(1, 2)) / 255.0)
    return np.concatenate(out)


def q(x):
    x = np.asarray(x)
    return {"n": int(len(x)), "q25": round(float(np.percentile(x, 25)), 4),
            "median": round(float(np.median(x)), 4), "q75": round(float(np.percentile(x, 75)), 4)}


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    imgs = pd.read_csv(LOG / "images_table.csv", dtype={"stem": str, "prefix": str})
    imgs = imgs.sort_values(["prefix", "idx_in_prefix"]).reset_index(drop=True)
    prefixes = sorted(imgs["prefix"].unique())
    pref = imgs["prefix"].values
    groups = {p: np.where(pref == p)[0] for p in prefixes}  # siralamadan dolayi bitisik

    # ---------- 1) Kucuk resim onbellegi ----------
    f_rgb, f_ir = CACHE / "thumbs_rgb.npy", CACHE / "thumbs_ir.npy"
    if f_rgb.exists() and f_ir.exists():
        T_rgb, T_ir = np.load(f_rgb), np.load(f_ir)
        assert len(T_rgb) == len(imgs), "Onbellek ile tablo uyusmuyor; onbellegi silin."
    else:
        T_rgb = np.zeros((len(imgs), TH, TW), np.uint8)
        T_ir = np.zeros((len(imgs), TH, TW), np.uint8)
        for i, (sp, st) in enumerate(tqdm(zip(imgs["split"], imgs["stem"]),
                                          total=len(imgs), desc="Kucuk resimler")):
            T_rgb[i] = thumb_gray(ROOT / "visible" / sp / f"{st}.jpg")
            T_ir[i] = thumb_gray(ROOT / "infrared" / sp / f"{st}.jpg")
        np.save(f_rgb, T_rgb)
        np.save(f_ir, T_ir)
        imgs[["stem", "split", "prefix", "idx_in_prefix"]].to_csv(
            CACHE / "thumbs_index.csv", index=False)

    # ---------- 2) On ek bazinda parlaklik ----------
    qp = pd.read_csv(LOG / "quality_pass.csv", dtype={"stem": str})
    m = imgs.merge(qp[["stem", "rgb_mean", "ir_mean"]], on="stem", how="left")
    pb = (m.groupby(["prefix", "split"])
          .agg(n=("stem", "count"), rgb_median=("rgb_mean", "median"),
               rgb_min=("rgb_mean", "min"), rgb_max=("rgb_mean", "max"),
               ir_median=("ir_mean", "median")).round(2).reset_index())
    pb.to_csv(LOG / "prefix_brightness.csv", index=False)
    print("=== ON EK BAZINDA PARLAKLIK ===")
    print(pb.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, col, title in zip(axes, ["rgb_mean", "ir_mean"],
                              ["RGB ortalama parlaklik (gri, 0-255)",
                               "IR ortalama yogunluk (0-255)"]):
        data = [m.loc[m["prefix"] == p, col].values for p in prefixes]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, p in zip(bp["boxes"], prefixes):
            sp = m.loc[m["prefix"] == p, "split"].iloc[0]
            patch.set_facecolor("tab:blue" if sp == "train" else "tab:orange")
        ax.set_xticklabels(prefixes, rotation=90, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("dosya adi on eki (olasi sahne)")
    axes[0].legend(handles=[Patch(color="tab:blue", label="train"),
                            Patch(color="tab:orange", label="test")])
    plt.tight_layout()
    plt.savefig(FIG / "fig06_brightness_by_prefix.png", dpi=150)
    plt.close()

    # ---------- 3) Numara bosluklari ----------
    gap_rows = []
    for p in prefixes:
        pos = groups[p]
        idx = imgs.loc[pos, "idx_in_prefix"].values
        stems = imgs.loc[pos, "stem"].values
        for k in np.where(np.diff(idx) > 1)[0]:
            gap_rows.append({"prefix": p, "after_stem": stems[k], "before_stem": stems[k + 1],
                             "missing_count": int(idx[k + 1] - idx[k] - 1)})
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(LOG / "index_gaps.csv", index=False)

    # ---------- 4) Benzerlik: ardisik / ayni sahne uzak / farkli on ek ----------
    rng = np.random.default_rng(42)
    ca, cb = [], []
    for p in prefixes:
        pos = groups[p]
        ca.append(pos[:-1])
        cb.append(pos[1:])
    ca, cb = np.concatenate(ca), np.concatenate(cb)

    wa, wb = [], []
    for p in prefixes:
        pos = groups[p]
        a = rng.choice(pos, 400)
        b = rng.choice(pos, 400)
        ok = np.abs(a - b) >= 100
        wa.append(a[ok])
        wb.append(b[ok])
    wa, wb = np.concatenate(wa), np.concatenate(wb)

    xa = rng.integers(0, len(imgs), 12000)
    xb = rng.integers(0, len(imgs), 12000)
    ok = pref[xa] != pref[xb]
    xa, xb = xa[ok], xb[ok]

    result = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, name, T in zip(axes, ["RGB", "IR"], [T_rgb, T_ir]):
        d_c = mad_pairs(T, ca, cb)
        d_w = mad_pairs(T, wa, wb)
        d_x = mad_pairs(T, xa, xb)
        result[name] = {
            "ardisik_kareler": q(d_c),
            "ayni_onek_uzak_kareler(>=100 kare)": q(d_w),
            "farkli_onek": q(d_x),
            "ardisik_birebir_ayni_kucuk_resim": int((d_c == 0).sum()),
            "farkli_onek_ciftlerinin_orani_<_ayni_onek_uzak_p95":
                round(float((d_x < np.percentile(d_w, 95)).mean()), 4),
        }
        ax.boxplot([np.maximum(d_c, 1e-4), np.maximum(d_w, 1e-4), np.maximum(d_x, 1e-4)],
                   tick_labels=["ardisik", "ayni on ek\nuzak", "farkli on ek"], showfliers=False)
        ax.set_yscale("log")
        ax.set_title(f"{name}: ortalama mutlak piksel farki (80x64 gri)")
    plt.tight_layout()
    plt.savefig(FIG / "fig07_similarity_groups.png", dpi=150)
    plt.close()

    # ---------- 5) Kenara degen kutular ----------
    bx = pd.read_csv(LOG / "boxes_table.csv", dtype={"stem": str, "prefix": str})
    bx = bx[~bx["flag_nonpositive"]]
    touch = (bx.xmin <= 1) | (bx.ymin <= 1) | (bx.xmax >= 1280) | (bx.ymax >= 1024)
    border = {"n_kenara_degen_kutu": int(touch.sum()),
              "oran": round(float(touch.mean()), 4),
              "en_cok_onek": bx[touch]["prefix"].value_counts().head(5).to_dict()}

    summary = {
        "n_pairs": int(len(imgs)),
        "ardisik_ciftlerde_numara_boslugu_olan": int((np.diff(
            imgs["idx_in_prefix"].values)[np.diff(imgs["idx_in_prefix"].values) != 0] > 1).sum()),
        "toplam_eksik_numara": int(gaps["missing_count"].sum()) if len(gaps) else 0,
        "benzerlik": result,
        "kenara_degen_kutular": border,
    }
    (LOG / "scene_similarity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== OZET ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== NUMARA BOSLUKLARI (ilk 20) ===")
    print(gaps.head(20).to_string(index=False))

    # ---------- 6) On ek temsilci kareler (RGB, gosterim icin gama 0.6) ----------
    fig, axes = plt.subplots(4, 7, figsize=(21, 12))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, p in zip(axes.ravel(), prefixes):
        pos = groups[p][len(groups[p]) // 2]          # ortadaki kare
        r = imgs.iloc[pos]
        with Image.open(ROOT / "visible" / r["split"] / f"{r['stem']}.jpg") as im:
            im.draft("RGB", (320, 256))
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        ax.imshow(np.power(arr, 0.6))
        ax.set_title(f"on ek {p} ({r['split']}) {r['stem']}", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "fig08_prefix_contact_sheet.png", dpi=120)
    print("\nGorseller: fig06, fig07, fig08 (outputs/figures/)")


if __name__ == "__main__":
    main()