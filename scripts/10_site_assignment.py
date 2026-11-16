"""10_site_assignment.py - 25 ve 26 numarali on eklerdeki karelerin sahne (site) atamasi.

Girdi : data/processed/thumbs_*.npy, thumbs_index.csv   (07 onbellegi)
        outputs/logs/quality_pass.csv                   (RGB parlaklik)
        outputs/logs/frame_best_matches.csv             (09; test->train taban degerleri)
Cikti : outputs/logs/frame_site_assignment_25_26.csv
        outputs/figures/fig16_site_assignment_hist.png
        outputs/figures/fig17_site_assignment_examples.png

Not: Esik SECILMEZ; farkli esiklerde sayilar tablo olarak yazdirilir.
Benzerlik = 4 ozelligin (RGB_z, IR_z, RGB_edge, IR_edge) Pearson korelasyonu ortalamasi (09 ile ayni).
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("data/raw/LLVIP_extracted")
LOG = Path("outputs/logs")
FIG = Path("outputs/figures")
CACHE = Path("data/processed")
SRC = ["25", "26"]
THRS = [0.35, 0.40, 0.45, 0.50, 0.60]


def zrows(a):
    a = a.astype(np.float32)
    a -= a.mean(axis=1, keepdims=True)
    a /= np.maximum(a.std(axis=1, keepdims=True), 1e-6)
    return a


def grad_mag(T):
    gy, gx = np.gradient(T.astype(np.float32), axis=(1, 2))
    return np.hypot(gx, gy)


def show_img(row, size=(320, 256)):
    with Image.open(ROOT / "visible" / row["split"] / f"{row['stem']}.jpg") as im:
        im.draft("RGB", size)
        a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return np.power(a, 0.6)            # gama YALNIZCA gosterim icin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-thr", type=float, default=85.0,
                    help="RGB ortalama parlakligi bu degerin ustundeyse 'gunduz' say.")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv(CACHE / "thumbs_index.csv", dtype={"stem": str, "prefix": str})
    T_rgb, T_ir = np.load(CACHE / "thumbs_rgb.npy"), np.load(CACHE / "thumbs_ir.npy")
    N = len(idx)
    D = T_rgb.shape[1] * T_rgb.shape[2]
    stems, pref = idx["stem"].values, idx["prefix"].values
    prefixes = sorted(idx["prefix"].unique())
    P = len(prefixes)
    starts = np.array([np.where(pref == p)[0][0] for p in prefixes])
    ends = np.array([np.where(pref == p)[0][-1] + 1 for p in prefixes])
    split_of = idx.groupby("prefix")["split"].first().to_dict()
    split_arr = np.array([split_of[p] for p in prefixes])

    qp = pd.read_csv(LOG / "quality_pass.csv", dtype={"stem": str})
    br = idx.merge(qp[["stem", "rgb_mean"]], on="stem", how="left")["rgb_mean"].values

    F = {"RGB_z": zrows(T_rgb.reshape(N, -1)), "IR_z": zrows(T_ir.reshape(N, -1)),
         "RGB_edge": zrows(grad_mag(T_rgb).reshape(N, -1)),
         "IR_edge": zrows(grad_mag(T_ir).reshape(N, -1))}

    src_idx = np.where(np.isin(pref, SRC))[0]
    n_src = len(src_idx)
    best_sim = np.full((n_src, P), -np.inf, np.float32)   # kare -> on ek: en yuksek benzerlik
    best_fr = np.zeros((n_src, P), np.int64)               # o benzerligi veren karenin indeksi

    t0 = time.time()
    CH = 400
    for c0 in range(0, n_src, CH):
        ids = src_idx[c0:c0 + CH]
        comb = sum((F[k][ids] @ F[k].T) / D for k in F) / len(F)    # (c, N)
        for r, i in enumerate(ids):                                  # kendi on ekini disla
            a = prefixes.index(pref[i])
            comb[r, starts[a]:ends[a]] = -np.inf
        for bi in range(P):
            blk = comb[:, starts[bi]:ends[bi]]
            j = blk.argmax(axis=1)
            best_sim[c0:c0 + len(ids), bi] = blk[np.arange(len(ids)), j]
            best_fr[c0:c0 + len(ids), bi] = starts[bi] + j
        print(f"  {min(c0 + CH, n_src)}/{n_src} kare ({time.time() - t0:.0f} sn)")

    i25, i26 = prefixes.index("25"), prefixes.index("26")
    rows = []
    for r, i in enumerate(src_idx):
        my = idx["split"].iloc[i]
        same = np.where(split_arr == my)[0]
        cross = np.where(split_arr != my)[0]
        s = best_sim[r, same]
        o = np.argsort(s)[::-1]
        b1, b2 = same[o[0]], same[o[1]]
        bc = cross[np.argmax(best_sim[r, cross])]
        other = i26 if pref[i] == "25" else i25
        rows.append({
            "stem": stems[i], "prefix": pref[i], "split": my,
            "rgb_mean": round(float(br[i]), 2), "is_day": bool(br[i] >= args.day_thr),
            "same_best_prefix": prefixes[b1], "same_best_sim": round(float(best_sim[r, b1]), 3),
            "same_best_stem": stems[best_fr[r, b1]],
            "same_second_prefix": prefixes[b2], "same_second_sim": round(float(best_sim[r, b2]), 3),
            "cross_best_prefix": prefixes[bc], "cross_best_sim": round(float(best_sim[r, bc]), 3),
            "sim_to_other_src": round(float(best_sim[r, other]), 3),
        })
    df = pd.DataFrame(rows)
    df.to_csv(LOG / "frame_site_assignment_25_26.csv", index=False)

    base = pd.read_csv(LOG / "frame_best_matches.csv", dtype={"test_prefix": str})
    base = base[base["test_prefix"] != "26"]["sim_comb"]
    bq = {f"p{q}": round(float(np.percentile(base, q)), 3) for q in (50, 95, 99)}
    bq["max"] = round(float(base.max()), 3)

    # ---------------- ekrana yazdir ----------------
    print("\n=== KARE SAYILARI (gunduz / gece; esik =", args.day_thr, ") ===")
    print(df.groupby(["prefix", "is_day"]).size().unstack(fill_value=0).to_string())

    print("\n=== TABAN: test->train en iyi kare benzerligi (26 haric, 09 ciktisi) ===")
    print(bq)

    print("\n=== CAPRAZ BOLME BENZERLIGI (cross_best_sim): kare -> diger bolmedeki en benzer kare ===")
    print(df.groupby(["prefix", "is_day"])["cross_best_sim"]
            .describe(percentiles=[.5, .95, .99]).round(3).to_string())

    print("\n=== 25 <-> 26 DOGRUDAN (sim_to_other_src) ===")
    print(df.groupby(["prefix", "is_day"])["sim_to_other_src"]
            .describe(percentiles=[.5, .95]).round(3).to_string())

    print("\n=== AYNI BOLME EN YAKIN ON EK: esige gore kare sayilari (gece kareleri) ===")
    night = df[~df["is_day"]]
    tab = {p: {t: int((night[night["prefix"] == p]["same_best_sim"] >= t).sum()) for t in THRS}
           for p in SRC}
    print(pd.DataFrame(tab).T.rename_axis("prefix").to_string())
    for p in SRC:
        g = night[(night["prefix"] == p) & (night["same_best_sim"] >= 0.45)]
        print(f"\n[{p}] gece karelerinde sim>=0.45 olanlarin en yakin on ek dagilimi:")
        print(g["same_best_prefix"].value_counts().to_string())

    print("\n=== GUNDUZ KARELERI: aynı bolmede en yakin on ek sim ozeti ===")
    print(df[df["is_day"]].groupby("prefix")["same_best_sim"].describe().round(3).to_string())

    # ---------------- fig16: histogramlar ----------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    bins = np.linspace(0.0, 0.9, 46)
    for ax, p in zip(axes, SRC):
        g = df[df["prefix"] == p]
        ax.hist(base, bins=bins, density=True, alpha=0.4, color="gray",
                label="taban: test->train (26 haric)")
        ax.hist(g[g["is_day"]]["same_best_sim"], bins=bins, density=True, alpha=0.6,
                color="tab:orange", label=f"{p} gunduz")
        ax.hist(g[~g["is_day"]]["same_best_sim"], bins=bins, density=True, alpha=0.6,
                color="tab:blue", label=f"{p} gece")
        ax.axvline(bq["p99"], color="k", ls="--", lw=1, label="taban p99")
        ax.set_title(f"on ek {p}: karenin ayni bolmedeki en yakin on eke benzerligi")
        ax.set_xlabel("COMB benzerlik")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig16_site_assignment_hist.png", dpi=130)
    plt.close()

    # ---------------- fig17: ornek eslesmeler ----------------
    pos_of = {s: k for k, s in enumerate(stems)}
    fig, axes = plt.subplots(8, 6, figsize=(21, 30))
    for ax in axes.ravel():
        ax.axis("off")
    for pi, p in enumerate(SRC):
        g = night[night["prefix"] == p].sort_values("same_best_sim", ascending=False)
        if len(g) < 12:
            continue
        mid = (g["same_best_sim"] - g["same_best_sim"].median()).abs().sort_values().index[:6]
        for sec, picks in enumerate([g.head(6), g.loc[mid]]):
            for j, (_, r) in enumerate(picks.iterrows()):
                rs = 4 * pi + 2 * sec
                axes[rs, j].imshow(show_img(idx.iloc[pos_of[r["stem"]]]))
                axes[rs, j].set_title(f"{p}: {r['stem']} ({'en yuksek' if sec == 0 else 'medyan'})",
                                      fontsize=9)
                m = idx.iloc[pos_of[r["same_best_stem"]]]
                axes[rs + 1, j].imshow(show_img(m))
                axes[rs + 1, j].set_title(
                    f"eslesen on ek {r['same_best_prefix']}: {r['same_best_stem']}  sim={r['same_best_sim']}",
                    fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "fig17_site_assignment_examples.png", dpi=80)
    plt.close()
    print(f"\nToplam sure: {time.time() - t0:.0f} sn")
    print("Gorseller: fig16, fig17;  CSV: frame_site_assignment_25_26.csv")


if __name__ == "__main__":
    main()