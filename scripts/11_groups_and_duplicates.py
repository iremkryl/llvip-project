"""11_groups_and_duplicates.py

(A) 25/26 on eklerindeki karelere sahne grubu atamasi (esik: benzerlik degerlerindeki en buyuk bosluk)
(B) Gece karelerini eslestikleri kareyle TAM COZUNURLUKTE karsilastirma (tekrar/yakin-tekrar testi)
(C) Aday on ek ciftlerinin (08<->09 vb.) kare duzeyinde karsilastirmasi

Girdi : outputs/logs/frame_site_assignment_25_26.csv  (10)
        outputs/logs/images_table.csv                  (05)
        outputs/logs/frame_best_matches.csv            (09)
        data/processed/thumbs_*.npy, thumbs_index.csv  (07)
Cikti : outputs/logs/frame_site_groups.csv
        outputs/logs/night_pairs_fullres.csv
        outputs/figures/fig18_candidate_prefix_pairs.png

Kullanim: python scripts/11_groups_and_duplicates.py
          python scripts/11_groups_and_duplicates.py --pairs 08:09 01:04
"""
import argparse
import hashlib
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
MIN_GAP = 100


def zrows(a):
    a = a.astype(np.float32)
    a -= a.mean(axis=1, keepdims=True)
    a /= np.maximum(a.std(axis=1, keepdims=True), 1e-6)
    return a


def grad_mag(T):
    gy, gx = np.gradient(T.astype(np.float32), axis=(1, 2))
    return np.hypot(gx, gy)


def gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.int16)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def show_img(row, size=(320, 256)):
    with Image.open(ROOT / "visible" / row["split"] / f"{row['stem']}.jpg") as im:
        im.draft("RGB", size)
        a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return np.power(a, 0.6)  # gama YALNIZCA gosterim icin


def pct(x, qs=(5, 50, 95)):
    x = np.asarray(x, dtype=float)
    return {f"p{q}": round(float(np.percentile(x, q)), 4) for q in qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["08:09", "01:04", "04:07", "14:15"])
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    t0 = time.time()

    # ================================================================== (A)
    a = pd.read_csv(LOG / "frame_site_assignment_25_26.csv",
                    dtype={"stem": str, "prefix": str, "same_best_prefix": str,
                           "same_best_stem": str})
    srt = a.sort_values("same_best_sim").reset_index(drop=True)
    vals = srt["same_best_sim"].to_numpy()
    k = int(np.diff(vals).argmax())
    lo, hi = float(vals[k]), float(vals[k + 1])
    thr = (lo + hi) / 2
    print("=== (A) BENZERLIK BOSLUGU (25+26 karelerinin same_best_sim degerleri) ===")
    print(f"en buyuk bosluk: {lo:.3f} -> {hi:.3f}  (esik = orta nokta {thr:.3f}); "
          f"altinda {k + 1} kare, ustunde {len(vals) - k - 1} kare")
    print("\nBoslugun iki yanindaki kareler:")
    print(srt.iloc[max(0, k - 4):k + 6][["stem", "prefix", "rgb_mean", "is_day",
                                          "same_best_prefix", "same_best_sim"]]
          .to_string(index=False))

    a["night_origin"] = a["same_best_sim"] >= thr
    print("\nOn ek x (parlaklik kurali: is_day) x (benzerlik kurali: night_origin):")
    print(a.groupby(["prefix", "is_day", "night_origin"]).size().rename("n")
          .reset_index().to_string(index=False))
    dis = a[a["is_day"] == a["night_origin"]]
    print(f"\nIki kuralin celistigi kare sayisi: {len(dis)}")
    if len(dis):
        print(dis[["stem", "prefix", "rgb_mean", "is_day", "same_best_prefix",
                   "same_best_sim"]].head(30).to_string(index=False))

    imgs = pd.read_csv(LOG / "images_table.csv", dtype={"stem": str, "prefix": str})
    g = imgs[["stem", "split", "prefix"]].copy()
    mp = {}
    for r in a.itertuples():
        mp[r.stem] = ((r.same_best_prefix, "frame_match") if r.night_origin
                      else (r.prefix + "D", "own_day_scene"))
    g["site_group"] = [mp.get(s, (p, "prefix"))[0] for s, p in zip(g["stem"], g["prefix"])]
    g["assigned_by"] = [mp.get(s, (p, "prefix"))[1] for s, p in zip(g["stem"], g["prefix"])]
    g.to_csv(LOG / "frame_site_groups.csv", index=False)
    print("\n=== GRUP BUYUKLUKLERI (split, site_group, kare) ===")
    sz = g.groupby(["split", "site_group"]).size().rename("n").reset_index()
    print(sz.to_string(index=False))
    print("\nassigned_by:", g["assigned_by"].value_counts().to_dict())

    # ================================================================== (B)
    print("\n=== (B) GECE KARELERI: ESLESTIKLERI KAREYLE TAM COZUNURLUKTE KARSILASTIRMA ===")
    nm = a[a["night_origin"]]
    rows = []
    for r in nm.itertuples():
        p1 = ROOT / "visible" / r.split / f"{r.stem}.jpg"
        p2 = ROOT / "visible" / r.split / f"{r.same_best_stem}.jpg"
        d = np.abs(gray(p1) - gray(p2))
        rows.append({"stem": r.stem, "prefix": r.prefix, "match_stem": r.same_best_stem,
                     "match_prefix": r.same_best_prefix, "thumb_sim": r.same_best_sim,
                     "mad": float(d.mean()), "p99_absdiff": float(np.percentile(d, 99)),
                     "identical_bytes": sha(p1) == sha(p2)})
    nb = pd.DataFrame(rows)
    nb.to_csv(LOG / "night_pairs_fullres.csv", index=False)

    idx_t = pd.read_csv(CACHE / "thumbs_index.csv", dtype={"stem": str, "prefix": str})
    cons, far = [], []
    for p in ["12", "16", "17"]:                       # hareketsiz on ekler
        sub = idx_t[idx_t["prefix"] == p].reset_index(drop=True)
        for i in rng.choice(len(sub) - 1, 60, replace=False):
            s1, s2 = sub.iloc[i], sub.iloc[i + 1]
            diff_c = np.abs(
                gray(ROOT / "visible" / s1["split"] / f"{s1['stem']}.jpg")
                - gray(ROOT / "visible" / s2["split"] / f"{s2['stem']}.jpg")
            )
            cons.append(float(diff_c.mean()))
        for _ in range(60):
            i = int(rng.integers(0, len(sub)))
            cand = np.where(np.abs(np.arange(len(sub)) - i) >= MIN_GAP)[0]
            j = int(rng.choice(cand))
            s1, s2 = sub.iloc[i], sub.iloc[j]
            diff_f = np.abs(
                gray(ROOT / "visible" / s1["split"] / f"{s1['stem']}.jpg")
                - gray(ROOT / "visible" / s2["split"] / f"{s2['stem']}.jpg")
            )
            far.append(float(diff_f.mean()))

    print("Ortalama mutlak fark (0-255 gri; kucuk = benzer):")
    for pfx, grp in nb.groupby("prefix"):
        print(f"  {pfx} gece->eslesen kare  (n={len(grp)}): {pct(grp['mad'])}")
    print(f"  REFERANS ardisik kareler (12,16,17; n={len(cons)}): {pct(cons)}")
    print(f"  REFERANS uzak kareler   (12,16,17; n={len(far)}): {pct(far)}")
    ref95 = float(np.percentile(cons, 95))
    print(f"\nGece->eslesen cifte mad <= ardisik-referans p95 ({ref95:.2f}) olan kare: "
          f"{int((nb['mad'] <= ref95).sum())}/{len(nb)}")
    print(f"Dosya baytlari birebir ayni olan cift: {int(nb['identical_bytes'].sum())}/{len(nb)}")
    print("Eslesen on ek dagilimi icinde en kucuk 10 mad:")
    print(nb.nsmallest(10, "mad")[["stem", "match_stem", "thumb_sim", "mad",
                                    "identical_bytes"]].to_string(index=False))

    # ================================================================== (C)
    print("\n=== (C) ADAY ON EK CIFTLERI: KARE DUZEYINDE EN YAKIN ESLESME ===")
    T_rgb, T_ir = np.load(CACHE / "thumbs_rgb.npy"), np.load(CACHE / "thumbs_ir.npy")
    N, D = len(idx_t), T_rgb.shape[1] * T_rgb.shape[2]
    F = {"RGB_z": zrows(T_rgb.reshape(N, -1)), "IR_z": zrows(T_ir.reshape(N, -1)),
         "RGB_edge": zrows(grad_mag(T_rgb).reshape(N, -1)),
         "IR_edge": zrows(grad_mag(T_ir).reshape(N, -1))}
    pref = idx_t["prefix"].to_numpy()

    def comb(ia, ib):
        return sum((F[k][ia] @ F[k][ib].T) / D for k in F) / len(F)

    def best(ia, ib, same=False):
        m = comb(ia, ib)
        if same:
            near = np.abs(np.arange(len(ia))[:, None] - np.arange(len(ib))[None, :]) < MIN_GAP
            m[near] = -np.inf
        j = m.argmax(axis=1)
        return m[np.arange(len(ia)), j], j

    bm = pd.read_csv(LOG / "frame_best_matches.csv", dtype={"test_prefix": str})
    base = bm[bm["test_prefix"] != "26"]["sim_comb"]
    print(f"TABAN (test->train, 09): p50={base.median():.3f}, p99={base.quantile(.99):.3f}, "
          f"max={base.max():.3f}")

    fig, axes = plt.subplots(2 * len(args.pairs), 6, figsize=(21, 6.4 * len(args.pairs)))
    for ax in axes.ravel():
        ax.axis("off")
    for pi, pr in enumerate(args.pairs):
        pa, pb = pr.split(":")
        ia, ib = np.where(pref == pa)[0], np.where(pref == pb)[0]
        s_ab, j_ab = best(ia, ib)
        s_ba, _ = best(ib, ia)
        r_a, _ = best(ia, ia, same=True)
        r_b, _ = best(ib, ib, same=True)
        print(f"\n[{pa} <-> {pb}]")
        print(f"  {pa}->{pb}: {pct(s_ab)}   {pb}->{pa}: {pct(s_ba)}")
        print(f"  referans (ayni on ek, >=100 kare uzak)  {pa}: {pct(r_a)}   {pb}: {pct(r_b)}")
        order = np.argsort(s_ab)[::-1]
        med = np.argsort(np.abs(s_ab - np.median(s_ab)))
        picks = [("en yuksek", i) for i in order[:3]] + [("medyan", i) for i in med[:3]]
        for c, (tag, i) in enumerate(picks):
            ra, rb = idx_t.iloc[ia[i]], idx_t.iloc[ib[j_ab[i]]]
            axes[2 * pi, c].imshow(show_img(ra))
            axes[2 * pi, c].set_title(f"{pa}: {ra['stem']} ({tag})", fontsize=9)
            axes[2 * pi + 1, c].imshow(show_img(rb))
            axes[2 * pi + 1, c].set_title(f"{pb}: {rb['stem']}  sim={s_ab[i]:.2f}", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "fig18_candidate_prefix_pairs.png", dpi=60)
    plt.close()

    print(f"\nToplam sure: {time.time() - t0:.0f} sn")
    print("CSV: frame_site_groups.csv, night_pairs_fullres.csv;  Gorsel: fig18")


if __name__ == "__main__":
    main()