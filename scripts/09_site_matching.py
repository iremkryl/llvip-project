"""09_site_matching.py - (A) 25/26 on eklerinin ic yapisi, (B) kare duzeyinde 'ayni yer' eslestirmesi.

Girdi : data/processed/thumbs_rgb.npy, thumbs_ir.npy, thumbs_index.csv (07 onbellegi)
        outputs/logs/quality_pass.csv (parlaklik)
Cikti : outputs/logs/site_sim_<ozellik>.csv    (on ek x on ek; a'daki karelerin b'deki en benzer
                                                kareye benzerliginin medyani; kosegen = ayni on ekte
                                                >=100 kare uzaktaki en benzer kare = referans)
        outputs/logs/test_to_train_site_match.csv
        outputs/logs/top_prefix_pairs_comb.csv
        outputs/logs/frame_best_matches.csv
        outputs/figures/fig12_prefix_anatomy.png
        outputs/figures/fig13_prefix25_26_contact.png
        outputs/figures/fig14_site_similarity_heatmap.png
        outputs/figures/fig15_test_train_best_pairs.png

Kullanim: python scripts/09_site_matching.py
          python scripts/09_site_matching.py --features IR_z RGB_edge   (daha hizli)
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
MIN_GAP = 100      # ayni on ekte 'uzak kare' esigi (kare numarasi farki)
CHUNK = 500        # matris carpimi icin satir bloku


def zrows(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    a -= a.mean(axis=1, keepdims=True)
    a /= np.maximum(a.std(axis=1, keepdims=True), 1e-6)
    return a


def grad_mag(T: np.ndarray) -> np.ndarray:
    a = T.astype(np.float32)
    gy, gx = np.gradient(a, axis=(1, 2))
    return np.hypot(gx, gy)


def show_img(row, folder_name="visible", size=(320, 256)):
    with Image.open(ROOT / folder_name / row["split"] / f"{row['stem']}.jpg") as im:
        im.draft("RGB", size)
        a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return np.power(a, 0.6)        # gama YALNIZCA gosterim icin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+",
                    default=["RGB_z", "IR_z", "RGB_edge", "IR_edge"],
                    choices=["RGB_z", "IR_z", "RGB_edge", "IR_edge"])
    args = ap.parse_args()
    names = args.features
    FIG.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv(CACHE / "thumbs_index.csv", dtype={"stem": str, "prefix": str})
    assert (idx.sort_values(["prefix", "idx_in_prefix"]).index == idx.index).all(), \
        "Indeks on ek/sira ile siralanmis olmali (07 betigi boyle uretiyor)."
    T_rgb, T_ir = np.load(CACHE / "thumbs_rgb.npy"), np.load(CACHE / "thumbs_ir.npy")
    N = len(idx)
    D = T_rgb.shape[1] * T_rgb.shape[2]

    prefixes = sorted(idx["prefix"].unique())
    P = len(prefixes)
    pref = idx["prefix"].values
    starts = np.array([np.where(pref == p)[0][0] for p in prefixes])
    ends = np.array([np.where(pref == p)[0][-1] + 1 for p in prefixes])
    split_of = idx.groupby("prefix")["split"].first().to_dict()
    labels = [p + ("*" if split_of[p] == "test" else "") for p in prefixes]
    is_train = idx["split"].values == "train"

    # ------------------------------------------------------------------ (A)
    qp = pd.read_csv(LOG / "quality_pass.csv", dtype={"stem": str})
    br = idx.merge(qp[["stem", "rgb_mean"]], on="stem", how="left")["rgb_mean"].values
    Zrgb = zrows(T_rgb.reshape(N, -1))

    fig, axes = plt.subplots(2, 3, figsize=(18, 6), sharex="col")
    for j, p in enumerate(["12", "25", "26"]):
        ai = prefixes.index(p)
        s, e = starts[ai], ends[ai]
        cons = 1.0 - (Zrgb[s:e - 1] * Zrgb[s + 1:e]).sum(axis=1) / D
        axes[0, j].plot(br[s:e], lw=0.8)
        axes[0, j].set_title(f"on ek {p}: RGB ortalama parlaklik (kare sirasina gore)")
        axes[1, j].plot(cons, lw=0.8, color="tab:red")
        axes[1, j].set_title(f"on ek {p}: ardisik kare mesafesi (1-korelasyon)")
        axes[1, j].set_xlabel("on ek icindeki kare sirasi")
    plt.tight_layout()
    plt.savefig(FIG / "fig12_prefix_anatomy.png", dpi=130)
    plt.close()

    fig, axes = plt.subplots(4, 6, figsize=(21, 12))
    for ax in axes.ravel():
        ax.axis("off")
    for blk, p in enumerate(["25", "26"]):
        ai = prefixes.index(p)
        ids = np.linspace(starts[ai], ends[ai] - 1, 12).astype(int)
        for k, i in enumerate(ids):
            ax = axes[blk * 2 + k // 6, k % 6]
            ax.imshow(show_img(idx.iloc[i]))
            ax.set_title(f"{idx.iloc[i]['stem']}  (sira {i - starts[ai]})", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "fig13_prefix25_26_contact.png", dpi=110)
    plt.close()
    print("(A) tamam: fig12, fig13")

    # ------------------------------------------------------------------ (B)
    F = {}
    if "RGB_z" in names:
        F["RGB_z"] = Zrgb
    if "IR_z" in names:
        F["IR_z"] = zrows(T_ir.reshape(N, -1))
    if "RGB_edge" in names:
        F["RGB_edge"] = zrows(grad_mag(T_rgb).reshape(N, -1))
    if "IR_edge" in names:
        F["IR_edge"] = zrows(grad_mag(T_ir).reshape(N, -1))
    keys = names + ["COMB"]

    M = {k: np.zeros((N, P), np.float32) for k in keys}   # kare -> on ek en yuksek benzerlik
    best_j = np.full(N, -1, np.int64)
    best_s = np.full(N, np.nan, np.float32)

    t0 = time.time()
    for ai, a in enumerate(prefixes):
        sa, ea = starts[ai], ends[ai]
        for r0 in range(sa, ea, CHUNK):
            r1 = min(r0 + CHUNK, ea)
            sims = {k: (F[k][r0:r1] @ F[k].T) / D for k in names}
            sims["COMB"] = sum(sims[k] for k in names) / len(names)
            near = np.abs(np.arange(r0, r1)[:, None] - np.arange(sa, ea)[None, :]) < MIN_GAP
            for k in keys:
                sims[k][:, sa:ea][near] = -np.inf       # ayni on ekte yakin kareleri disla
            for k in keys:
                for bi in range(P):
                    M[k][r0:r1, bi] = sims[k][:, starts[bi]:ends[bi]].max(axis=1)
            if split_of[a] == "test":
                c = np.where(is_train[None, :], sims["COMB"], -np.inf)
                j = c.argmax(axis=1)
                best_j[r0:r1] = j
                best_s[r0:r1] = c[np.arange(r1 - r0), j]
        print(f"  on ek {a} bitti  ({time.time() - t0:.0f} sn)")

    S = {k: np.zeros((P, P)) for k in keys}
    for ai in range(P):
        for k in keys:
            S[k][ai] = np.median(M[k][starts[ai]:ends[ai]], axis=0)
    for k in keys:
        pd.DataFrame(S[k], index=labels, columns=labels).round(4).to_csv(
            LOG / f"site_sim_{k}.csv")

    # test -> en yakin train on eki
    train_p = [i for i, p in enumerate(prefixes) if split_of[p] == "train"]
    rows = []
    for ai, a in enumerate(prefixes):
        if split_of[a] != "test":
            continue
        for k in keys:
            bi = max(train_p, key=lambda t: S[k][ai, t])
            ref = S[k][ai, ai]
            rows.append({"test_prefix": a, "feature": k,
                         "best_train_prefix": prefixes[bi],
                         "site_sim": round(float(S[k][ai, bi]), 3),
                         "same_prefix_ref": round(float(ref), 3),
                         "ratio": round(float(S[k][ai, bi] / ref), 2) if ref > 0 else np.nan})
    tt = pd.DataFrame(rows)
    tt.to_csv(LOG / "test_to_train_site_match.csv", index=False)

    # tum on ek ciftleri (simetrik)
    C = S["COMB"]
    pairs = []
    for i in range(P):
        for j in range(i + 1, P):
            sym = (C[i, j] + C[j, i]) / 2
            ref = min(C[i, i], C[j, j])
            pairs.append({"a": labels[i], "b": labels[j], "sym_site_sim": round(float(sym), 3),
                          "ref_min": round(float(ref), 3),
                          "ratio": round(float(sym / ref), 2) if ref > 0 else np.nan})
    pairs = pd.DataFrame(pairs).sort_values("sym_site_sim", ascending=False)
    pairs.to_csv(LOG / "top_prefix_pairs_comb.csv", index=False)

    fm = pd.DataFrame({
        "test_stem": idx["stem"].values[~is_train],
        "test_prefix": idx["prefix"].values[~is_train],
        "train_stem": idx["stem"].values[best_j[~is_train]],
        "train_prefix": idx["prefix"].values[best_j[~is_train]],
        "sim_comb": best_s[~is_train]})
    fm.to_csv(LOG / "frame_best_matches.csv", index=False)

    # ----- gorseller
    fig, ax = plt.subplots(figsize=(10, 8.5))
    Csym = (C + C.T) / 2
    im = ax.imshow(Csym, cmap="viridis")
    ax.set_xticks(range(P)); ax.set_yticks(range(P))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Kare duzeyinde 'ayni yer' benzerligi (COMB, medyan; * = test)\n"
                 "kosegen = ayni on ekte >=100 kare uzaktaki en benzer kare (referans)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(FIG / "fig14_site_similarity_heatmap.png", dpi=130)
    plt.close()

    test_prefs = [p for p in prefixes if split_of[p] == "test"]
    fig, axes = plt.subplots(4, len(test_prefs), figsize=(3.4 * len(test_prefs), 11),
                             squeeze=False)
    for j, p in enumerate(test_prefs):
        g = fm[fm["test_prefix"] == p].reset_index(drop=True)
        picks = [("en yuksek", g["sim_comb"].idxmax()),
                 ("medyan", (g["sim_comb"] - g["sim_comb"].median()).abs().idxmin())]
        for k, (tag, gi) in enumerate(picks):
            r = g.iloc[gi]
            t_row = idx[idx["stem"] == r["test_stem"]].iloc[0]
            m_row = idx[idx["stem"] == r["train_stem"]].iloc[0]
            axes[2 * k, j].imshow(show_img(t_row))
            axes[2 * k, j].set_title(f"TEST {r['test_stem']} ({tag})", fontsize=8)
            axes[2 * k + 1, j].imshow(show_img(m_row))
            axes[2 * k + 1, j].set_title(f"TRAIN {r['train_stem']}  sim={r['sim_comb']:.2f}", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIG / "fig15_test_train_best_pairs.png", dpi=110)
    plt.close()

    # ----- ekrana yazdir
    print("\n=== TEST -> EN YAKIN TRAIN ON EKI (ratio = site_sim / ayni-on-ek referansi) ===")
    print(tt.to_string(index=False))
    print("\n=== EN BENZER 20 ON EK CIFTI (COMB, simetrik) ===")
    print(pairs.head(20).to_string(index=False))
    print("\n=== TEST KARELERI: en yakin train karesine COMB benzerligi (on ek bazinda) ===")
    print(fm.groupby("test_prefix")["sim_comb"].describe().round(3).to_string())
    print(f"\nToplam sure: {time.time() - t0:.0f} sn")
    print("Gorseller: fig12-fig15 (outputs/figures/)")


if __name__ == "__main__":
    main()