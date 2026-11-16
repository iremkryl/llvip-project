"""08_site_analysis.py - Onek homojenligi ve onekler arasi 'ayni yer' benzerligi.

Girdi : data/processed/thumbs_rgb.npy, thumbs_ir.npy, thumbs_index.csv (07 onbellegi)
Cikti : outputs/logs/prefix_homogeneity.csv
        outputs/logs/prefix_distance_{rgb,ir,combined}.csv
        outputs/logs/closest_prefix_pairs.csv
        outputs/logs/test_nearest_train.csv
        outputs/logs/consecutive_spikes.csv
        outputs/figures/fig09_prefix_distance_heatmaps.png
        outputs/figures/fig10_prefix_dendrogram.png
        outputs/figures/fig11_test_vs_nearest_train.png

Mesafe = 1 - Pearson korelasyonu (goruntu basina z-skor -> parlakliktan bagimsiz).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform

ROOT = Path("data/raw/LLVIP_extracted")
LOG = Path("outputs/logs")
FIG = Path("outputs/figures")
CACHE = Path("data/processed")


def zscore_rows(a: np.ndarray) -> np.ndarray:
    """Her satiri (goruntuyu) ortalama 0, std 1 olacak sekilde olcekler."""
    a = a.astype(np.float32)
    a = a - a.mean(axis=1, keepdims=True)
    s = a.std(axis=1, keepdims=True)
    return a / np.maximum(s, 1e-6)


def median_bg(Z: np.ndarray) -> np.ndarray:
    """Z-skorlu karelerin piksel bazinda medyani (arka plan), tekrar z-skorlanir."""
    return zscore_rows(np.median(Z, axis=0, keepdims=True))[0]


def corr_dist(B: np.ndarray) -> np.ndarray:
    """B: (P, n) z-skorlu satirlar -> (P, P) mesafe = 1 - korelasyon."""
    M = 1.0 - np.clip((B @ B.T) / B.shape[1], -1.0, 1.0)
    M = np.maximum((M + M.T) / 2.0, 0.0)
    np.fill_diagonal(M, 0.0)
    return M


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv(CACHE / "thumbs_index.csv", dtype={"stem": str, "prefix": str})
    T = {"RGB": np.load(CACHE / "thumbs_rgb.npy"), "IR": np.load(CACHE / "thumbs_ir.npy")}
    assert len(idx) == len(T["RGB"]) == len(T["IR"]), "Onbellek ile indeks uyusmuyor."

    prefixes = sorted(idx["prefix"].unique())
    pos = {p: np.where(idx["prefix"].values == p)[0] for p in prefixes}  # zaman sirali
    split_of = idx.groupby("prefix")["split"].first().to_dict()
    labels = [p + ("*" if split_of[p] == "test" else "") for p in prefixes]  # * = test
    stems = idx["stem"].values
    rng = np.random.default_rng(42)

    bgs = {"RGB": [], "IR": []}
    hom_rows, spike_rows = [], []
    base_rand = {"RGB": {}, "IR": {}}
    base_time = {"RGB": {}, "IR": {}}

    for mod in ["RGB", "IR"]:
        for p in prefixes:
            ids = pos[p]
            Z = zscore_rows(T[mod][ids].reshape(len(ids), -1))
            nf = Z.shape[1]
            b = median_bg(Z)
            bgs[mod].append(b)

            own = (Z @ b) / nf                                  # kare ~ kendi arka plani
            perm = rng.permutation(len(ids))
            h = len(ids) // 2
            d_rand = 1.0 - float(median_bg(Z[perm[:h]]) @ median_bg(Z[perm[h:]]) / nf)
            d_time = 1.0 - float(median_bg(Z[:h]) @ median_bg(Z[h:]) / nf)
            base_rand[mod][p], base_time[mod][p] = d_rand, d_time

            cons = 1.0 - (Z[:-1] * Z[1:]).sum(axis=1) / nf       # ardisik kare mesafesi
            hom_rows.append({
                "prefix": p, "split": split_of[p], "modality": mod, "n": len(ids),
                "own_corr_median": round(float(np.median(own)), 3),
                "own_corr_p05": round(float(np.percentile(own, 5)), 3),
                "dist_random_halves": round(d_rand, 3),
                "dist_time_halves": round(d_time, 3),
                "consec_median": round(float(np.median(cons)), 4),
                "consec_p99": round(float(np.percentile(cons, 99)), 4),
                "consec_max": round(float(cons.max()), 4),
            })
            for k in np.argsort(cons)[::-1][:3]:
                spike_rows.append({
                    "modality": mod, "prefix": p,
                    "after_stem": stems[ids[k]], "before_stem": stems[ids[k + 1]],
                    "dist": round(float(cons[k]), 4),
                    "x_median": round(float(cons[k] / max(np.median(cons), 1e-6)), 1),
                })

    hom = pd.DataFrame(hom_rows)
    hom.to_csv(LOG / "prefix_homogeneity.csv", index=False)
    spikes = pd.DataFrame(spike_rows).sort_values("x_median", ascending=False)
    spikes.to_csv(LOG / "consecutive_spikes.csv", index=False)

    # ---- onekler arasi mesafe matrisleri ----
    D = {m: corr_dist(np.stack(bgs[m])) for m in ["RGB", "IR"]}
    D["COMB"] = (D["RGB"] + D["IR"]) / 2.0
    for key, name in [("RGB", "rgb"), ("IR", "ir"), ("COMB", "combined")]:
        pd.DataFrame(D[key], index=prefixes, columns=prefixes).round(4).to_csv(
            LOG / f"prefix_distance_{name}.csv")

    # ---- en yakin onek ciftleri (tum splitler) ----
    pairs = []
    for i in range(len(prefixes)):
        for j in range(i + 1, len(prefixes)):
            pairs.append({"a": labels[i], "b": labels[j],
                          "d_comb": round(float(D["COMB"][i, j]), 3),
                          "d_rgb": round(float(D["RGB"][i, j]), 3),
                          "d_ir": round(float(D["IR"][i, j]), 3)})
    closest = pd.DataFrame(pairs).sort_values("d_comb").reset_index(drop=True)
    closest.to_csv(LOG / "closest_prefix_pairs.csv", index=False)

    # ---- her test oneki icin en yakin train oneki ----
    train_i = [i for i, p in enumerate(prefixes) if split_of[p] == "train"]
    rows = []
    for i, p in enumerate(prefixes):
        if split_of[p] != "test":
            continue
        rec = {"test_prefix": p}
        for key in ["COMB", "RGB", "IR"]:
            j = min(train_i, key=lambda t: D[key][i, t])
            rec[f"nearest_train_{key.lower()}"] = prefixes[j]
            rec[f"d_{key.lower()}"] = round(float(D[key][i, j]), 3)
        rec["own_random_halves_comb"] = round(
            (base_rand["RGB"][p] + base_rand["IR"][p]) / 2, 3)
        rec["own_time_halves_comb"] = round(
            (base_time["RGB"][p] + base_time["IR"][p]) / 2, 3)
        rows.append(rec)
    nn = pd.DataFrame(rows)
    nn.to_csv(LOG / "test_nearest_train.csv", index=False)

    # ---- Gorseller ----
    Zl = linkage(squareform(D["COMB"], checks=False), method="average")
    order = leaves_list(Zl)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5))
    for ax, key, title in zip(axes, ["RGB", "IR", "COMB"],
                              ["RGB (z-skor, arka plan korelasyonu)", "IR", "Birlesik (ortalama)"]):
        M = D[key][np.ix_(order, order)]
        im = ax.imshow(M, cmap="viridis_r")
        ax.set_xticks(range(len(order)))
        ax.set_yticks(range(len(order)))
        ax.set_xticklabels([labels[i] for i in order], rotation=90, fontsize=8)
        ax.set_yticklabels([labels[i] for i in order], fontsize=8)
        ax.set_title(f"{title}: onek mesafesi (koyu = benzer, * = test)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(FIG / "fig09_prefix_distance_heatmaps.png", dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Zl, labels=labels, ax=ax, leaf_font_size=10)
    ax.set_title("Onekler arasi hiyerarsik kumeleme (birlesik mesafe, ortalama baglanti)")
    ax.set_ylabel("mesafe (1 - korelasyon)")
    plt.tight_layout()
    plt.savefig(FIG / "fig10_prefix_dendrogram.png", dpi=130)
    plt.close()

    def mid_frame(p: str):
        ids = pos[p]
        r = idx.iloc[ids[len(ids) // 2]]
        with Image.open(ROOT / "visible" / r["split"] / f"{r['stem']}.jpg") as im:
            im.draft("RGB", (320, 256))
            a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        return np.power(a, 0.6), r["stem"]   # gama yalnizca gosterim icin

    fig, axes = plt.subplots(2, len(nn), figsize=(3.6 * len(nn), 6), squeeze=False)
    for j, r in nn.iterrows():
        for i, (pp, tag) in enumerate([(r["test_prefix"], "TEST"),
                                       (r["nearest_train_comb"], "en yakin TRAIN")]):
            img, st = mid_frame(pp)
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            extra = f"\nd={r['d_comb']}" if i == 1 else ""
            axes[i, j].set_title(f"{tag} on ek {pp} ({st}){extra}", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig11_test_vs_nearest_train.png", dpi=130)
    plt.close()

    # ---- Ekrana yazdir ----
    print("=== ON EK HOMOJENLIGI (own_corr: kare-kendi arka plani; dist_*_halves: yari-yari mesafe) ===")
    print(hom.to_string(index=False))
    print("\n=== EN YAKIN 15 ON EK CIFTI (tum splitler, * = test) ===")
    print(closest.head(15).to_string(index=False))
    print("\n=== HER TEST ONEKI ICIN EN YAKIN TRAIN ONEKI ===")
    print(nn.to_string(index=False))
    print("\n=== ARDISIK KARE MESAFESINDE EN BUYUK 15 SICRAMA ===")
    print(spikes.head(15).to_string(index=False))
    print("\nGorseller: fig09, fig10, fig11 (outputs/figures/)")


if __name__ == "__main__":
    main()