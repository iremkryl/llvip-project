"""12_duplicates_and_calibration.py

(A) Tum dosyalarin SHA-256 ozeti -> birebir kopya taramasi (RGB ve IR ayri)
(B) Gece-eslesme ciftleri: ayni mad degerine sahip ciftler (anomali) + dogru referans
    (eslesen karenin +-1 komsusu ve ayni on ekteki >=100 kare uzak kare)
(C) Gunduz-gece kalibrasyonu: 25D ve 26D karelerinin IR/kenar benzerligi, tabanla karsilastirma

Girdi : outputs/logs/images_table.csv, night_pairs_fullres.csv (11), frame_site_groups.csv (11)
        data/processed/thumbs_*.npy, thumbs_index.csv (07)
Cikti : outputs/logs/file_hashes.csv, night_ref_check.csv, daynight_calibration.csv
        outputs/figures/fig19_daynight_calibration.png
"""
import hashlib
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path("data/raw/LLVIP_extracted")
LOG = Path("outputs/logs")
FIG = Path("outputs/figures")
CACHE = Path("data/processed")
MIN_GAP = 100


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.int16)


def mad(a, b):
    return float(np.abs(a - b).mean())


def pct(x, qs=(5, 50, 95)):
    return {f"p{q}": round(float(np.percentile(np.asarray(x, float), q)), 3) for q in qs}


def zrows(a):
    a = a.astype(np.float32)
    a -= a.mean(axis=1, keepdims=True)
    a /= np.maximum(a.std(axis=1, keepdims=True), 1e-6)
    return a


def grad_mag(T):
    gy, gx = np.gradient(T.astype(np.float32), axis=(1, 2))
    return np.hypot(gx, gy)


def main():
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    imgs = pd.read_csv(LOG / "images_table.csv", dtype={"stem": str, "prefix": str})
    split_of = dict(zip(imgs["stem"], imgs["split"]))
    stem_set = set(imgs["stem"])

    # ============================================================ (A)
    print("=== (A) BAYT DUZEYINDE KOPYA TARAMASI ===")
    hp = LOG / "file_hashes.csv"
    if hp.exists():
        hh = pd.read_csv(hp, dtype={"stem": str})
    else:
        rows = []
        for sp, st in tqdm(zip(imgs["split"], imgs["stem"]), total=len(imgs), desc="SHA-256"):
            rows.append({"stem": st, "split": sp,
                         "rgb_sha": sha(ROOT / "visible" / sp / f"{st}.jpg"),
                         "ir_sha": sha(ROOT / "infrared" / sp / f"{st}.jpg")})
        hh = pd.DataFrame(rows)
        hh.to_csv(hp, index=False)
    for col, name in [("rgb_sha", "RGB"), ("ir_sha", "IR")]:
        vc = hh[col].value_counts()
        dup = vc[vc > 1].index
        d = hh[hh[col].isin(dup)]
        print(f"{name}: {len(dup)} tekrar grubu, {len(d)} dosya")
        if len(d):
            g = d.groupby(col).agg(n=("stem", "count"),
                                   stems=("stem", lambda s: ",".join(s.head(6))),
                                   splits=("split", lambda s: "+".join(sorted(set(s)))))
            print(g.sort_values("n", ascending=False).head(15).to_string())
            other = "ir_sha" if col == "rgb_sha" else "rgb_sha"
            n_diff = int((d.groupby(col)[other].nunique() > 1).sum())
            print(f"  bu gruplarin {n_diff} tanesinde diger modalite FARKLI")

    # ============================================================ (B)
    print("\n=== (B1) AYNI mad DEGERI VEREN GECE CIFTLERI (anomali taramasi) ===")
    nb = pd.read_csv(LOG / "night_pairs_fullres.csv",
                     dtype={"stem": str, "prefix": str, "match_stem": str, "match_prefix": str})
    nb["mad_r"] = nb["mad"].round(5)
    same = nb.groupby(["match_stem", "mad_r"]).filter(lambda g: len(g) > 1)
    print(f"ayni (eslesen kare, mad) degerini paylasan cift: {len(same)}")
    if len(same):
        print(same.sort_values(["match_stem", "mad_r"])[
            ["stem", "match_stem", "thumb_sim", "mad"]].head(20).to_string(index=False))
        for (ms, mr), g in list(same.groupby(["match_stem", "mad_r"]))[:8]:
            s1, s2 = g["stem"].iloc[0], g["stem"].iloc[1]
            sp1, sp2 = split_of[s1], split_of[s2]
            rgb1, rgb2 = gray(ROOT / "visible" / sp1 / f"{s1}.jpg"), gray(ROOT / "visible" / sp2 / f"{s2}.jpg")
            ir1, ir2 = gray(ROOT / "infrared" / sp1 / f"{s1}.jpg"), gray(ROOT / "infrared" / sp2 / f"{s2}.jpg")
            print(f"  {s1} vs {s2}: RGB mad={mad(rgb1, rgb2):.4f}, IR mad={mad(ir1, ir2):.4f}, "
                  f"RGB bayt-ayni={sha(ROOT / 'visible' / sp1 / f'{s1}.jpg') == sha(ROOT / 'visible' / sp2 / f'{s2}.jpg')}")

    print("\n=== (B2) DOGRU REFERANS: eslesen karenin kendi on ekindeki komsulari ===")
    rng = np.random.default_rng(42)
    by_prefix = imgs.groupby("prefix")["stem"].apply(list).to_dict()
    rows = []
    for r in tqdm(nb.itertuples(), total=len(nb), desc="Referans"):
        sp = split_of[r.match_stem]
        g0 = gray(ROOT / "visible" / sp / f"{r.match_stem}.jpg")
        nei = [f"{int(r.match_stem) + d:06d}" for d in (-1, 1)]
        nei = [s for s in nei if s in stem_set and s[:2] == r.match_stem[:2]]
        far = [s for s in by_prefix[r.match_prefix]
               if abs(int(s) - int(r.match_stem)) >= MIN_GAP]
        if not nei or not far:
            continue
        ref_adj = np.mean([mad(g0, gray(ROOT / "visible" / split_of[s] / f"{s}.jpg")) for s in nei])
        fs = far[int(rng.integers(0, len(far)))]
        ref_far = mad(g0, gray(ROOT / "visible" / split_of[fs] / f"{fs}.jpg"))
        rows.append({"stem": r.stem, "prefix": r.prefix, "match_stem": r.match_stem,
                     "match_prefix": r.match_prefix, "mad_pair": r.mad,
                     "ref_adj": ref_adj, "ref_far": ref_far})
    rc = pd.DataFrame(rows)
    rc.to_csv(LOG / "night_ref_check.csv", index=False)
    rc["cat"] = np.where(rc["mad_pair"] <= rc["ref_adj"], "eslesme <= yakin komsu",
                 np.where(rc["mad_pair"] <= rc["ref_far"], "yakin komsu < eslesme <= uzak kare",
                          "eslesme > uzak kare"))
    print(f"Referansi hesaplanabilen cift: {len(rc)} / {len(nb)}")
    print(rc.groupby(["prefix", "cat"]).size().unstack(fill_value=0).to_string())
    print("mad_pair:", pct(rc["mad_pair"]), " ref_adj:", pct(rc["ref_adj"]),
          " ref_far:", pct(rc["ref_far"]))
    print("oran mad_pair/ref_adj:", pct(rc["mad_pair"] / rc["ref_adj"]),
          " mad_pair/ref_far:", pct(rc["mad_pair"] / rc["ref_far"]))

    # ============================================================ (C)
    print("\n=== (C) GUNDUZ-GECE KALIBRASYONU (25D, 26D) ===")
    idx = pd.read_csv(CACHE / "thumbs_index.csv", dtype={"stem": str, "prefix": str})
    grp = pd.read_csv(LOG / "frame_site_groups.csv",
                      dtype={"stem": str, "prefix": str, "site_group": str})
    idx = idx.merge(grp[["stem", "site_group"]], on="stem", how="left")
    T_rgb, T_ir = np.load(CACHE / "thumbs_rgb.npy"), np.load(CACHE / "thumbs_ir.npy")
    N, D = len(idx), T_rgb.shape[1] * T_rgb.shape[2]
    F = {"IR_z": zrows(T_ir.reshape(N, -1)),
         "IR_edge": zrows(grad_mag(T_ir).reshape(N, -1)),
         "RGB_edge": zrows(grad_mag(T_rgb).reshape(N, -1))}
    sg, sp = idx["site_group"].to_numpy(), idx["split"].to_numpy()
    q25, q26 = sg == "25D", sg == "26D"
    tr, te = sp == "train", sp == "test"
    base_q = np.where(te & ~q26)[0]          # tabani: gunduz olmayan test kareleri -> train

    def best_to(q, t, f):
        out = []
        for c in range(0, len(q), 500):
            m = (F[f][q[c:c + 500]] @ F[f][t].T) / D
            out.append(m.max(axis=1))
        return np.concatenate(out)

    sets = {
        "26D -> test (26D haric)": (np.where(q26)[0], np.where(te & ~q26)[0]),
        "26D -> train": (np.where(q26)[0], np.where(tr)[0]),
        "25D -> train (25D haric)": (np.where(q25)[0], np.where(tr & ~q25)[0]),
        "25D -> test": (np.where(q25)[0], np.where(te)[0]),
    }
    res, rows = {}, []
    for f in F:
        base = best_to(base_q, np.where(tr)[0], f)
        res[("TABAN: test -> train", f)] = base
        rows.append({"set": "TABAN: test -> train", "feature": f, **pct(base, (50, 95, 99)),
                     "max": round(float(base.max()), 3)})
        for name, (q, t) in sets.items():
            v = best_to(q, t, f)
            res[(name, f)] = v
            rows.append({"set": name, "feature": f, **pct(v, (50, 95, 99)),
                         "max": round(float(v.max()), 3)})
    cal = pd.DataFrame(rows)
    cal.to_csv(LOG / "daynight_calibration.csv", index=False)
    print(cal.to_string(index=False))

    print("\n--- Hedef grup bazinda medyan en iyi benzerlik (IR_edge) ---")
    for name, qm, tmask in [("26D", q26, te & ~q26), ("25D", q25, tr & ~q25),
                            ("25D", q25, te), ("26D", q26, tr)]:
        q = np.where(qm)[0]
        tab = {tg: round(float(np.median(best_to(q, np.where(sg == tg)[0], "IR_edge"))), 3)
               for tg in sorted(set(sg[tmask]))}
        print(f"{name} -> {'test' if tmask[np.where(te)[0][0]] else 'train'} gruplari:", tab)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    bins = np.linspace(0, 1, 51)
    for ax, f in zip(axes, F):
        ax.hist(res[("TABAN: test -> train", f)], bins=bins, density=True, alpha=0.4,
                color="gray", label="taban: test -> train")
        for name, c in [("26D -> test (26D haric)", "tab:orange"), ("25D -> train (25D haric)", "tab:blue"),
                        ("25D -> test", "tab:green"), ("26D -> train", "tab:red")]:
            ax.hist(res[(name, f)], bins=bins, density=True, alpha=0.5, color=c, label=name)
        ax.set_title(f"{f}: kare -> hedefteki en yakin kare")
        ax.set_xlabel("benzerlik")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG / "fig19_daynight_calibration.png", dpi=130)
    plt.close()
    print(f"\nToplam sure: {time.time() - t0:.0f} sn;  CSV: file_hashes, night_ref_check, daynight_calibration")


if __name__ == "__main__":
    main()