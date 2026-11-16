"""
ADIM 14B - 40 GORUNTULUK MANUEL DENETIM SECIMI

AMAC:
Manuel denetim suresini azaltmak icin 40 RGB-IR
goruntu cifti secmek. 37 goruntude bir, diger
uc goruntude iki, uc ve dort insan kutusu bulunur.

YONTEM:
1. Temiz anotasyon ve kalite bayraklari okunur.
2. Tek kutulu 27 goruntu rastgele secilir.
3. Iki, uc ve dort kutulu birer goruntu secilir.
4. Kalite bayrakli 10 tek kutulu goruntu eklenir.
5. Tekrar eden goruntuler engellenir.
6. Denetim listesi ayri CSV olarak kaydedilir.

GIRDI:
data/processed/quality_flags.csv
outputs/logs/frame_site_groups.csv

CIKTI:
outputs/logs/manual_audit_40.csv
Terminal: Secim ve kutu sayisi ozeti.
"""

from pathlib import Path
import pandas as pd

LOG = Path("outputs/logs")
PROCESSED = Path("data/processed")

df = pd.read_csv(
    PROCESSED / "quality_flags.csv",
    dtype={"stem": str, "prefix": str}
)

groups = pd.read_csv(
    LOG / "frame_site_groups.csv",
    dtype={"stem": str, "site_group": str}
)

df["stem"] = df["stem"].str.zfill(6)
groups["stem"] = groups["stem"].str.zfill(6)

df = df.merge(
    groups[["stem", "split", "site_group"]],
    on=["stem", "split"],
    how="left",
    validate="one_to_one"
)

if df["site_group"].isna().any():
    raise ValueError("Eksik sahne grubu bulundu.")

df["clean_box_count"] = pd.to_numeric(
    df["clean_box_count"]
)

df["flag_any_review"] = (
    df["flag_any_review"]
    .astype(str)
    .str.lower()
    .eq("true")
)

# 1. Rastgele tek kutulu 27 goruntu
one = df[df["clean_box_count"] == 1]

random_one = one.sample(
    n=27, random_state=42
).copy()

random_one["selection_group"] = "random"

selected = [random_one]

# 2. Iki, uc ve dort kutulu birer goruntu
for n, seed in [(2, 43), (3, 44), (4, 45)]:

    candidates = df[
        df["clean_box_count"] == n
    ]

    if candidates.empty:
        raise ValueError(
            f"{n} kutulu goruntu bulunamadi."
        )

    picked = candidates.sample(
        n=1, random_state=seed
    ).copy()

    picked["selection_group"] = "random"

    selected.append(picked)

# 3. Kalite bayrakli 10 tek kutulu goruntu
already = pd.concat(
    selected, ignore_index=True
)

remaining = one[
    ~one.set_index(["split", "stem"]).index.isin(
        already.set_index(["split", "stem"]).index
    )
]

flagged = remaining[
    remaining["flag_any_review"]
]

if len(flagged) < 10:
    raise ValueError(
        "Yeterli tek kutulu kalite bayrakli "
        "goruntu bulunamadi."
    )

targeted = flagged.sample(
    n=10, random_state=46
).copy()

targeted["selection_group"] = "quality_flagged"

selected.append(targeted)

# 4. Tum secimleri birlestir
sample = pd.concat(
    selected, ignore_index=True
)

sample = sample.sample(
    frac=1, random_state=47
).reset_index(drop=True)

sample.insert(
    0, "audit_id",
    range(1, len(sample) + 1)
)

# 5. Son kontroller
counts = (
    sample["clean_box_count"]
    .value_counts()
    .to_dict()
)

expected = {1: 37, 2: 1, 3: 1, 4: 1}

assert counts == expected, (
    f"Kutu sayisi dagilimi hatali: {counts}"
)

assert len(sample) == 40
assert sample[["split", "stem"]].duplicated().sum() == 0
assert sample["clean_box_count"].sum() == 46

# 6. Manuel denetim bilgileri
columns = [
    "audit_id",
    "stem",
    "split",
    "prefix",
    "site_group",
    "clean_box_count",
    "selection_group",
    "flag_any_review",
    "flag_low_rgb",
    "flag_ir_intensity_tail",
    "flag_invalid_box_removed",
    "flag_duplicate_rgb",
    "flag_duplicate_ir"
]

sample = sample[columns]

output = LOG / "manual_audit_40.csv"

if output.exists():
    raise FileExistsError(
        f"{output} zaten var. "
        "Mevcut denetim listesini korumak icin "
        "uzerine yazilmadi."
    )

sample.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("\n===== ADIM 14B - SECIM OZETI =====")
print(f"Secilen goruntu: {len(sample)}")
print(f"Toplam insan kutusu: {sample['clean_box_count'].sum()}")

print("\nKutu sayisi dagilimi:")
print(
    sample["clean_box_count"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nTrain/Test dagilimi:")
print(
    sample["split"]
    .value_counts()
    .to_string()
)

print("\nSecim turu:")
print(
    sample["selection_group"]
    .value_counts()
    .to_string()
)

print(f"\nCSV: {output}")
print("\nSECIM TAMAMLANDI.")
