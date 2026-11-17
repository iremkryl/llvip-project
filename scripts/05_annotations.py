"""05_annotations.py - XML (VOC) anotasyonlarini ayristirir ve dogrular.

Uretilen dosyalar:
  outputs/logs/images_table.csv       -> goruntu cifti basina 1 satir
  outputs/logs/boxes_table.csv        -> kutu basina 1 satir
  outputs/logs/scene_summary.csv      -> on ek (olasi sahne) basina ozet
  outputs/logs/annotation_summary.json
  outputs/figures/fig02_box_overlay.png
"""
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path("data/raw/LLVIP_extracted")
LOG = Path("outputs/logs")
FIG = Path("outputs/figures")
SPLITS = ["train", "test"]


def _text(node, tag):
    if node is None:
        return None
    child = node.find(tag)
    return child.text.strip() if child is not None and child.text else None


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_xml(path: Path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    info = {
        "xml_filename": _text(root, "filename"),
        "xml_w": _num(_text(size, "width")),
        "xml_h": _num(_text(size, "height")),
        "root_tags": {c.tag for c in root},
    }
    objs, obj_tags = [], set()
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        objs.append({
            "name": _text(obj, "name"),
            "xmin": _num(_text(bb, "xmin")),
            "ymin": _num(_text(bb, "ymin")),
            "xmax": _num(_text(bb, "xmax")),
            "ymax": _num(_text(bb, "ymax")),
        })
        obj_tags |= {c.tag for c in obj}
    return info, objs, obj_tags


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    # Ilk XML'in ham halini goster (yapi kontrolu)
    first_xml = sorted((ROOT / "Annotations").glob("*.xml"))[0]
    print(f"=== HAM XML ORNEGI: {first_xml.name} ===")
    print(first_xml.read_text(encoding="utf-8")[:1500])
    print("=" * 40)

    img_rows, box_rows = [], []
    parse_errors, all_root_tags, all_obj_tags = [], set(), set()

    for split in SPLITS:
        stems = sorted(p.stem for p in (ROOT / "visible" / split).glob("*.jpg"))
        for s in tqdm(stems, desc=f"{split}"):
            rgb_p = ROOT / "visible" / split / f"{s}.jpg"
            ir_p = ROOT / "infrared" / split / f"{s}.jpg"
            xml_p = ROOT / "Annotations" / f"{s}.xml"

            # Goruntu basliklari (piksel verisini okumadan boyut ve mod)
            with Image.open(rgb_p) as im:
                rgb_w, rgb_h, rgb_mode = im.size[0], im.size[1], im.mode
            with Image.open(ir_p) as im:
                ir_w, ir_h, ir_mode = im.size[0], im.size[1], im.mode

            try:
                info, objs, obj_tags = parse_xml(xml_p)
            except Exception as e:  # bozuk XML
                parse_errors.append({"stem": s, "error": repr(e)})
                continue

            all_root_tags |= info["root_tags"]
            all_obj_tags |= obj_tags

            img_rows.append({
                "stem": s, "split": split,
                "prefix": s[:2], "idx_in_prefix": int(s[2:]),
                "rgb_w": rgb_w, "rgb_h": rgb_h, "rgb_mode": rgb_mode,
                "ir_w": ir_w, "ir_h": ir_h, "ir_mode": ir_mode,
                "xml_w": info["xml_w"], "xml_h": info["xml_h"],
                "xml_filename": info["xml_filename"],
                "n_boxes": len(objs),
            })
            for o in objs:
                box_rows.append({"stem": s, "split": split, "prefix": s[:2], **o})

    imgs = pd.DataFrame(img_rows)
    boxes = pd.DataFrame(box_rows)

    # --- Kutu turetilmis ozellikleri ve gecerlilik bayraklari ---
    boxes["w"] = boxes["xmax"] - boxes["xmin"]
    boxes["h"] = boxes["ymax"] - boxes["ymin"]
    boxes["area"] = boxes["w"] * boxes["h"]
    boxes = boxes.merge(imgs[["stem", "rgb_w", "rgb_h"]], on="stem", how="left")
    coords = ["xmin", "ymin", "xmax", "ymax"]
    boxes["flag_missing_coord"] = boxes[coords].isna().any(axis=1)
    boxes["flag_nonpositive"] = (boxes["w"] <= 0) | (boxes["h"] <= 0)
    boxes["flag_out_of_bounds"] = (
        (boxes["xmin"] < 0) | (boxes["ymin"] < 0)
        | (boxes["xmax"] > boxes["rgb_w"]) | (boxes["ymax"] > boxes["rgb_h"])
    )

    imgs.to_csv(LOG / "images_table.csv", index=False)
    boxes.to_csv(LOG / "boxes_table.csv", index=False)

    # --- Sahne (on ek) ozeti ---
    scene = (
        imgs.groupby(["prefix", "split"])
        .agg(n_images=("stem", "count"), n_boxes=("n_boxes", "sum"),
             mean_boxes=("n_boxes", "mean"), max_boxes=("n_boxes", "max"),
             zero_box_images=("n_boxes", lambda x: int((x == 0).sum())))
        .reset_index()
    )
    scene.to_csv(LOG / "scene_summary.csv", index=False)

    # --- Ozet ---
    fn_mismatch = int((imgs["xml_filename"].fillna("").str.replace(
        r"\.[A-Za-z]+$", "", regex=True) != imgs["stem"]).sum())
    summary = {
        "n_image_pairs_parsed": int(len(imgs)),
        "xml_parse_errors": parse_errors,
        "n_boxes_total": int(len(boxes)),
        "n_boxes_by_split": boxes["split"].value_counts().to_dict(),
        "class_names": boxes["name"].value_counts(dropna=False).to_dict(),
        "images_with_zero_boxes": int((imgs["n_boxes"] == 0).sum()),
        "boxes_per_image": imgs["n_boxes"].describe().round(3).to_dict(),
        "rgb_sizes": {f"{k[0]}x{k[1]}": int(v) for k, v in
                      imgs.groupby(["rgb_w", "rgb_h"]).size().items()},
        "ir_sizes": {f"{k[0]}x{k[1]}": int(v) for k, v in
                     imgs.groupby(["ir_w", "ir_h"]).size().items()},
        "xml_sizes": {f"{k[0]}x{k[1]}": int(v) for k, v in
                      imgs.groupby(["xml_w", "xml_h"]).size().items()},
        "rgb_modes": imgs["rgb_mode"].value_counts().to_dict(),
        "ir_modes": imgs["ir_mode"].value_counts().to_dict(),
        "rgb_size_ne_ir_size": int(((imgs.rgb_w != imgs.ir_w) | (imgs.rgb_h != imgs.ir_h)).sum()),
        "xml_size_ne_rgb_size": int(((imgs.xml_w != imgs.rgb_w) | (imgs.xml_h != imgs.rgb_h)).sum()),
        "xml_filename_tag_mismatch": fn_mismatch,
        "boxes_missing_coord": int(boxes["flag_missing_coord"].sum()),
        "boxes_nonpositive_size": int(boxes["flag_nonpositive"].sum()),
        "boxes_out_of_bounds": int(boxes["flag_out_of_bounds"].sum()),
        "xml_root_tags_seen": sorted(all_root_tags),
        "xml_object_tags_seen": sorted(all_obj_tags),
        "box_w_describe": boxes["w"].describe().round(2).to_dict(),
        "box_h_describe": boxes["h"].describe().round(2).to_dict(),
    }
    (LOG / "annotation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))

    print("\n=== SAHNE (ON EK) OZETI ===")
    print(scene.round(2).to_string(index=False))

    # --- Gorsel kanit: kutular RGB ve IR uzerine ---
    random.seed(42)
    tr = imgs[(imgs.split == "train") & (imgs.n_boxes > 0)]["stem"].tolist()
    te = imgs[(imgs.split == "test") & (imgs.n_boxes > 0)]["stem"].tolist()
    crowded = imgs.sort_values("n_boxes", ascending=False).iloc[0]
    picks = [("train", random.choice(tr)), ("train", random.choice(tr)),
             ("test", random.choice(te)), (crowded["split"], crowded["stem"])]

    fig, axes = plt.subplots(2, 4, figsize=(18, 6))
    for j, (split, s) in enumerate(picks):
        b = boxes[boxes["stem"] == s]
        for i, folder in enumerate(["visible", "infrared"]):
            im = Image.open(ROOT / folder / split / f"{s}.jpg")
            ax = axes[i, j]
            ax.imshow(im, cmap="gray" if im.mode == "L" else None)
            for _, r in b.iterrows():
                ax.add_patch(patches.Rectangle(
                    (r.xmin, r.ymin), r.w, r.h,
                    fill=False, edgecolor="red", linewidth=1.3))
            ax.set_title(f"{'RGB' if i == 0 else 'IR'} {split}/{s} ({len(b)} boxes)",
                         fontsize=9)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIG / "fig02_box_overlay_en.png", dpi=150)
    plt.close(fig)
    print("\nGorsel kaydedildi: outputs/figures/fig02_box_overlay_en.png")
    print("Secilen:", picks)


if __name__ == "__main__":
    main()