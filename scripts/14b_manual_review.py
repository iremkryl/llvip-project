
"""
ADIM 14B - 40 GORUNTUNUN MANUEL ETIKET DENETIMI

AMAC:
Secilen 40 RGB-IR ciftinde eksik kisi, etiket dogrulugu,
hizalama ve gorunurlugu incelemek.

YONTEM:
1. 14B secim listesi ve temiz insan kutulari okunur.
2. Tam cozunurluklu RGB-IR goruntuler incelenir.
3. Goruntu ve her kutu icin ayri karar verilir.
4. Kararlar ayri CSV'lere kaydedilir ve tekrar acilabilir.

GIRDI:
outputs/logs/manual_audit_40.csv
data/processed/boxes_cleaned.csv
data/raw/LLVIP_extracted/

CIKTI:
outputs/logs/manual_audit_40_image_reviews.csv
outputs/logs/manual_audit_40_box_reviews.csv
"""

from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
from PIL import Image, ImageDraw, ImageTk

ROOT = Path("data/raw/LLVIP_extracted")
LOG = Path("outputs/logs")

IMAGE_CSV = LOG / "manual_audit_40_image_reviews.csv"
BOX_CSV = LOG / "manual_audit_40_box_reviews.csv"

PREVIEW = (465, 372)

FIELD_OPTIONS = {
    "ir_box_correct": ("", "Yes", "No", "Uncertain"),
    "rgb_alignment": (
        "", "Aligned", "Shifted",
        "Not assessable", "Uncertain"
    ),
    "rgb_visibility": (
        "", "Clear", "Difficult",
        "Not visible", "Uncertain"
    ),
    "ir_visibility": (
        "", "Clear", "Difficult",
        "Not visible", "Uncertain"
    ),
}


def read_old(path):
    if not path.exists():
        return []

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False
    ).to_dict("records")


def write_safe(path, records, columns):
    df = pd.DataFrame(records, columns=columns)

    if not df.empty:
        ordering = ["audit_id"]

        if "box_index" in df:
            ordering.append("box_index")

        df = df.sort_values(
            ordering,
            key=lambda s: pd.to_numeric(s)
        )

    temp = path.with_name(path.name + ".tmp")

    df.to_csv(
        temp,
        index=False,
        encoding="utf-8-sig"
    )

    temp.replace(path)


class Reviewer:

    def __init__(self, root):
        self.root = root
        root.title("LLVIP | Manual Audit: 40 Image Pairs")
        root.geometry("1040x840")

        sample = pd.read_csv(
            LOG / "manual_audit_40.csv",
            dtype=str,
            keep_default_na=False
        )

        sample["stem"] = sample["stem"].str.zfill(6)

        sample["audit_id_number"] = (
            sample["audit_id"].astype(int)
        )

        self.sample = sample.sort_values(
            "audit_id_number"
        ).reset_index(drop=True)

        assert len(self.sample) == 40, (
            "Expected 40 image pairs."
        )

        boxes = pd.read_csv(
            Path("data/processed/boxes_cleaned.csv"),
            dtype={"stem": str}
        )

        boxes["stem"] = boxes["stem"].str.zfill(6)

        for col in ["xmin", "ymin", "xmax", "ymax"]:
            boxes[col] = pd.to_numeric(
                boxes[col], errors="raise"
            )

        self.box_table = boxes

        self.image_records = {
            str(r["audit_id"]): r
            for r in read_old(IMAGE_CSV)
        }

        self.box_records = {
            (str(r["audit_id"]), str(r["box_index"])): r
            for r in read_old(BOX_CSV)
        }

        self.position = 0
        self.active_box = 0
        self.focus = (640, 512)

        self.zoom = tk.StringVar(value="1x")
        self.show_boxes = tk.BooleanVar(value=True)

        self.missing = tk.StringVar()
        self.image_note = tk.StringVar()
        self.box_note = tk.StringVar()

        self.box_vars = {
            field: tk.StringVar()
            for field in FIELD_OPTIONS
        }

        self.photos = {}
        self.viewport = {}
        self.images = {}

        self.make_ui()
        self.load(0)

        root.protocol("WM_DELETE_WINDOW", self.exit)

    def make_ui(self):
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill="x")

        self.status = ttk.Label(
            toolbar,
            font=("Arial", 11, "bold")
        )

        self.status.pack(
            side="left", padx=(0, 12)
        )

        ttk.Button(
            toolbar,
            text="Previous",
            command=self.previous
        ).pack(side="left", padx=3)

        ttk.Button(
            toolbar,
            text="Save draft",
            command=self.save
        ).pack(side="left", padx=3)

        ttk.Button(
            toolbar,
            text="Save & Next",
            command=self.next
        ).pack(side="left", padx=3)

        ttk.Label(
            toolbar, text="Zoom"
        ).pack(side="left", padx=(14, 4))

        cb = ttk.Combobox(
            toolbar,
            textvariable=self.zoom,
            values=("1x", "2x", "4x", "8x"),
            width=4,
            state="readonly"
        )

        cb.pack(side="left")

        cb.bind(
            "<<ComboboxSelected>>",
            lambda _: self.render()
        )

        ttk.Checkbutton(
            toolbar,
            text="Show boxes",
            variable=self.show_boxes,
            command=self.render
        ).pack(side="left", padx=12)

        pictures = ttk.Frame(
            self.root, padding=8
        )

        pictures.pack()

        self.canvases = {}

        for modality, label in [
            ("rgb", "RGB (Visible)"),
            ("ir", "IR (Infrared)")
        ]:
            pane = ttk.Frame(pictures)
            pane.pack(side="left", padx=8)

            ttk.Label(
                pane,
                text=label,
                font=("Arial", 11, "bold")
            ).pack()

            canvas = tk.Canvas(
                pane,
                width=PREVIEW[0],
                height=PREVIEW[1],
                highlightthickness=1,
                highlightbackground="#999"
            )

            canvas.pack()

            canvas.bind(
                "<Button-1>",
                lambda event, m=modality:
                    self.pan(event, m)
            )

            self.canvases[modality] = canvas

        ttk.Label(
            self.root,
            text=(
                "Click an image to center the zoom. "
                "Yellow: selected box; green: other boxes."
            )
        ).pack()

        form = ttk.Frame(
            self.root, padding=12
        )

        form.pack(fill="x")

        ttk.Label(
            form,
            text="Image-level: Any person missing a box?"
        ).grid(row=0, column=0, sticky="w")

        ttk.Combobox(
            form,
            textvariable=self.missing,
            values=("", "No", "Yes", "Uncertain"),
            width=17,
            state="readonly"
        ).grid(row=0, column=1, sticky="w", padx=7)

        ttk.Label(
            form, text="Image notes:"
        ).grid(
            row=1, column=0,
            sticky="w", pady=(8, 0)
        )

        ttk.Entry(
            form,
            textvariable=self.image_note,
            width=95
        ).grid(
            row=2, column=0,
            columnspan=5, sticky="we"
        )

        ttk.Separator(form).grid(
            row=3, column=0,
            columnspan=5,
            sticky="ew", pady=12
        )

        ttk.Label(
            form, text="Selected box:"
        ).grid(row=4, column=0, sticky="w")

        self.box_select = ttk.Combobox(
            form,
            state="readonly",
            width=16
        )

        self.box_select.grid(
            row=4, column=1, sticky="w"
        )

        self.box_select.bind(
            "<<ComboboxSelected>>",
            self.select_box
        )

        labels = {
            "ir_box_correct": "IR box covers person?",
            "rgb_alignment": "RGB box alignment?",
            "rgb_visibility": "Person visible in RGB?",
            "ir_visibility": "Person visible in IR?"
        }

        for i, (field, values) in enumerate(
            FIELD_OPTIONS.items()
        ):
            ttk.Label(
                form,
                text=labels[field]
            ).grid(
                row=5, column=i,
                sticky="w",
                padx=5, pady=(12, 3)
            )

            ttk.Combobox(
                form,
                textvariable=self.box_vars[field],
                values=values,
                width=21,
                state="readonly"
            ).grid(
                row=6, column=i,
                sticky="w", padx=5
            )

        ttk.Label(
            form, text="Selected box notes:"
        ).grid(
            row=7, column=0,
            sticky="w", pady=(9, 0)
        )

        ttk.Entry(
            form,
            textvariable=self.box_note,
            width=95
        ).grid(
            row=8, column=0,
            columnspan=5,
            sticky="we"
        )

        ttk.Label(
            self.root,
            text=(
                "Use Uncertain rather than guessing. "
                "For unclear RGB visibility, "
                "RGB alignment can be Not assessable."
            )
        ).pack(pady=8)

        self.progress = ttk.Label(
            self.root,
            font=("Arial", 10, "bold")
        )

        self.progress.pack()

    def load(self, index):
        self.position = index
        self.row = self.sample.iloc[index]

        self.audit_id = str(
            self.row["audit_id"]
        )

        self.split = self.row["split"]
        self.stem = self.row["stem"]

        self.boxes = self.box_table[
            (self.box_table["split"] == self.split) &
            (self.box_table["stem"] == self.stem)
        ].reset_index(drop=True)

        expected = int(
            self.row["clean_box_count"]
        )

        if len(self.boxes) != expected:
            raise ValueError(
                f"Box count mismatch for {self.stem}: "
                f"{len(self.boxes)} vs {expected}"
            )

        self.images = {}

        for modality, folder in [
            ("rgb", "visible"),
            ("ir", "infrared")
        ]:
            path = (
                ROOT / folder / self.split /
                f"{self.stem}.jpg"
            )

            with Image.open(path) as im:
                self.images[modality] = im.convert("RGB")

        assert (
            self.images["rgb"].size
            == self.images["ir"].size
        )

        width, height = self.images["rgb"].size

        self.focus = (
            width / 2, height / 2
        )

        self.zoom.set("1x")

        saved = self.image_records.get(
            self.audit_id, {}
        )

        self.missing.set(
            saved.get("missing_person", "")
        )

        self.image_note.set(
            saved.get("image_notes", "")
        )

        self.box_select["values"] = [
            f"Box {j+1} / {len(self.boxes)}"
            for j in range(len(self.boxes))
        ]

        self.active_box = 0
        self.box_select.current(0)

        self.load_box(0)

        self.status.config(
            text=(
                f"{index+1}/40 | "
                f"{self.split.upper()} {self.stem} | "
                f"{len(self.boxes)} box(es)"
            )
        )

        self.render()
        self.show_progress()

    def load_box(self, idx):
        self.active_box = idx

        record = self.box_records.get(
            (self.audit_id, str(idx)), {}
        )

        for name, var in self.box_vars.items():
            var.set(
                record.get(name, "")
            )

        self.box_note.set(
            record.get("box_notes", "")
        )

        box = self.boxes.iloc[idx]

        self.focus = (
            (box.xmin + box.xmax) / 2,
            (box.ymin + box.ymax) / 2
        )

    def capture_box(self):
        box = self.boxes.iloc[
            self.active_box
        ]

        self.box_records[
            (self.audit_id, str(self.active_box))
        ] = {
            "audit_id": self.audit_id,
            "split": self.split,
            "stem": self.stem,
            "selection_group": self.row["selection_group"],
            "box_index": str(self.active_box),
            "xmin": box.xmin,
            "ymin": box.ymin,
            "xmax": box.xmax,
            "ymax": box.ymax,
            **{
                name: var.get()
                for name, var in self.box_vars.items()
            },
            "box_notes": self.box_note.get().strip()
        }

    def select_box(self, _=None):
        self.capture_box()

        self.load_box(
            self.box_select.current()
        )

        self.render()

    def is_complete(self):
        if not self.missing.get():
            return False

        for idx in range(len(self.boxes)):
            record = self.box_records.get(
                (self.audit_id, str(idx)), {}
            )

            if any(
                not record.get(field)
                for field in FIELD_OPTIONS
            ):
                return False

        return True

    def save(self):
        self.capture_box()
        complete = self.is_complete()

        self.image_records[self.audit_id] = {
            "audit_id": self.audit_id,
            "split": self.split,
            "stem": self.stem,
            "selection_group": self.row["selection_group"],
            "missing_person": self.missing.get(),
            "image_notes": self.image_note.get().strip(),
            "review_status": (
                "complete" if complete else "partial"
            )
        }

        image_fields = [
            "audit_id", "split", "stem",
            "selection_group",
            "missing_person", "image_notes",
            "review_status"
        ]

        box_fields = [
            "audit_id", "split", "stem",
            "selection_group", "box_index",
            "xmin", "ymin", "xmax", "ymax",
            *FIELD_OPTIONS, "box_notes"
        ]

        write_safe(
            IMAGE_CSV,
            list(self.image_records.values()),
            image_fields
        )

        write_safe(
            BOX_CSV,
            list(self.box_records.values()),
            box_fields
        )

        self.show_progress()

        print(
            f"Saved {self.stem}: "
            f"{'complete' if complete else 'partial'}"
        )

        return complete

    def show_progress(self):
        completed = sum(
            r.get("review_status") == "complete"
            for r in self.image_records.values()
        )

        self.progress.config(
            text=(
                f"Completed: {completed} / 40 | "
                "Saves to separate CSV files"
            )
        )

    def next(self):
        if not self.save():
            messagebox.showwarning(
                "Incomplete review",
                (
                    "Fill the image-level answer and "
                    "all four fields for every box. "
                    "Select Uncertain if needed."
                )
            )
            return

        if self.position == len(self.sample) - 1:
            messagebox.showinfo(
                "Finished",
                (
                    "40/40 reached. "
                    "Send the two review CSV "
                    "files for analysis."
                )
            )
        else:
            self.load(
                self.position + 1
            )

    def previous(self):
        self.save()

        if self.position > 0:
            self.load(
                self.position - 1
            )

    def render(self):
        factor = int(
            self.zoom.get().replace("x", "")
        )

        width, height = self.images["rgb"].size

        cw = max(1, width // factor)
        ch = max(1, height // factor)

        left = max(
            0,
            min(
                width - cw,
                int(self.focus[0] - cw / 2)
            )
        )

        top = max(
            0,
            min(
                height - ch,
                int(self.focus[1] - ch / 2)
            )
        )

        for modality in ("rgb", "ir"):
            crop = self.images[modality].crop(
                (left, top, left + cw, top + ch)
            )

            pic = crop.resize(
                PREVIEW,
                Image.Resampling.LANCZOS
            )

            if self.show_boxes.get():
                draw = ImageDraw.Draw(pic)

                for j, box in self.boxes.iterrows():
                    rectangle = [
                        (box.xmin - left) * PREVIEW[0] / cw,
                        (box.ymin - top) * PREVIEW[1] / ch,
                        (box.xmax - left) * PREVIEW[0] / cw,
                        (box.ymax - top) * PREVIEW[1] / ch
                    ]

                    if (
                        rectangle[2] < 0 or
                        rectangle[3] < 0 or
                        rectangle[0] > PREVIEW[0] or
                        rectangle[1] > PREVIEW[1]
                    ):
                        continue

                    color = (
                        "yellow"
                        if j == self.active_box
                        else "lime"
                    )

                    draw.rectangle(
                        rectangle,
                        outline=color,
                        width=3
                    )

                    draw.text(
                        (
                            max(0, rectangle[0]),
                            max(0, rectangle[1] - 13)
                        ),
                        str(j + 1),
                        fill=color
                    )

            photo = ImageTk.PhotoImage(pic)

            self.photos[modality] = photo
            canvas = self.canvases[modality]

            canvas.delete("all")

            canvas.create_image(
                0, 0,
                image=photo,
                anchor="nw"
            )

            self.viewport[modality] = (
                left, top, cw, ch
            )

    def pan(self, event, modality):
        left, top, cw, ch = (
            self.viewport[modality]
        )

        self.focus = (
            left + event.x * cw / PREVIEW[0],
            top + event.y * ch / PREVIEW[1]
        )

        self.render()

    def exit(self):
        answer = messagebox.askyesnocancel(
            "Exit",
            "Save current review before closing?"
        )

        if answer is None:
            return

        if answer:
            self.save()

        self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    Reviewer(window)
    window.mainloop()