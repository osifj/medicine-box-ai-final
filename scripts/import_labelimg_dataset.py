#!/usr/bin/env python3
"""Import old LabelImg YOLO annotations into this project's class schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from PIL import Image


CURRENT_CLASSES = ["medicine_box", "barcode", "text"]


def read_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def map_label_line(line: str, old_classes: list[str]) -> str | None:
    parts = line.split()
    if len(parts) != 5:
        return None
    old_id = int(float(parts[0]))
    if old_id < 0 or old_id >= len(old_classes):
        return None
    name = old_classes[old_id]
    if name not in CURRENT_CLASSES:
        return None
    new_id = CURRENT_CLASSES.index(name)
    return " ".join([str(new_id), *parts[1:]])


def split_items(items: list[Path], val_fraction: float) -> tuple[list[Path], list[Path]]:
    ordered = sorted(items, key=lambda path: path.name)
    val_count = max(1, round(len(ordered) * val_fraction)) if ordered else 0
    val_names = {item.name for index, item in enumerate(ordered) if index % max(1, len(ordered) // val_count) == 0}
    train = [item for item in ordered if item.name not in val_names]
    val = [item for item in ordered if item.name in val_names]
    return train, val


def copy_split(
    label_files: list[Path],
    split: str,
    source_images: Path,
    output: Path,
    old_classes: list[str],
) -> dict[str, Any]:
    image_out = output / "images" / split
    label_out = output / "labels" / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    imported = 0
    boxes = 0
    skipped = 0
    class_counts = {name: 0 for name in CURRENT_CLASSES}

    for label_file in label_files:
        image_file = source_images / f"{label_file.stem}.jpg"
        if not image_file.exists():
            image_file = source_images / f"{label_file.stem}.png"
        if not image_file.exists():
            skipped += 1
            continue

        mapped: list[str] = []
        for line in label_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            converted = map_label_line(line, old_classes)
            if converted is None:
                continue
            mapped.append(converted)
            class_counts[CURRENT_CLASSES[int(converted.split()[0])]] += 1

        if not mapped:
            skipped += 1
            continue

        dst_image = image_out / image_file.name
        shutil.copy2(image_file, dst_image)
        (label_out / f"{label_file.stem}.txt").write_text("\n".join(mapped) + "\n", encoding="utf-8")
        imported += 1
        boxes += len(mapped)

    return {
        "split": split,
        "images": imported,
        "boxes": boxes,
        "class_counts": class_counts,
        "skipped": skipped,
    }


def write_data_yaml(output: Path) -> None:
    yaml = "\n".join(
        [
            "path: .",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: medicine_box",
            "  1: barcode",
            "  2: text",
            "",
        ]
    )
    (output / "data.yaml").write_text(yaml, encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(CURRENT_CLASSES) + "\n", encoding="utf-8")


def import_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).expanduser().resolve()
    source_images = source / "images"
    source_labels = source / args.labels_dir
    old_classes = read_classes(source_labels / "classes.txt")
    output = Path(args.output).expanduser().resolve()
    if output.exists() and args.clean:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    label_files = sorted(path for path in source_labels.glob("*.txt") if path.name != "classes.txt")
    train_files, val_files = split_items(label_files, args.val_fraction)
    train_summary = copy_split(train_files, "train", source_images, output, old_classes)
    val_summary = copy_split(val_files, "val", source_images, output, old_classes)
    write_data_yaml(output)

    manifest = {
        "source": str(source),
        "source_labels": str(source_labels),
        "source_classes": old_classes,
        "target_classes": CURRENT_CLASSES,
        "class_mapping": {str(i): CURRENT_CLASSES.index(name) for i, name in enumerate(old_classes) if name in CURRENT_CLASSES},
        "warning": "Imported labels are old LabelImg annotations. Current source contains text boxes only; medicine_box/barcode may be absent.",
        "splits": [train_summary, val_summary],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import old LabelImg YOLO annotations")
    parser.add_argument("--source", default="/Users/dep/Projects/New_project/captures/host_session_001")
    parser.add_argument("--labels-dir", default="labels")
    parser.add_argument("--output", default="data/real_labelimg_yolo")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--clean", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = import_dataset(args)
    print(f"Imported LabelImg dataset: {args.output}")
    for split in manifest["splits"]:
        print(
            f"  {split['split']}: images={split['images']} boxes={split['boxes']} "
            f"class_counts={split['class_counts']} skipped={split['skipped']}"
        )
    print("  data.yaml:", str(Path(args.output) / "data.yaml"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
