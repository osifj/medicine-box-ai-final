#!/usr/bin/env python3
"""Audit YOLO labels for the medicine-box host pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yolo_dataset_utils as yolo  # noqa: E402


def validate_box(box: yolo.YoloBox, class_count: int) -> list[str]:
    errors: list[str] = []
    if box.class_id < 0 or box.class_id >= class_count:
        errors.append("invalid_class")
    if not (0.0 <= box.cx <= 1.0 and 0.0 <= box.cy <= 1.0):
        errors.append("center_out_of_bounds")
    if not (0.0 < box.width <= 1.0 and 0.0 < box.height <= 1.0):
        errors.append("size_out_of_bounds")
    x1 = box.cx - box.width / 2
    y1 = box.cy - box.height / 2
    x2 = box.cx + box.width / 2
    y2 = box.cy + box.height / 2
    if x1 < -0.002 or y1 < -0.002 or x2 > 1.002 or y2 > 1.002:
        errors.append("box_extends_outside_image")
    return errors


def audit_split(data_yaml: Path, split: str, config: dict[str, Any]) -> dict[str, Any]:
    names = config["names"]
    class_names = [names.get(index, str(index)) for index in range(max(names) + 1)]
    images = yolo.image_files(data_yaml, split, config)
    labels = yolo.label_files(data_yaml, split, config)
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    class_counts = {name: 0 for name in class_names}
    problems: list[dict[str, Any]] = []
    duplicate_boxes = 0
    box_count = 0
    total_pixels = 0

    for image_path in images:
        try:
            with Image.open(image_path) as image:
                total_pixels += image.size[0] * image.size[1]
        except Exception as exc:
            problems.append({"type": "unreadable_image", "path": str(image_path), "error": str(exc)})

        label_path = yolo.label_path_for_image(data_yaml, split, image_path, config)
        if not label_path.exists():
            problems.append({"type": "missing_label", "image": str(image_path)})
            continue
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            problems.append({"type": "empty_label", "label": str(label_path)})
            continue
        duplicates = sum(count - 1 for count in Counter(lines).values() if count > 1)
        duplicate_boxes += duplicates
        if duplicates:
            problems.append({"type": "duplicate_boxes", "label": str(label_path), "count": duplicates})
        for line_index, line in enumerate(lines, start=1):
            try:
                box = yolo.parse_yolo_line(line)
            except ValueError as exc:
                problems.append({"type": "malformed_line", "label": str(label_path), "line": line_index, "error": str(exc)})
                continue
            box_count += 1
            if 0 <= box.class_id < len(class_names):
                class_counts[class_names[box.class_id]] += 1
            for error in validate_box(box, len(class_names)):
                problems.append({"type": error, "label": str(label_path), "line": line_index, "value": line})

    for label_path in labels:
        if label_path.stem not in image_stems:
            problems.append({"type": "label_without_image", "label": str(label_path)})

    return {
        "split": split,
        "images": len(images),
        "labels": len(labels),
        "boxes": box_count,
        "class_counts": class_counts,
        "missing_labels": len(image_stems - label_stems),
        "labels_without_images": len(label_stems - image_stems),
        "duplicate_boxes": duplicate_boxes,
        "mean_image_pixels": round(total_pixels / len(images), 2) if images else 0,
        "problems": problems,
    }


def audit_dataset(data_yaml: Path) -> dict[str, Any]:
    config = yolo.parse_data_yaml(data_yaml)
    splits = [split for split in ("train", "val", "test") if config.get(split)]
    split_reports = [audit_split(data_yaml, split, config) for split in splits]
    total_problems = sum(len(item["problems"]) for item in split_reports)
    total_boxes = sum(item["boxes"] for item in split_reports)
    totals = {name: 0 for name in yolo.CLASS_NAMES}
    for split in split_reports:
        for name, count in split["class_counts"].items():
            totals[name] = totals.get(name, 0) + count
    return {
        "status": "ok" if total_problems == 0 else "problems_found",
        "data_yaml": str(data_yaml),
        "dataset_root": str(yolo.dataset_root(data_yaml, config)),
        "classes": [config["names"].get(index, str(index)) for index in sorted(config["names"])],
        "splits": split_reports,
        "total_boxes": total_boxes,
        "total_class_counts": totals,
        "total_problems": total_problems,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit YOLO medicine-box dataset")
    parser.add_argument("--data", default="data/real_labelimg_yolo/data.yaml")
    parser.add_argument("--output", default="")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when problems are found")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = audit_dataset(Path(args.data))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Audit: {report['status']} boxes={report['total_boxes']} problems={report['total_problems']}")
    for split in report["splits"]:
        print(
            f"  {split['split']}: images={split['images']} labels={split['labels']} boxes={split['boxes']} "
            f"class_counts={split['class_counts']}"
        )
    if args.output:
        print(f"  report: {args.output}")
    return 1 if args.strict and report["total_problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
