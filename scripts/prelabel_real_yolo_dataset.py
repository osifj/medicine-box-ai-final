#!/usr/bin/env python3
"""Add conservative auto labels to imported real LabelImg YOLO data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_real_image_pipeline as real_pipeline  # noqa: E402
import yolo_dataset_utils as yolo  # noqa: E402


def text_union(boxes: list[yolo.YoloBox], image_width: int, image_height: int) -> list[int] | None:
    text_boxes = [yolo.yolo_to_xyxy(box, image_width, image_height) for box in boxes if box.class_id == 2]
    if not text_boxes:
        return None
    x1 = min(box[0] for box in text_boxes)
    y1 = min(box[1] for box in text_boxes)
    x2 = max(box[2] for box in text_boxes)
    y2 = max(box[3] for box in text_boxes)
    return [x1, y1, x2, y2]


def expanded_text_union(boxes: list[yolo.YoloBox], image_width: int, image_height: int) -> list[int] | None:
    raw = text_union(boxes, image_width, image_height)
    if raw is None:
        return None
    x1, y1, x2, y2 = raw
    bw = x2 - x1
    bh = y2 - y1
    return [
        max(0, x1 - int(bw * 0.45) - 32),
        max(0, y1 - int(bh * 0.38) - 30),
        min(image_width, x2 + int(bw * 0.18) + 32),
        min(image_height, y2 + int(bh * 1.05) + 40),
    ]


def color_box_is_plausible(box: list[int], image_width: int, image_height: int) -> bool:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    area_ratio = (bw * bh) / max(1, image_width * image_height)
    if area_ratio > 0.78:
        return False
    if bw > image_width * 0.96 and bh > image_height * 0.82:
        return False
    if x1 <= 2 and y1 <= 2 and x2 >= image_width - 2:
        return False
    return bw >= image_width * 0.25 and bh >= image_height * 0.12


def contains_box(outer: list[int], inner: list[int], margin: int = 12) -> bool:
    return (
        outer[0] <= inner[0] + margin
        and outer[1] <= inner[1] + margin
        and outer[2] >= inner[2] - margin
        and outer[3] >= inner[3] - margin
    )


def estimate_medicine_box(image_path: Path, existing_boxes: list[yolo.YoloBox]) -> tuple[list[int], str, float]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    scale = min(1.0, 640 / max(width, height))
    detector_image = image
    if scale < 1.0:
        detector_image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    arr = np.asarray(detector_image)
    candidate = real_pipeline.choose_medicine_box(arr)
    color_box = None
    if candidate:
        inv = 1.0 / scale
        color_box = [int(round(value * inv)) for value in candidate["bbox"]]
        color_box = [
            max(0, min(width - 1, color_box[0])),
            max(0, min(height - 1, color_box[1])),
            max(1, min(width, color_box[2])),
            max(1, min(height, color_box[3])),
        ]
        if not color_box_is_plausible(color_box, width, height):
            color_box = None
    color_score = float(candidate.get("score", 0.0)) if candidate else 0.0
    raw_text_box = text_union(existing_boxes, width, height)
    text_box = expanded_text_union(existing_boxes, width, height)

    if color_box and text_box:
        if raw_text_box and contains_box(color_box, raw_text_box):
            return color_box, "color_segmentation_contains_text", color_score
        # Use visual box as anchor, but ensure all human text labels remain inside.
        merged = [
            max(0, min(color_box[0], text_box[0])),
            max(0, min(color_box[1], text_box[1])),
            min(width, max(color_box[2], text_box[2])),
            min(height, max(color_box[3], text_box[3])),
        ]
        return merged, "color_plus_text_union", color_score
    if color_box:
        return color_box, "color_segmentation", color_score
    if text_box:
        return text_box, "expanded_text_union", 0.35
    return [int(width * 0.08), int(height * 0.14), int(width * 0.92), int(height * 0.84)], "central_fallback", 0.2


def maybe_detect_barcode(image_path: Path, medicine_xyxy: list[int]) -> list[yolo.YoloBox]:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image)
    height, width = arr.shape[:2]
    detections = real_pipeline.detect_barcode_regions(arr, medicine_xyxy)
    return [yolo.xyxy_to_yolo(det["bbox_xyxy"], width, height, 1) for det in detections if det["score"] >= 0.75]


def link_or_copy_split_images(src_dir: Path, dst_dir: Path, mode: str) -> None:
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    if dst_dir.exists() or dst_dir.is_symlink():
        if dst_dir.is_symlink() or dst_dir.is_file():
            dst_dir.unlink()
        else:
            shutil.rmtree(dst_dir)
    if mode == "symlink":
        target = os.path.relpath(src_dir.resolve(), dst_dir.parent.resolve())
        dst_dir.symlink_to(target, target_is_directory=True)
    else:
        shutil.copytree(src_dir, dst_dir)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def prelabel_dataset(args: argparse.Namespace) -> dict[str, Any]:
    input_yaml = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and args.clean:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    config = yolo.parse_data_yaml(input_yaml)

    summaries: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        image_dir = yolo.split_image_dir(input_yaml, split, config)
        label_dir = yolo.split_label_dir(input_yaml, split, config)
        if image_dir is None or label_dir is None or not image_dir.exists():
            continue
        output_image_dir = output / "images" / split
        output_label_dir = output / "labels" / split
        link_or_copy_split_images(image_dir, output_image_dir, args.image_mode)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        added_medicine = 0
        existing_medicine = 0
        added_barcode = 0
        images = yolo.image_files(input_yaml, split, config)
        method_counts: dict[str, int] = {}
        for image_path in images:
            source_label = label_dir / f"{image_path.stem}.txt"
            boxes = yolo.read_yolo_boxes(source_label)
            with Image.open(image_path) as image:
                width, height = image.size
            out_boxes = list(boxes)
            medicine_xyxy: list[int] | None = None
            if any(box.class_id == 0 for box in boxes) and not args.overwrite_medicine_box:
                existing_medicine += 1
                medicine_xyxy = yolo.yolo_to_xyxy(next(box for box in boxes if box.class_id == 0), width, height)
            else:
                medicine_xyxy, method, score = estimate_medicine_box(image_path, boxes)
                if score >= args.min_medicine_score or method != "central_fallback":
                    out_boxes = [box for box in out_boxes if box.class_id != 0]
                    out_boxes.insert(0, yolo.xyxy_to_yolo(medicine_xyxy, width, height, 0))
                    added_medicine += 1
                    method_counts[method] = method_counts.get(method, 0) + 1

            if args.add_barcode and medicine_xyxy is not None and not any(box.class_id == 1 for box in out_boxes):
                barcode_boxes = maybe_detect_barcode(image_path, medicine_xyxy)
                if barcode_boxes:
                    out_boxes.extend(barcode_boxes)
                    added_barcode += len(barcode_boxes)

            lines = [yolo.format_yolo_line(box) for box in out_boxes]
            (output_label_dir / f"{image_path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        summaries.append(
            {
                "split": split,
                "images": len(images),
                "existing_medicine_box": existing_medicine,
                "added_medicine_box": added_medicine,
                "added_barcode": added_barcode,
                "medicine_box_methods": method_counts,
                "image_mode": args.image_mode,
            }
        )

    data_yaml = yolo.write_data_yaml(output, include_test=(output / "images" / "test").exists())
    manifest = {
        "source_data_yaml": display_path(input_yaml),
        "data_yaml": display_path(data_yaml),
        "target_classes": yolo.CLASS_NAMES,
        "policy": (
            "medicine_box labels are auto-prelabels from color/text geometry; "
            "barcode labels are added only when --add-barcode is set and detector confidence is high."
        ),
        "splits": summaries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-prelabel imported real YOLO data")
    parser.add_argument("--input", default="data/real_labelimg_yolo/data.yaml")
    parser.add_argument("--output", default="data/real_labelimg_yolo_prelabelled")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--image-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--overwrite-medicine-box", action="store_true")
    parser.add_argument("--add-barcode", action="store_true")
    parser.add_argument("--min-medicine-score", type=float, default=0.18)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = prelabel_dataset(args)
    print(f"Prelabelled YOLO: {manifest['data_yaml']}")
    for split in manifest["splits"]:
        print(
            f"  {split['split']}: images={split['images']} "
            f"added_medicine_box={split['added_medicine_box']} added_barcode={split['added_barcode']} "
            f"methods={split['medicine_box_methods']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
