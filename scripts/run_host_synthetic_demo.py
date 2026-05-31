#!/usr/bin/env python3
"""Self-contained host-side synthetic medicine-box demo.

This script intentionally avoids Grove hardware and external project paths. It
generates synthetic medicine-box images, calibrates a tiny color/geometry
detector from generated labels, then runs inference and writes JSON + overlays.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import random
from statistics import median
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REQUIRED_LABELS = ("medicine_box", "barcode", "text")
IMAGE_SIZE = (900, 620)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clamp_box(box: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    return [max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)]


def xyxy_to_xywh(box: list[int]) -> list[int]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def box_area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def iou(a: list[int], b: list[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = box_area([x1, y1, x2, y2])
    union = box_area(a) + box_area(b) - inter
    return inter / union if union else 0.0


def connected_components(mask: np.ndarray, min_area: int) -> list[dict[str, Any]]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[dict[str, Any]] = []
    ys, xs = np.nonzero(mask)

    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue

        q: deque[tuple[int, int]] = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        area = 0
        min_x = max_x = start_x
        min_y = max_y = start_y

        while q:
            y, x = q.popleft()
            area += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and mask[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    q.append((ny, nx))

        if area >= min_area:
            components.append({"bbox": [min_x, min_y, max_x + 1, max_y + 1], "area": area})

    return sorted(components, key=lambda item: item["area"], reverse=True)


def draw_background(draw: ImageDraw.ImageDraw, rng: random.Random, width: int, height: int) -> None:
    base = rng.choice([(198, 208, 218), (205, 198, 188), (190, 211, 205)])
    draw.rectangle((0, 0, width, height), fill=base)
    for _ in range(16):
        tone = rng.randint(-15, 18)
        color = tuple(max(120, min(230, ch + tone)) for ch in base)
        x = rng.randint(-80, width)
        y = rng.randint(-60, height)
        draw.ellipse((x, y, x + rng.randint(80, 220), y + rng.randint(50, 160)), fill=color)


def generate_image(path: Path, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    width, height = IMAGE_SIZE
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    draw_background(draw, rng, width, height)

    box_w = rng.randint(560, 680)
    box_h = rng.randint(315, 390)
    x1 = rng.randint(70, width - box_w - 70)
    y1 = rng.randint(70, height - box_h - 70)
    x2 = x1 + box_w
    y2 = y1 + box_h

    body = (246, 249, 255)
    accent = rng.choice([(33, 86, 160), (40, 128, 90), (180, 72, 88)])
    barcode_fill = (212, 250, 214)
    text_fill = (255, 236, 170)

    draw.rounded_rectangle((x1 + 8, y1 + 10, x2 + 8, y2 + 10), radius=18, fill=(75, 75, 75))
    draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=body, outline=accent, width=6)
    draw.rectangle((x1, y1, x2, y1 + 46), fill=accent)
    draw.rectangle((x1, y2 - 34, x2, y2), fill=accent)

    font = ImageFont.load_default()
    draw.text((x1 + 24, y1 + 16), "SYNTHETIC MEDICINE", fill=(255, 255, 255), font=font)

    labels: list[dict[str, Any]] = [
        {
            "label": "medicine_box",
            "bbox_xyxy": [x1, y1, x2, y2],
            "text": "whole package",
        }
    ]

    text_specs = [
        ("IBUPROFEN CAPSULES", x1 + 34, y1 + 78, 300, 50),
        ("EXP 2028-05  LOT A17", x1 + 34, y1 + 148, 270, 44),
        ("DOSAGE: 1 capsule", x1 + 34, y1 + 212, 235, 42),
    ]
    for content, tx, ty, tw, th in text_specs:
        draw.rounded_rectangle((tx, ty, tx + tw, ty + th), radius=6, fill=text_fill, outline=(150, 112, 50), width=2)
        draw.text((tx + 12, ty + 15), content, fill=(45, 45, 45), font=font)
        labels.append({"label": "text", "bbox_xyxy": [tx, ty, tx + tw, ty + th], "text": content})

    bx2 = x2 - 48
    by2 = y2 - 64
    bx1 = bx2 - 205
    by1 = by2 - 74
    draw.rounded_rectangle((bx1, by1, bx2, by2), radius=5, fill=barcode_fill, outline=(72, 128, 78), width=2)
    cursor = bx1 + 12
    while cursor < bx2 - 12:
        bar_w = rng.choice([2, 3, 4, 6])
        gap = rng.choice([2, 3, 5])
        draw.rectangle((cursor, by1 + 10, cursor + bar_w, by2 - 16), fill=(20, 20, 20))
        cursor += bar_w + gap
    draw.text((bx1 + 18, by2 - 14), "6901234567890", fill=(20, 20, 20), font=font)
    labels.append({"label": "barcode", "bbox_xyxy": [bx1, by1, bx2, by2], "text": "6901234567890"})

    image.save(path, quality=94)
    return {"image": str(path), "width": width, "height": height, "labels": labels}


def generate_dataset(root: Path, count: int, seed: int, prefix: str) -> list[dict[str, Any]]:
    images_dir = ensure_dir(root / "images")
    records = []
    for index in range(count):
        image_path = images_dir / f"{prefix}_{index:03d}.jpg"
        records.append(generate_image(image_path, seed + index))
    (root / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def class_masks(arr: np.ndarray) -> dict[str, np.ndarray]:
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    return {
        "medicine_box": (r > 238) & (g > 238) & (b > 238),
        "barcode": (g > 238) & (r < 230) & (b < 230),
        "text": (r > 240) & (g > 220) & (g < 245) & (b < 210),
    }


def raw_detect(image_path: Path, model: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image)
    height, width = arr.shape[:2]
    masks = class_masks(arr)
    detections: list[dict[str, Any]] = []
    model = model or {}
    pads = model.get("medicine_padding", [0, 0, 0, 0])

    medicine_components = connected_components(masks["medicine_box"], min_area=12_000)
    if medicine_components:
        box = medicine_components[0]["bbox"]
        box = [box[0] - pads[0], box[1] - pads[1], box[2] + pads[2], box[3] + pads[3]]
        detections.append(
            {
                "label": "medicine_box",
                "score": 0.99,
                "bbox_xyxy": clamp_box([int(v) for v in box], width, height),
                "source": "synthetic_color_profile",
            }
        )

    for item in connected_components(masks["barcode"], min_area=2_000):
        detections.append(
            {
                "label": "barcode",
                "score": 0.98,
                "bbox_xyxy": clamp_box(item["bbox"], width, height),
                "source": "synthetic_color_profile",
            }
        )

    for item in connected_components(masks["text"], min_area=1_000):
        detections.append(
            {
                "label": "text",
                "score": 0.97,
                "bbox_xyxy": clamp_box(item["bbox"], width, height),
                "source": "synthetic_color_profile",
            }
        )

    for det in detections:
        det["bbox_xywh"] = xyxy_to_xywh(det["bbox_xyxy"])
    return detections


def train_detector(records: list[dict[str, Any]], model_path: Path) -> dict[str, Any]:
    left_pads: list[int] = []
    top_pads: list[int] = []
    right_pads: list[int] = []
    bottom_pads: list[int] = []

    for record in records:
        raw = [d for d in raw_detect(Path(record["image"]), model={"medicine_padding": [0, 0, 0, 0]}) if d["label"] == "medicine_box"]
        gt = next(item for item in record["labels"] if item["label"] == "medicine_box")["bbox_xyxy"]
        if not raw:
            continue
        raw_box = raw[0]["bbox_xyxy"]
        left_pads.append(max(0, raw_box[0] - gt[0]))
        top_pads.append(max(0, raw_box[1] - gt[1]))
        right_pads.append(max(0, gt[2] - raw_box[2]))
        bottom_pads.append(max(0, gt[3] - raw_box[3]))

    model = {
        "name": "synthetic_color_geometry_detector",
        "classes": list(REQUIRED_LABELS),
        "medicine_padding": [
            int(median(left_pads or [0])),
            int(median(top_pads or [0])),
            int(median(right_pads or [0])),
            int(median(bottom_pads or [0])),
        ],
        "training_images": len(records),
        "note": "Calibrated from generated labels; intended for deterministic host-side demo images.",
    }
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    return model


def match_generated_labels(detections: list[dict[str, Any]], labels: list[dict[str, Any]]) -> None:
    for det in detections:
        candidates = [item for item in labels if item["label"] == det["label"]]
        if not candidates:
            continue
        best = max(candidates, key=lambda item: iou(det["bbox_xyxy"], item["bbox_xyxy"]))
        score = iou(det["bbox_xyxy"], best["bbox_xyxy"])
        det["matched_generated_iou"] = round(score, 3)
        if score > 0.3 and best.get("text"):
            det["matched_generated_text"] = best["text"]


def draw_overlay(image_path: Path, detections: list[dict[str, Any]], out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {"medicine_box": (0, 120, 255), "barcode": (30, 170, 70), "text": (230, 150, 0)}
    font = ImageFont.load_default()
    for det in detections:
        box = det["bbox_xyxy"]
        label = f"{det['label']} {det['score']:.2f}"
        color = colors.get(det["label"], (255, 0, 0))
        draw.rectangle(box, outline=color, width=4)
        tx, ty = box[0], max(0, box[1] - 18)
        tw = max(90, len(label) * 7)
        draw.rectangle((tx, ty, tx + tw, ty + 17), fill=color)
        draw.text((tx + 4, ty + 3), label, fill=(255, 255, 255), font=font)
    image.save(out_path, quality=94)


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    output = ensure_dir(Path(args.output))
    train_root = ensure_dir(output / "train")
    demo_root = ensure_dir(output / "demo")
    results_root = ensure_dir(output / "results")
    model_path = output / "synthetic_detector_profile.json"

    train_records = generate_dataset(train_root, args.train_count, args.seed, "train")
    if args.force_train or not model_path.exists():
        model = train_detector(train_records, model_path)
        model_state = "trained"
    else:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        model_state = "loaded"

    demo_records = generate_dataset(demo_root, args.demo_count, args.seed + 10_000, "demo")
    summary: dict[str, Any] = {
        "status": "ok",
        "model_state": model_state,
        "model_path": str(model_path),
        "required_labels": list(REQUIRED_LABELS),
        "results": [],
    }

    for record in demo_records:
        image_path = Path(record["image"])
        detections = raw_detect(image_path, model=model)
        match_generated_labels(detections, record["labels"])
        counts = {label: sum(1 for det in detections if det["label"] == label) for label in REQUIRED_LABELS}
        missing = [label for label, count in counts.items() if count == 0]
        status = "pass" if not missing else "fail"
        if missing:
            summary["status"] = "failed"

        stem = image_path.stem
        json_path = results_root / f"{stem}.json"
        overlay_path = results_root / f"{stem}_overlay.jpg"
        result = {
            "status": status,
            "image": str(image_path),
            "overlay": str(overlay_path),
            "counts": counts,
            "missing": missing,
            "detections": detections,
            "generated_labels": record["labels"],
            "note": "Text/barcode strings are generated ground-truth hints, not live OCR or barcode decoding.",
        }
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        draw_overlay(image_path, detections, overlay_path)
        summary["results"].append({"json": str(json_path), "overlay": str(overlay_path), **result})

        print(
            f"{status.upper()} {image_path.name}: "
            f"medicine_box={counts['medicine_box']} barcode={counts['barcode']} text={counts['text']}"
        )
        print(f"  JSON: {json_path}")
        print(f"  Overlay: {overlay_path}")

    summary_path = results_root / "summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}")
    print(f"Model: {model_path} ({model_state})")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run host-only synthetic medicine-box demo")
    parser.add_argument("--output", default="out/host_synthetic_demo")
    parser.add_argument("--train-count", type=int, default=24)
    parser.add_argument("--demo-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--force-train", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_demo(args)
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
