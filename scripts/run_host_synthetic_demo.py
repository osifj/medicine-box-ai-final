#!/usr/bin/env python3
"""Self-contained host-side synthetic medicine-box demo.

The default path avoids Grove hardware and external project paths. It generates
synthetic medicine-box images, calibrates a tiny color/geometry detector,
exports a standard YOLO dataset, evaluates on a fixed synthetic test split, and
writes JSON + overlay outputs.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import random
import shutil
from statistics import median
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REQUIRED_LABELS = ("medicine_box", "barcode", "text")
CLASS_TO_ID = {name: index for index, name in enumerate(REQUIRED_LABELS)}
IMAGE_SIZE = (960, 680)

BRAND_TEXTS = ["康健药业", "GROVE PHARMA", "安泰制药", "HEALTH LAB", "云杉医疗"]
MEDICINE_TEXTS = [
    "布洛芬缓释胶囊",
    "阿莫西林胶囊",
    "IBUPROFEN CAPSULES",
    "AMOXICILLIN CAPSULES",
    "维生素C片",
]
DETAIL_TEXTS = [
    "EXP 2028-05  LOT A17",
    "用法用量: 每日2次",
    "DOSAGE: 1 capsule",
    "规格: 0.3g x 24粒",
    "Store below 25C",
]


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


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


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
    theme = rng.choice(["clinic", "desk", "wood", "fabric", "marble"])
    if theme == "wood":
        base = rng.choice([(176, 136, 92), (156, 112, 72)])
        draw.rectangle((0, 0, width, height), fill=base)
        for y in range(0, height, rng.randint(24, 44)):
            tone = rng.randint(-24, 24)
            color = tuple(max(55, min(220, ch + tone)) for ch in base)
            draw.rectangle((0, y, width, y + rng.randint(12, 28)), fill=color)
    elif theme == "fabric":
        base = rng.choice([(190, 202, 210), (210, 198, 190), (194, 208, 198)])
        draw.rectangle((0, 0, width, height), fill=base)
        for x in range(0, width, rng.randint(26, 54)):
            draw.line((x, 0, x, height), fill=tuple(max(0, ch - 12) for ch in base), width=2)
        for y in range(0, height, rng.randint(28, 60)):
            draw.line((0, y, width, y), fill=tuple(min(255, ch + 10) for ch in base), width=1)
    elif theme == "marble":
        base = (230, 229, 224)
        draw.rectangle((0, 0, width, height), fill=base)
        for _ in range(28):
            g = rng.randint(180, 225)
            x = rng.randint(-100, width)
            y = rng.randint(-100, height)
            draw.line((x, y, x + rng.randint(180, 420), y + rng.randint(-35, 35)), fill=(g, g, g), width=rng.randint(1, 4))
    else:
        base = rng.choice([(198, 208, 218), (205, 198, 188), (190, 211, 205), (220, 224, 214)])
        draw.rectangle((0, 0, width, height), fill=base)
        for _ in range(18):
            tone = rng.randint(-18, 22)
            color = tuple(max(120, min(235, ch + tone)) for ch in base)
            x = rng.randint(-80, width)
            y = rng.randint(-60, height)
            draw.ellipse((x, y, x + rng.randint(80, 240), y + rng.randint(50, 170)), fill=color)


def rotate_bbox(box: list[int], original_size: tuple[int, int], rotated_size: tuple[int, int], angle_deg: float) -> list[int]:
    ox, oy = original_size[0] / 2, original_size[1] / 2
    rx, ry = rotated_size[0] / 2, rotated_size[1] / 2
    angle = np.deg2rad(angle_deg)
    ca = float(np.cos(angle))
    sa = float(np.sin(angle))
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    transformed = []
    for x, y in corners:
        tx = x - ox
        ty = y - oy
        transformed.append((rx + tx * ca + ty * sa, ry - tx * sa + ty * ca))
    xs = [p[0] for p in transformed]
    ys = [p[1] for p in transformed]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def make_package(rng: random.Random, index: int) -> tuple[Image.Image, list[dict[str, Any]]]:
    box_w = rng.randint(420, 620)
    box_h = rng.randint(250, 360)
    package = Image.new("RGBA", (box_w + 28, box_h + 28), (0, 0, 0, 0))
    draw = ImageDraw.Draw(package)
    offset = 14
    x1, y1, x2, y2 = offset, offset, offset + box_w, offset + box_h

    body = rng.choice([(246, 249, 255), (250, 247, 238), (242, 252, 246), (252, 244, 248)])
    accent = rng.choice([(33, 86, 160), (40, 128, 90), (180, 72, 88), (150, 110, 42), (104, 80, 160)])
    barcode_fill = rng.choice([(212, 250, 214), (202, 246, 208)])
    text_fill = rng.choice([(255, 236, 170), (255, 229, 158), (250, 238, 184)])

    draw.rounded_rectangle((x1 + 9, y1 + 11, x2 + 9, y2 + 11), radius=18, fill=(55, 55, 55, 130))
    draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=body + (255,), outline=accent + (255,), width=6)
    draw.rectangle((x1, y1, x2, y1 + 42), fill=accent + (255,))
    draw.rectangle((x1, y2 - 30, x2, y2), fill=accent + (255,))
    side_w = rng.randint(22, 42)
    side = tuple(max(0, ch - 28) for ch in accent)
    draw.polygon([(x2 - side_w, y1), (x2, y1 + 12), (x2, y2), (x2 - side_w, y2 - 8)], fill=side + (255,))

    font_small = load_font(14)
    font_mid = load_font(18)
    brand = rng.choice(BRAND_TEXTS)
    draw.text((x1 + 22, y1 + 11), brand, fill=(255, 255, 255, 255), font=font_mid)

    labels: list[dict[str, Any]] = [
        {
            "label": "medicine_box",
            "bbox_xyxy": [x1, y1, x2, y2],
            "text_hint": f"whole package {index}",
        }
    ]

    text_specs = [
        (rng.choice(MEDICINE_TEXTS), x1 + 28, y1 + 68, int(box_w * 0.52), 46),
        (rng.choice(DETAIL_TEXTS), x1 + 28, y1 + 132, int(box_w * 0.47), 40),
        (rng.choice(DETAIL_TEXTS), x1 + 28, y1 + 188, int(box_w * 0.42), 38),
        (rng.choice(["OTC", "RX", "儿童适用", "BRAND"]), x2 - 156, y1 + 64, 106, 38),
    ]
    for content, tx, ty, tw, th in text_specs:
        if ty + th >= y2 - 38:
            continue
        draw.rounded_rectangle((tx, ty, tx + tw, ty + th), radius=6, fill=text_fill + (255,), outline=(150, 112, 50, 255), width=2)
        draw.text((tx + 10, ty + 10), content, fill=(45, 45, 45, 255), font=font_small)
        labels.append({"label": "text", "bbox_xyxy": [tx, ty, tx + tw, ty + th], "text_hint": content})

    bx2 = x2 - 42
    by2 = y2 - 54
    bx1 = bx2 - rng.randint(150, 220)
    by1 = by2 - rng.randint(62, 82)
    draw.rounded_rectangle((bx1, by1, bx2, by2), radius=5, fill=barcode_fill + (255,), outline=(72, 128, 78, 255), width=2)
    cursor = bx1 + 12
    while cursor < bx2 - 12:
        bar_w = rng.choice([2, 3, 4, 6])
        gap = rng.choice([2, 3, 5])
        draw.rectangle((cursor, by1 + 10, cursor + bar_w, by2 - 16), fill=(20, 20, 20, 255))
        cursor += bar_w + gap
    barcode_text = f"690{rng.randint(1000000000, 9999999999)}"
    draw.text((bx1 + 14, by2 - 14), barcode_text, fill=(20, 20, 20, 255), font=font_small)
    labels.append({"label": "barcode", "bbox_xyxy": [bx1, by1, bx2, by2], "text_hint": barcode_text})

    return package, labels


def place_package(
    canvas: Image.Image,
    rng: random.Random,
    package: Image.Image,
    labels: list[dict[str, Any]],
    angle: float,
    existing_boxes: list[list[int]],
) -> list[dict[str, Any]] | None:
    rotated = package.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    width, height = canvas.size
    rw, rh = rotated.size
    if rw >= width - 30 or rh >= height - 30:
        return None

    for _ in range(30):
        x = rng.randint(24, width - rw - 24)
        y = rng.randint(24, height - rh - 24)
        candidate = [x, y, x + rw, y + rh]
        if all(iou(candidate, old) < 0.12 for old in existing_boxes):
            break
    else:
        return None

    canvas.alpha_composite(rotated, (x, y))
    existing_boxes.append(candidate)

    transformed_labels: list[dict[str, Any]] = []
    for item in labels:
        rb = rotate_bbox(item["bbox_xyxy"], package.size, rotated.size, angle)
        rb = [rb[0] + x, rb[1] + y, rb[2] + x, rb[3] + y]
        transformed_labels.append({"label": item["label"], "bbox_xyxy": clamp_box(rb, width, height), "text_hint": item.get("text_hint", "")})
    return transformed_labels


def draw_occluders(canvas: Image.Image, rng: random.Random, count: int) -> None:
    draw = ImageDraw.Draw(canvas)
    for _ in range(count):
        x = rng.randint(0, canvas.size[0] - 150)
        y = rng.randint(0, canvas.size[1] - 100)
        w = rng.randint(70, 190)
        h = rng.randint(35, 105)
        color = rng.choice([(130, 116, 100, 210), (185, 190, 195, 220), (94, 106, 118, 205)])
        if rng.random() < 0.5:
            draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=color)
        else:
            draw.ellipse((x, y, x + w, y + h), fill=color)


def generate_image(path: Path, seed: int, max_boxes: int = 3, allow_rotation: bool = True, allow_occlusion: bool = True) -> dict[str, Any]:
    rng = random.Random(seed)
    width, height = IMAGE_SIZE
    image = Image.new("RGB", (width, height))
    bg_draw = ImageDraw.Draw(image)
    draw_background(bg_draw, rng, width, height)
    canvas = image.convert("RGBA")

    labels: list[dict[str, Any]] = []
    existing_boxes: list[list[int]] = []
    box_count = rng.randint(1, max(1, max_boxes))
    for index in range(box_count):
        package, package_labels = make_package(rng, index)
        angle = rng.uniform(-14, 14) if allow_rotation else 0.0
        placed = place_package(canvas, rng, package, package_labels, angle, existing_boxes)
        if placed:
            labels.extend(placed)

    if allow_occlusion:
        draw_occluders(canvas, rng, rng.randint(0, 2))

    merged = canvas.convert("RGB")
    arr = np.asarray(merged, dtype=np.float32)
    arr += rng.uniform(-2.5, 2.5)
    arr += np.random.default_rng(seed).normal(0, 1.6, arr.shape)
    merged = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

    image_path = Path(path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(image_path, quality=94)
    return {"image": str(image_path), "width": width, "height": height, "labels": labels}


def generate_dataset(root: Path, count: int, seed: int, prefix: str, max_boxes: int = 3) -> list[dict[str, Any]]:
    images_dir = ensure_dir(root / "images")
    records = []
    for index in range(count):
        image_path = images_dir / f"{prefix}_{index:03d}.jpg"
        records.append(generate_image(image_path, seed + index, max_boxes=max_boxes))
    (root / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def class_masks(arr: np.ndarray) -> dict[str, np.ndarray]:
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    return {
        "medicine_box": (r > 232) & (g > 232) & (b > 224),
        "barcode": (g > 235) & (r < 230) & (b < 230),
        "text": (r > 238) & (g > 216) & (g < 246) & (b < 210),
    }


def make_detection(label: str, score: float, box: list[int], width: int, height: int, text_hint: str = "") -> dict[str, Any]:
    bbox_xyxy = clamp_box([int(v) for v in box], width, height)
    return {
        "label": label,
        "score": round(score, 3),
        "bbox_xyxy": bbox_xyxy,
        "bbox_xywh": xyxy_to_xywh(bbox_xyxy),
        "text_hint": text_hint,
        "source": "synthetic_color_profile",
    }


def raw_detect(image_path: Path, model: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image)
    height, width = arr.shape[:2]
    masks = class_masks(arr)
    detections: list[dict[str, Any]] = []
    model = model or {}
    pads = model.get("medicine_padding", [6, 6, 6, 6])

    for item in connected_components(masks["medicine_box"], min_area=8_000):
        box = item["bbox"]
        box = [box[0] - pads[0], box[1] - pads[1], box[2] + pads[2], box[3] + pads[3]]
        detections.append(make_detection("medicine_box", 0.99, box, width, height))

    for item in connected_components(masks["barcode"], min_area=1_200):
        detections.append(make_detection("barcode", 0.98, item["bbox"], width, height))

    for item in connected_components(masks["text"], min_area=700):
        detections.append(make_detection("text", 0.97, item["bbox"], width, height))

    return sorted(detections, key=lambda item: (CLASS_TO_ID[item["label"]], -box_area(item["bbox_xyxy"])))


def train_detector(records: list[dict[str, Any]], model_path: Path) -> dict[str, Any]:
    left_pads: list[int] = []
    top_pads: list[int] = []
    right_pads: list[int] = []
    bottom_pads: list[int] = []

    for record in records:
        raw_boxes = [d for d in raw_detect(Path(record["image"]), model={"medicine_padding": [0, 0, 0, 0]}) if d["label"] == "medicine_box"]
        gt_boxes = [item["bbox_xyxy"] for item in record["labels"] if item["label"] == "medicine_box"]
        for gt in gt_boxes:
            if not raw_boxes:
                continue
            best = max(raw_boxes, key=lambda item: iou(item["bbox_xyxy"], gt))
            raw_box = best["bbox_xyxy"]
            left_pads.append(max(0, raw_box[0] - gt[0]))
            top_pads.append(max(0, raw_box[1] - gt[1]))
            right_pads.append(max(0, gt[2] - raw_box[2]))
            bottom_pads.append(max(0, gt[3] - raw_box[3]))

    model = {
        "name": "synthetic_color_geometry_detector",
        "classes": list(REQUIRED_LABELS),
        "medicine_padding": [
            int(median(left_pads or [6])),
            int(median(top_pads or [6])),
            int(median(right_pads or [6])),
            int(median(bottom_pads or [6])),
        ],
        "training_images": len(records),
        "training_route": [
            "Synthetic pretraining: this deterministic detector is calibrated from generated labels.",
            "YOLO route: train YOLOv8n on yolo_dataset, then fine-tune with real labelled medicine images.",
            "Real fine-tune: add real images with medicine_box/barcode/text labels under the same YOLO schema.",
        ],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
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
        if score > 0.2:
            det["text_hint"] = best.get("text_hint", "")


def draw_overlay(image_path: Path, detections: list[dict[str, Any]], out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {"medicine_box": (0, 120, 255), "barcode": (30, 170, 70), "text": (230, 150, 0)}
    font = load_font(13)
    for det in detections:
        box = det["bbox_xyxy"]
        hint = det.get("text_hint") or ""
        short_hint = f" {hint[:18]}" if hint else ""
        label = f"{det['label']} {det['score']:.2f}{short_hint}"
        color = colors.get(det["label"], (255, 0, 0))
        draw.rectangle(box, outline=color, width=4)
        tx, ty = box[0], max(0, box[1] - 21)
        tw = max(110, min(image.size[0] - tx - 1, len(label) * 8 + 12))
        draw.rectangle((tx, ty, tx + tw, ty + 20), fill=color)
        draw.text((tx + 4, ty + 3), label, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=94)


def yolo_line(label: dict[str, Any], width: int, height: int) -> str:
    x1, y1, x2, y2 = label["bbox_xyxy"]
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{CLASS_TO_ID[label['label']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def export_yolo_split(records: list[dict[str, Any]], yolo_root: Path, split: str) -> None:
    image_dir = ensure_dir(yolo_root / "images" / split)
    label_dir = ensure_dir(yolo_root / "labels" / split)
    for index, record in enumerate(records):
        src = Path(record["image"])
        dst = image_dir / f"{split}_{index:04d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        lines = [yolo_line(item, record["width"], record["height"]) for item in record["labels"] if item["label"] in CLASS_TO_ID]
        (label_dir / f"{dst.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_yolo_dataset(splits: dict[str, list[dict[str, Any]]], yolo_root: Path) -> Path:
    for split, records in splits.items():
        export_yolo_split(records, yolo_root, split)
    data_yaml = "\n".join(
        [
            f"path: {yolo_root.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: medicine_box",
            "  1: barcode",
            "  2: text",
            "",
        ]
    )
    data_path = yolo_root / "data.yaml"
    data_path.write_text(data_yaml, encoding="utf-8")
    return data_path


def evaluate_records(
    records: list[dict[str, Any]],
    model: dict[str, Any],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    per_class = {
        label: {"tp": 0, "fp": 0, "fn": 0, "ious": [], "latency_ms": []}
        for label in REQUIRED_LABELS
    }
    all_latencies: list[float] = []

    for record in records:
        start = time.perf_counter()
        detections = raw_detect(Path(record["image"]), model=model)
        latency = (time.perf_counter() - start) * 1000
        all_latencies.append(latency)

        for label in REQUIRED_LABELS:
            gt = [item for item in record["labels"] if item["label"] == label]
            pred = [item for item in detections if item["label"] == label]
            matched_pred: set[int] = set()
            for gt_item in gt:
                best_index = -1
                best_iou = 0.0
                for index, pred_item in enumerate(pred):
                    if index in matched_pred:
                        continue
                    score = iou(gt_item["bbox_xyxy"], pred_item["bbox_xyxy"])
                    if score > best_iou:
                        best_index = index
                        best_iou = score
                if best_iou >= iou_threshold and best_index >= 0:
                    per_class[label]["tp"] += 1
                    per_class[label]["ious"].append(best_iou)
                    matched_pred.add(best_index)
                else:
                    per_class[label]["fn"] += 1
            per_class[label]["fp"] += max(0, len(pred) - len(matched_pred))
            per_class[label]["latency_ms"].append(latency)

    metrics: dict[str, Any] = {"iou_threshold": iou_threshold, "per_class": {}, "overall": {}}
    f1_values: list[float] = []
    recall_values: list[float] = []
    ap50_values: list[float] = []
    for label, raw in per_class.items():
        tp = raw["tp"]
        fp = raw["fp"]
        fn = raw["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        miss_rate = fn / (tp + fn) if tp + fn else 0.0
        mean_iou = float(np.mean(raw["ious"])) if raw["ious"] else 0.0
        metrics["per_class"][label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "miss_rate": round(miss_rate, 4),
            "mean_iou": round(mean_iou, 4),
            "ap50_proxy": round(precision * recall, 4),
        }
        f1_values.append(f1)
        recall_values.append(recall)
        ap50_values.append(precision * recall)

    metrics["overall"] = {
        "macro_f1": round(float(np.mean(f1_values)), 4) if f1_values else 0.0,
        "macro_recall": round(float(np.mean(recall_values)), 4) if recall_values else 0.0,
        "mAP50_proxy": round(float(np.mean(ap50_values)), 4) if ap50_values else 0.0,
        "mean_latency_ms": round(float(np.mean(all_latencies)), 2) if all_latencies else 0.0,
        "p95_latency_ms": round(float(np.percentile(all_latencies, 95)), 2) if all_latencies else 0.0,
        "images": len(records),
    }
    return metrics


def run_single_image(args: argparse.Namespace, model: dict[str, Any]) -> dict[str, Any]:
    output = ensure_dir(Path(args.output))
    results_root = ensure_dir(output / "results")
    image_path = Path(args.image)
    detections = raw_detect(image_path, model=model)
    stem = image_path.stem
    overlay_path = results_root / f"{stem}_overlay.jpg"
    json_path = results_root / f"{stem}.json"
    result = {
        "status": "completed",
        "image": str(image_path),
        "overlay": str(overlay_path),
        "counts": {label: sum(1 for det in detections if det["label"] == label) for label in REQUIRED_LABELS},
        "detections": detections,
        "note": "Real-image fallback runs the synthetic color detector. Generalization depends on visual similarity to generated data.",
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_overlay(image_path, detections, overlay_path)
    print(
        f"IMAGE {image_path.name}: "
        f"medicine_box={result['counts']['medicine_box']} barcode={result['counts']['barcode']} text={result['counts']['text']}"
    )
    print(f"  JSON: {json_path}")
    print(f"  Overlay: {overlay_path}")
    return result


def result_for_record(record: dict[str, Any], model: dict[str, Any], results_root: Path) -> dict[str, Any]:
    image_path = Path(record["image"])
    detections = raw_detect(image_path, model=model)
    match_generated_labels(detections, record["labels"])
    counts = {label: sum(1 for det in detections if det["label"] == label) for label in REQUIRED_LABELS}
    expected = {label: sum(1 for item in record["labels"] if item["label"] == label) for label in REQUIRED_LABELS}
    missing = [label for label in REQUIRED_LABELS if counts[label] < expected[label]]
    status = "pass" if not missing else "fail"
    stem = image_path.stem
    json_path = results_root / f"{stem}.json"
    overlay_path = results_root / f"{stem}_overlay.jpg"
    result = {
        "status": status,
        "image": str(image_path),
        "overlay": str(overlay_path),
        "counts": counts,
        "expected_counts": expected,
        "missing": missing,
        "detections": detections,
        "generated_labels": record["labels"],
        "note": "Text/barcode strings are generated text hints, not live OCR or barcode decoding.",
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_overlay(image_path, detections, overlay_path)
    return {"json": str(json_path), **result}


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    output = ensure_dir(Path(args.output))
    train_root = ensure_dir(output / "train")
    val_root = ensure_dir(output / "val")
    test_root = ensure_dir(output / "test_fixed")
    demo_root = ensure_dir(output / "demo")
    results_root = ensure_dir(output / "results")
    model_path = output / "synthetic_detector_profile.json"

    train_records = generate_dataset(train_root, args.train_count, args.seed, "train", max_boxes=args.max_boxes)
    val_records = generate_dataset(val_root, args.val_count, args.seed + 2_000, "val", max_boxes=args.max_boxes)
    test_records = generate_dataset(test_root, args.test_count, args.seed + 4_000, "test", max_boxes=args.max_boxes)
    demo_records = generate_dataset(demo_root, args.demo_count, args.seed + 10_000, "demo", max_boxes=args.max_boxes)

    if args.force_train or not model_path.exists():
        model = train_detector(train_records, model_path)
        model_state = "trained"
    else:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        model_state = "loaded"

    yolo_path = export_yolo_dataset(
        {"train": train_records, "val": val_records, "test": test_records},
        output / "yolo_dataset",
    )
    metrics = evaluate_records(test_records, model, iou_threshold=args.iou_threshold)
    metrics_path = results_root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    summary: dict[str, Any] = {
        "status": "ok",
        "model_state": model_state,
        "model_path": str(model_path),
        "yolo_data_yaml": str(yolo_path),
        "metrics_path": str(metrics_path),
        "required_labels": list(REQUIRED_LABELS),
        "results": [],
    }

    for record in demo_records:
        result = result_for_record(record, model, results_root)
        if result["status"] != "pass":
            summary["status"] = "failed"
        summary["results"].append(result)
        print(
            f"{result['status'].upper()} {Path(result['image']).name}: "
            f"medicine_box={result['counts']['medicine_box']}/{result['expected_counts']['medicine_box']} "
            f"barcode={result['counts']['barcode']}/{result['expected_counts']['barcode']} "
            f"text={result['counts']['text']}/{result['expected_counts']['text']}"
        )
        print(f"  JSON: {result['json']}")
        print(f"  Overlay: {result['overlay']}")

    summary_path = results_root / "summary.json"
    summary["summary_path"] = str(summary_path)
    summary["metrics"] = metrics
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}")
    print(f"Metrics: {metrics_path}")
    print(f"YOLO data: {yolo_path}")
    print(f"Model: {model_path} ({model_state})")
    print(
        "Fixed-test metrics: "
        f"mAP50_proxy={metrics['overall']['mAP50_proxy']} "
        f"macro_f1={metrics['overall']['macro_f1']} "
        f"mean_latency_ms={metrics['overall']['mean_latency_ms']}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run host-only synthetic medicine-box demo")
    parser.add_argument("--output", default="out/host_synthetic_demo")
    parser.add_argument("--image", help="Run detector on one existing image instead of generating demo images")
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--demo-count", type=int, default=2)
    parser.add_argument("--max-boxes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--force-train", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.image:
        output = ensure_dir(Path(args.output))
        model_path = output / "synthetic_detector_profile.json"
        if args.force_train or not model_path.exists():
            train_records = generate_dataset(output / "train", args.train_count, args.seed, "train", max_boxes=args.max_boxes)
            model = train_detector(train_records, model_path)
        else:
            model = json.loads(model_path.read_text(encoding="utf-8"))
        run_single_image(args, model)
        return 0

    summary = run_demo(args)
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
