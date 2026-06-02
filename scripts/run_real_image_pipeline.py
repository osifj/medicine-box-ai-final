#!/usr/bin/env python3
"""Real-image host pipeline for hand-held medicine-box photos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_host_synthetic_demo as synth  # noqa: E402


ORIENTATION_ORDER = ("normal", "rotated_180", "mirrored", "mirrored_rotated")
YOLO_CLASS_NAMES = {0: "medicine_box", 1: "barcode", 2: "text"}
DEFAULT_YOLO_MODEL_PATHS = (
    Path("out/models_yolo_final/yolo/medicine_box_mixed/weights/best.pt"),
    Path("out/models_yolo_path_smoke/yolo/medicine_box_mixed/weights/best.pt"),
)


def make_real_detection(
    label: str,
    score: float,
    box: list[int],
    width: int,
    height: int,
    text_hint: str = "",
    source: str = "real_visual_detector",
) -> dict[str, Any]:
    detection = synth.make_detection(label, score, box, width, height, text_hint=text_hint)
    detection["source"] = source
    return detection


def resolve_yolo_model_path(model_path: Path | None = None) -> Path | None:
    if model_path is not None:
        expanded = model_path.expanduser()
        return expanded if expanded.exists() else None
    for candidate in DEFAULT_YOLO_MODEL_PATHS:
        if candidate.exists():
            return candidate
    return None


def load_yolo_detector(model_path: Path | None, disabled: bool = False) -> tuple[Any | None, dict[str, Any]]:
    if disabled:
        return None, {"status": "disabled", "path": ""}
    resolved = resolve_yolo_model_path(model_path)
    if resolved is None:
        return None, {
            "status": "missing",
            "path": str(model_path.expanduser()) if model_path else "",
            "searched": [str(path) for path in DEFAULT_YOLO_MODEL_PATHS],
        }
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        return None, {"status": "unavailable", "path": str(resolved), "error": str(exc)}
    try:
        return YOLO(str(resolved)), {"status": "loaded", "path": str(resolved)}
    except Exception as exc:
        return None, {"status": "failed", "path": str(resolved), "error": str(exc)}


def clipped_xyxy(values: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = values
    ix1 = max(0, min(width - 1, int(round(x1))))
    iy1 = max(0, min(height - 1, int(round(y1))))
    ix2 = max(ix1 + 1, min(width, int(round(x2))))
    iy2 = max(iy1 + 1, min(height, int(round(y2))))
    return [ix1, iy1, ix2, iy2]


def yolo_result_detections(result: Any, width: int, height: int) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    try:
        cls_values = boxes.cls.detach().cpu().numpy().tolist()
        conf_values = boxes.conf.detach().cpu().numpy().tolist()
        xyxy_values = boxes.xyxy.detach().cpu().numpy().tolist()
    except Exception:
        return []

    detections: list[dict[str, Any]] = []
    for class_id, confidence, raw_box in zip(cls_values, conf_values, xyxy_values):
        label = YOLO_CLASS_NAMES.get(int(class_id))
        if label is None:
            continue
        box = clipped_xyxy([float(value) for value in raw_box], width, height)
        if synth.box_area(box) < 12:
            continue
        detections.append(
            make_real_detection(
                label,
                float(confidence),
                box,
                width,
                height,
                text_hint=f"yolo_{label}",
                source="yolo",
            )
        )
    return detections


def mask_saturation(arr: np.ndarray) -> np.ndarray:
    arr_f = arr.astype(np.float32)
    maxc = arr_f.max(axis=2)
    minc = arr_f.min(axis=2)
    return (maxc - minc) / np.maximum(maxc, 1.0)


def dilate(mask: np.ndarray, horizontal: int = 6, vertical: int = 2) -> np.ndarray:
    out = mask.copy()
    for _ in range(horizontal):
        shifted = out.copy()
        shifted[:, 1:] |= out[:, :-1]
        shifted[:, :-1] |= out[:, 1:]
        out = shifted
    for _ in range(vertical):
        shifted = out.copy()
        shifted[1:, :] |= out[:-1, :]
        shifted[:-1, :] |= out[1:, :]
        out = shifted
    return out


def real_medicine_mask(arr: np.ndarray) -> np.ndarray:
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    sat = mask_saturation(arr)
    pink_or_purple = (r > 130) & (b > 120) & (r > g + 16) & (b > g + 10) & (sat > 0.10)
    maroon = (r > 70) & (r > g + 18) & (b > 42) & (b > g + 4) & (sat > 0.12)
    green_strip = (g > 105) & (b > 70) & (g > r + 18) & (sat > 0.18)
    blue_or_teal = (b > 95) & (b > r + 22) & (sat > 0.18)
    mask = pink_or_purple | maroon | green_strip | blue_or_teal
    return dilate(mask, horizontal=12, vertical=5)


def choose_medicine_box(arr: np.ndarray) -> dict[str, Any] | None:
    height, width = arr.shape[:2]
    components = synth.connected_components(real_medicine_mask(arr), min_area=max(5_000, width * height // 90))
    candidates: list[dict[str, Any]] = []
    for component in components[:8]:
        box = component["bbox"]
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        if bw < width * 0.22 or bh < height * 0.12:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 1.2:
            continue
        area_ratio = synth.box_area(box) / (width * height)
        if area_ratio > 0.72:
            continue
        touches_borders = sum((x1 <= 2, y1 <= 2, x2 >= width - 2, y2 >= height - 2))
        if touches_borders >= 3:
            continue
        rectangular_score = min(aspect / 3.5, 1.0)
        score = area_ratio * 4.0 + rectangular_score
        expanded = [
            max(0, x1 - int(bw * 0.005)),
            max(0, y1 - int(bh * 0.02)),
            min(width - 1, x2 + int(bw * 0.005)),
            min(height - 1, y2 + int(bh * 0.02)),
        ]
        candidates.append({"bbox": expanded, "score": score, "area_ratio": area_ratio})
    if candidates:
        best = max(candidates, key=lambda item: item["score"])
        best["method"] = "color_segmentation"
        return best

    # Fallback: large central box if color segmentation fails.
    return {
        "bbox": [int(width * 0.08), int(height * 0.15), int(width * 0.92), int(height * 0.82)],
        "score": 0.25,
        "area_ratio": 0.56,
        "method": "central_fallback",
    }


def text_mask_for_crop(crop: np.ndarray) -> np.ndarray:
    gray = (0.299 * crop[:, :, 0] + 0.587 * crop[:, :, 1] + 0.114 * crop[:, :, 2]).astype(np.uint8)
    dark = gray < np.percentile(gray, 20)
    gx = np.zeros_like(gray, dtype=np.int16)
    gy = np.zeros_like(gray, dtype=np.int16)
    gx[:, 1:] = np.abs(gray[:, 1:].astype(np.int16) - gray[:, :-1].astype(np.int16))
    gy[1:, :] = np.abs(gray[1:, :].astype(np.int16) - gray[:-1, :].astype(np.int16))
    edges = (gx + gy) > max(28, np.percentile(gx + gy, 88))
    return dilate((dark | edges), horizontal=7, vertical=2)


def detect_text_regions(arr: np.ndarray, medicine_box: list[int]) -> list[dict[str, Any]]:
    height, width = arr.shape[:2]
    x1, y1, x2, y2 = medicine_box
    crop = arr[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    mask = text_mask_for_crop(crop)
    min_area = max(80, crop.shape[0] * crop.shape[1] // 3_500)
    components = synth.connected_components(mask, min_area=min_area)
    detections: list[dict[str, Any]] = []
    for component in components:
        bx1, by1, bx2, by2 = component["bbox"]
        bw = bx2 - bx1
        bh = by2 - by1
        if bw < 28 or bh < 8:
            continue
        if bw > crop.shape[1] * 0.92 or bh > crop.shape[0] * 0.34:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.9:
            continue
        box = [x1 + bx1, y1 + by1, x1 + bx2, y1 + by2]
        detections.append(
            make_real_detection(
                "text",
                0.72,
                box,
                width,
                height,
                text_hint=f"text_region_{len(detections) + 1}",
            )
        )

    # Merge near-duplicate detections caused by highlights.
    kept: list[dict[str, Any]] = []
    for det in sorted(detections, key=lambda item: synth.box_area(item["bbox_xyxy"]), reverse=True):
        if all(synth.iou(det["bbox_xyxy"], old["bbox_xyxy"]) < 0.45 for old in kept):
            kept.append(det)
    return sorted(kept[:6], key=lambda item: (item["bbox_xyxy"][1], item["bbox_xyxy"][0]))


def detect_barcode_regions(arr: np.ndarray, medicine_box: list[int]) -> list[dict[str, Any]]:
    height, width = arr.shape[:2]
    x1, y1, x2, y2 = medicine_box
    crop = arr[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    gray = (0.299 * crop[:, :, 0] + 0.587 * crop[:, :, 1] + 0.114 * crop[:, :, 2]).astype(np.uint8)
    dark = dilate(gray < 70, horizontal=2, vertical=5)
    components = synth.connected_components(dark, min_area=max(300, crop.shape[0] * crop.shape[1] // 4_500))
    detections: list[dict[str, Any]] = []
    for component in components:
        bx1, by1, bx2, by2 = component["bbox"]
        bw = bx2 - bx1
        bh = by2 - by1
        if bw < 90 or bh < 32 or bw / max(bh, 1) < 2.1:
            continue
        patch = gray[by1:by2, bx1:bx2]
        col_dark = (patch < 85).mean(axis=0)
        stripe_columns = col_dark > 0.45
        runs = 0
        in_run = False
        for value in stripe_columns.tolist():
            if value and not in_run:
                runs += 1
                in_run = True
            elif not value:
                in_run = False
        if runs < 6:
            continue
        stripe_fraction = float(stripe_columns.mean())
        if runs < 18 or not (0.18 <= stripe_fraction <= 0.55):
            continue
        if float(col_dark[stripe_columns].mean()) < 0.65:
            continue
        box = [x1 + bx1, y1 + by1, x1 + bx2, y1 + by2]
        detections.append(make_real_detection("barcode", 0.76, box, width, height, text_hint="barcode_region"))
    return detections[:3]


def merge_ocr_word_boxes(words: list[dict[str, Any]], image_size: tuple[int, int]) -> list[dict[str, Any]]:
    if not words:
        return []
    heights = [item["bbox"][3] - item["bbox"][1] for item in words]
    row_tolerance = max(12, int(np.median(heights) * 0.75))
    groups: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: ((item["bbox"][1] + item["bbox"][3]) / 2, item["bbox"][0])):
        center_y = (word["bbox"][1] + word["bbox"][3]) / 2
        placed = False
        for group in groups:
            group_center = np.mean([(item["bbox"][1] + item["bbox"][3]) / 2 for item in group])
            if abs(center_y - group_center) <= row_tolerance:
                group.append(word)
                placed = True
                break
        if not placed:
            groups.append([word])

    width, height = image_size
    regions: list[dict[str, Any]] = []
    for group in groups:
        ordered = sorted(group, key=lambda item: item["bbox"][0])
        x1 = max(0, min(item["bbox"][0] for item in ordered) - 4)
        y1 = max(0, min(item["bbox"][1] for item in ordered) - 3)
        x2 = min(width, max(item["bbox"][2] for item in ordered) + 4)
        y2 = min(height, max(item["bbox"][3] for item in ordered) + 3)
        if x2 - x1 < 24 or y2 - y1 < 8:
            continue
        hint = " ".join(item["text"] for item in ordered)[:80]
        confidence = float(np.mean([item["confidence"] for item in ordered]))
        regions.append({"bbox": [x1, y1, x2, y2], "text_hint": hint, "confidence": confidence})
    return sorted(regions, key=lambda item: (-(item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1])))[:6]


def tesseract_text_analysis(image: Image.Image) -> tuple[float, list[str], bool, list[dict[str, Any]]]:
    exe = shutil.which("tesseract")
    if not exe:
        return 0.0, [], False, []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.png"
        image.save(path)
        try:
            tsv = subprocess.run(
                [exe, str(path), "stdout", "-l", "eng+chi_sim", "--psm", "6", "tsv"],
                text=True,
                capture_output=True,
                timeout=8,
                check=False,
            )
            if not tsv.stdout.strip():
                tsv = subprocess.run(
                    [exe, str(path), "stdout", "-l", "eng", "--psm", "6", "tsv"],
                    text=True,
                    capture_output=True,
                    timeout=8,
                    check=False,
                )
        except Exception:
            return 0.0, [], False, []
        words: list[str] = []
        confidences: list[float] = []
        word_boxes: list[dict[str, Any]] = []
        for line in tsv.stdout.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 12:
                continue
            text = parts[11].strip()
            if not text:
                continue
            try:
                confidence = float(parts[10])
            except ValueError:
                confidence = -1.0
            if confidence > 0:
                words.append(text)
                confidences.append(confidence)
                try:
                    x, y, w, h = (int(float(parts[index])) for index in (6, 7, 8, 9))
                except ValueError:
                    continue
                has_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in text)
                if confidence >= 25 and (sum(ch.isalnum() for ch in text) >= 2 or has_chinese):
                    word_boxes.append(
                        {
                            "text": text,
                            "confidence": confidence,
                            "bbox": [x, y, x + w, y + h],
                        }
                    )
        try:
            text_result = subprocess.run(
                [exe, str(path), "stdout", "-l", "eng+chi_sim", "--psm", "6"],
                text=True,
                capture_output=True,
                timeout=8,
                check=False,
            )
            lines = [line.strip() for line in text_result.stdout.splitlines() if line.strip()]
        except Exception:
            lines = []
    clean_words = [word for word in words if any(ch.isalnum() for ch in word)]
    joined = " ".join(clean_words).lower()
    keywords = (
        "ibuprofen",
        "capsule",
        "capsules",
        "sustained",
        "release",
        "amoxicillin",
        "vitamin",
        "otc",
        "united",
        "laboratories",
        "pharma",
        "dosage",
        "exp",
    )
    keyword_hits = sum(1 for keyword in keywords if keyword in joined)
    valid_words = [word for word, confidence in zip(words, confidences) if confidence >= 25 and sum(ch.isalnum() for ch in word) >= 2]
    mean_conf = float(np.mean([confidence for confidence in confidences if confidence > 0])) if confidences else 0.0
    score = min(12.0, mean_conf / 18.0 + len(valid_words) * 0.22 + keyword_hits * 1.75)
    if not lines and clean_words:
        lines = [" ".join(clean_words[:10])]
    return score, lines[:8], True, merge_ocr_word_boxes(word_boxes, image.size)


def tesseract_text_score(image: Image.Image) -> tuple[float, list[str], bool]:
    score, lines, available, _regions = tesseract_text_analysis(image)
    return score, lines, available


def detect_candidate(image: Image.Image, orientation: str, use_ocr: bool = True) -> dict[str, Any]:
    transformed = synth.transform_image_orientation(image, orientation).convert("RGB")
    arr = np.asarray(transformed)
    height, width = arr.shape[:2]
    box_candidate = choose_medicine_box(arr)
    if box_candidate is None:
        medicine_box = [0, 0, width - 1, height - 1]
        box_score = 0.0
        box_method = "full_image_fallback"
    else:
        medicine_box = box_candidate["bbox"]
        box_score = float(box_candidate["score"])
        box_method = box_candidate.get("method", "unknown")
    detections = [
        make_real_detection(
            "medicine_box",
            min(0.96, 0.55 + box_score / 5),
            medicine_box,
            width,
            height,
            text_hint="real image medicine box candidate",
        )
    ]
    if use_ocr:
        ocr_score, ocr_lines, ocr_available, ocr_text_boxes = tesseract_text_analysis(transformed)
    else:
        ocr_score, ocr_lines, ocr_available, ocr_text_boxes = 0.0, [], False, []
    text_regions = detect_text_regions(arr, medicine_box)
    if ocr_text_boxes:
        mx1, my1, mx2, my2 = medicine_box
        mw = max(1, mx2 - mx1)
        mh = max(1, my2 - my1)
        filtered_ocr_boxes = []
        for item in ocr_text_boxes:
            x1, y1, x2, y2 = item["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if not (mx1 <= cx <= mx2 and my1 <= cy <= my2):
                continue
            if (x2 - x1) > mw * 0.86 or (y2 - y1) > mh * 0.24:
                continue
            filtered_ocr_boxes.append(item)
        ocr_regions = [
            make_real_detection(
                "text",
                min(0.9, 0.62 + item["confidence"] / 250),
                item["bbox"],
                width,
                height,
                text_hint=item["text_hint"],
            )
            for item in filtered_ocr_boxes
        ]
        for region in text_regions:
            if all(synth.iou(region["bbox_xyxy"], ocr_region["bbox_xyxy"]) < 0.25 for ocr_region in ocr_regions):
                ocr_regions.append(region)
            if len(ocr_regions) >= 6:
                break
        text_regions = sorted(ocr_regions[:6], key=lambda item: (item["bbox_xyxy"][1], item["bbox_xyxy"][0]))
    barcode_regions = detect_barcode_regions(arr, medicine_box)
    detections.extend(text_regions)
    detections.extend(barcode_regions)
    score = box_score + min(len(text_regions), 8) * 0.85 + len(barcode_regions) * 0.35 + ocr_score
    return {
        "orientation": orientation,
        "score": round(score, 4),
        "image": transformed,
        "detections": detections,
        "text_count": len(text_regions),
        "barcode_count": len(barcode_regions),
        "medicine_box_method": box_method,
        "ocr_available": ocr_available,
        "ocr_lines": ocr_lines,
    }


def label_count(detections: list[dict[str, Any]], label: str) -> int:
    return sum(1 for detection in detections if detection["label"] == label)


def merge_supplemental_detections(
    primary: list[dict[str, Any]],
    supplemental: list[dict[str, Any]],
    max_text: int = 8,
) -> list[dict[str, Any]]:
    merged = list(primary)
    for detection in supplemental:
        label = detection["label"]
        if label == "medicine_box" and label_count(merged, "medicine_box") >= 1:
            continue
        if label == "barcode" and label_count(merged, "barcode") >= 3:
            continue
        if label == "text" and label_count(merged, "text") >= max_text:
            continue
        if all(
            old["label"] != label or synth.iou(detection["bbox_xyxy"], old["bbox_xyxy"]) < 0.35
            for old in merged
        ):
            merged.append(detection)
    return sorted(merged, key=lambda item: (item["label"] != "medicine_box", item["bbox_xyxy"][1], item["bbox_xyxy"][0]))


def detect_yolo_candidate(
    image: Image.Image,
    orientation: str,
    yolo_model: Any,
    yolo_device: str = "",
    use_ocr: bool = True,
) -> dict[str, Any]:
    transformed = synth.transform_image_orientation(image, orientation).convert("RGB")
    width, height = transformed.size
    try:
        kwargs: dict[str, Any] = {
            "source": np.asarray(transformed),
            "imgsz": 640,
            "conf": 0.05,
            "iou": 0.5,
            "max_det": 24,
            "verbose": False,
        }
        if yolo_device:
            kwargs["device"] = yolo_device
        prediction = yolo_model.predict(**kwargs)
        yolo_detections = yolo_result_detections(prediction[0], width, height) if prediction else []
        yolo_error = ""
    except Exception as exc:
        yolo_detections = []
        yolo_error = str(exc)

    visual = detect_candidate(image, orientation, use_ocr=use_ocr)
    detections = merge_supplemental_detections(yolo_detections, visual["detections"])
    text_count = label_count(detections, "text")
    barcode_count = label_count(detections, "barcode")
    medicine_count = label_count(detections, "medicine_box")
    yolo_text = label_count(yolo_detections, "text")
    yolo_medicine = label_count(yolo_detections, "medicine_box")
    yolo_barcode = label_count(yolo_detections, "barcode")
    yolo_conf_score = sum(float(item["score"]) for item in yolo_detections)
    score = (
        visual["score"]
        + yolo_conf_score * 1.8
        + min(yolo_text, 8) * 0.65
        + yolo_medicine * 2.4
        + yolo_barcode * 0.45
    )
    if yolo_error:
        score -= 2.0
    if medicine_count == 0:
        score -= 4.0
    return {
        "orientation": orientation,
        "score": round(score, 4),
        "image": transformed,
        "detections": detections,
        "text_count": text_count,
        "barcode_count": barcode_count,
        "medicine_box_method": "yolo" if yolo_medicine else visual["medicine_box_method"],
        "ocr_available": visual["ocr_available"],
        "ocr_lines": visual["ocr_lines"],
        "model_backend": "yolo_hybrid",
        "yolo_error": yolo_error,
        "yolo_counts": {
            "medicine_box": yolo_medicine,
            "barcode": yolo_barcode,
            "text": yolo_text,
        },
        "yolo_detection_count": len(yolo_detections),
    }


def draw_overlay(image: Image.Image, detections: list[dict[str, Any]], out_path: Path) -> None:
    draw = ImageDraw.Draw(image)
    colors = {"medicine_box": (0, 120, 255), "barcode": (30, 170, 70), "text": (230, 150, 0)}
    font = synth.load_font(15)
    for det in detections:
        box = det["bbox_xyxy"]
        label = f"{det['label']} {det['score']:.2f}"
        color = colors.get(det["label"], (255, 0, 0))
        draw.rectangle(box, outline=color, width=5 if det["label"] == "medicine_box" else 3)
        tx, ty = box[0], max(0, box[1] - 22)
        tw = max(110, min(image.size[0] - tx - 1, len(label) * 9 + 12))
        draw.rectangle((tx, ty, tx + tw, ty + 21), fill=color)
        draw.text((tx + 4, ty + 3), label, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=94)


def load_model_metadata(model_dir: Path) -> dict[str, Any]:
    candidates = [
        model_dir / "host_detector_profile.json",
        Path("out/models/host_detector_profile.json"),
        Path("out/host_synthetic_demo/synthetic_detector_profile.json"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8")) | {"loaded_from": str(path)}
            except json.JSONDecodeError:
                continue
    return {"loaded_from": "", "status": "deterministic_fallback"}


def quality_warnings(best: dict[str, Any], barcode_status: str, yolo_status: dict[str, Any] | None = None) -> list[str]:
    warnings: list[str] = []
    if best.get("medicine_box_method") in {"central_fallback", "full_image_fallback"}:
        warnings.append("medicine_box_used_fallback_estimate")
    if best["text_count"] < 3:
        warnings.append("low_text_region_count")
    if barcode_status == "not_visible":
        warnings.append("barcode_not_visible_or_not_detected")
    if not best["ocr_available"]:
        warnings.append("ocr_engine_not_available")
    if yolo_status and yolo_status.get("status") not in {"loaded", "disabled"}:
        warnings.append(f"yolo_{yolo_status.get('status', 'unknown')}")
    if best.get("model_backend") == "deterministic_fallback" and yolo_status and yolo_status.get("status") == "loaded":
        warnings.append("yolo_loaded_but_not_used")
    if best.get("yolo_error"):
        warnings.append("yolo_prediction_error")
    return warnings


def run_pipeline(
    image_path: Path,
    output_dir: Path,
    model_dir: Path | None = None,
    yolo_model_path: Path | None = None,
    yolo_device: str = "",
    prefer_yolo: bool = True,
    no_yolo: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = synth.ensure_dir(output_dir)
    results_dir = synth.ensure_dir(output_dir / "results")
    model_dir = model_dir or Path("out/models")
    model_metadata = load_model_metadata(model_dir)
    image = Image.open(image_path).convert("RGB")

    yolo_model, yolo_status = load_yolo_detector(yolo_model_path, disabled=no_yolo or not prefer_yolo)
    if yolo_model is not None:
        candidates = [
            detect_yolo_candidate(image, orientation, yolo_model, yolo_device=yolo_device)
            for orientation in ORIENTATION_ORDER
        ]
        usable = [
            item
            for item in candidates
            if item.get("yolo_detection_count", 0) > 0
            and label_count(item["detections"], "medicine_box") >= 1
            and label_count(item["detections"], "text") >= 1
        ]
        if usable:
            backend = "yolo_hybrid"
            best = max(usable, key=lambda item: item["score"])
        else:
            backend = "deterministic_fallback"
            candidates = [detect_candidate(image, orientation) for orientation in ORIENTATION_ORDER]
            best = max(candidates, key=lambda item: item["score"])
    else:
        backend = "deterministic_fallback"
        candidates = [detect_candidate(image, orientation) for orientation in ORIENTATION_ORDER]
        best = max(candidates, key=lambda item: item["score"])
    best["model_backend"] = backend
    barcode_status = "detected" if best["barcode_count"] else "not_visible"
    warnings = quality_warnings(best, barcode_status, yolo_status)

    stem = image_path.stem
    corrected_path = results_dir / f"{stem}_corrected.jpg"
    overlay_path = results_dir / f"{stem}_overlay.jpg"
    json_path = results_dir / f"{stem}.json"
    best["image"].save(corrected_path, quality=94)
    draw_overlay(best["image"].copy(), best["detections"], overlay_path)
    elapsed_ms = (time.perf_counter() - started) * 1000

    result = {
        "status": "completed",
        "image": str(image_path),
        "corrected_image": str(corrected_path),
        "overlay": str(overlay_path),
        "model_backend": backend,
        "yolo_model_path": yolo_status.get("path", ""),
        "orientation": best["orientation"],
        "orientation_candidates": [
            {
                "orientation": item["orientation"],
                "score": item["score"],
                "text_count": item["text_count"],
                "barcode_count": item["barcode_count"],
                "medicine_box_method": item["medicine_box_method"],
                "model_backend": item.get("model_backend", backend),
                "yolo_counts": item.get("yolo_counts", {}),
                "yolo_detection_count": item.get("yolo_detection_count", 0),
            }
            for item in candidates
        ],
        "barcode_status": barcode_status,
        "medicine_box_method": best["medicine_box_method"],
        "quality_warnings": warnings,
        "pipeline_mode": "real_image_yolo_hybrid" if backend == "yolo_hybrid" else "deterministic_real_visual",
        "counts": {
            label: sum(1 for det in best["detections"] if det["label"] == label)
            for label in synth.REQUIRED_LABELS
        },
        "detections": best["detections"],
        "ocr_available": best["ocr_available"],
        "ocr_lines": best["ocr_lines"],
        "model": model_metadata | {"yolo": yolo_status},
        "latency_ms": round(elapsed_ms, 2),
        "note": "Real-image pipeline prefers YOLOv8n when a trained weight exists, then supplements/falls back with deterministic visual detection and optional OCR scoring.",
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"REAL {image_path.name}: backend={backend} orientation={result['orientation']} "
        f"medicine_box={result['counts']['medicine_box']} text={result['counts']['text']} "
        f"barcode={result['counts']['barcode']} barcode_status={barcode_status}"
    )
    print(f"  JSON: {json_path}")
    print(f"  Overlay: {overlay_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-image medicine-box pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="out/real_image_check")
    parser.add_argument("--model-dir", default="out/models")
    parser.add_argument("--yolo-model", default="", help="Path to YOLOv8 best.pt; defaults to out/models_yolo_final then smoke weight")
    parser.add_argument("--yolo-device", default="", help="YOLO device, for example mps/cpu/cuda")
    parser.add_argument("--prefer-yolo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO and force deterministic fallback")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_pipeline(
        Path(args.image),
        Path(args.output),
        Path(args.model_dir),
        yolo_model_path=Path(args.yolo_model) if args.yolo_model else None,
        yolo_device=args.yolo_device,
        prefer_yolo=args.prefer_yolo,
        no_yolo=args.no_yolo,
    )
    return 0 if result["counts"]["medicine_box"] >= 1 and result["counts"]["text"] >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
