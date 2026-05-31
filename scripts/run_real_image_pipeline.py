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
    pink_or_purple = (r > 105) & (b > 85) & (r > g + 12) & (sat > 0.10)
    maroon = (r > 70) & (r > g + 18) & (b > 45) & (sat > 0.12)
    green_strip = (g > 105) & (b > 70) & (g > r + 18) & (sat > 0.18)
    blue_or_teal = (b > 95) & (b > r + 22) & (sat > 0.18)
    mask = pink_or_purple | maroon | green_strip | blue_or_teal
    return dilate(mask, horizontal=18, vertical=8)


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
        rectangular_score = min(aspect / 3.5, 1.0)
        score = area_ratio * 4.0 + rectangular_score
        expanded = [
            max(0, x1 - int(bw * 0.03)),
            max(0, y1 - int(bh * 0.05)),
            min(width - 1, x2 + int(bw * 0.03)),
            min(height - 1, y2 + int(bh * 0.05)),
        ]
        candidates.append({"bbox": expanded, "score": score, "area_ratio": area_ratio})
    if candidates:
        return max(candidates, key=lambda item: item["score"])

    # Fallback: large central box if color segmentation fails.
    return {
        "bbox": [int(width * 0.08), int(height * 0.15), int(width * 0.92), int(height * 0.82)],
        "score": 0.25,
        "area_ratio": 0.56,
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
            synth.make_detection(
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
    return sorted(kept[:12], key=lambda item: (item["bbox_xyxy"][1], item["bbox_xyxy"][0]))


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
        if bw < 70 or bh < 28 or bw / max(bh, 1) < 1.4:
            continue
        patch = gray[by1:by2, bx1:bx2]
        col_dark = (patch < 85).mean(axis=0)
        stripe_columns = col_dark > 0.28
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
        box = [x1 + bx1, y1 + by1, x1 + bx2, y1 + by2]
        detections.append(synth.make_detection("barcode", 0.76, box, width, height, text_hint="barcode_region"))
    return detections[:3]


def tesseract_text_score(image: Image.Image) -> tuple[float, list[str], bool]:
    exe = shutil.which("tesseract")
    if not exe:
        return 0.0, [], False
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.png"
        image.save(path)
        try:
            result = subprocess.run(
                [exe, str(path), "stdout", "-l", "eng+chi_sim"],
                text=True,
                capture_output=True,
                timeout=8,
                check=False,
            )
        except Exception:
            return 0.0, [], False
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    joined = " ".join(lines)
    alpha = sum(ch.isalpha() for ch in joined)
    digits = sum(ch.isdigit() for ch in joined)
    chinese = sum("\u4e00" <= ch <= "\u9fff" for ch in joined)
    score = min(8.0, alpha * 0.08 + digits * 0.05 + chinese * 0.18 + len(lines) * 0.25)
    return score, lines[:8], True


def detect_candidate(image: Image.Image, orientation: str) -> dict[str, Any]:
    transformed = synth.transform_image_orientation(image, orientation).convert("RGB")
    arr = np.asarray(transformed)
    height, width = arr.shape[:2]
    box_candidate = choose_medicine_box(arr)
    if box_candidate is None:
        medicine_box = [0, 0, width - 1, height - 1]
        box_score = 0.0
    else:
        medicine_box = box_candidate["bbox"]
        box_score = float(box_candidate["score"])
    detections = [
        synth.make_detection(
            "medicine_box",
            min(0.96, 0.55 + box_score / 5),
            medicine_box,
            width,
            height,
            text_hint="real image medicine box candidate",
        )
    ]
    text_regions = detect_text_regions(arr, medicine_box)
    barcode_regions = detect_barcode_regions(arr, medicine_box)
    detections.extend(text_regions)
    detections.extend(barcode_regions)
    ocr_score, ocr_lines, ocr_available = tesseract_text_score(transformed)
    score = box_score + min(len(text_regions), 8) * 0.85 + len(barcode_regions) * 0.35 + ocr_score
    return {
        "orientation": orientation,
        "score": round(score, 4),
        "image": transformed,
        "detections": detections,
        "text_count": len(text_regions),
        "barcode_count": len(barcode_regions),
        "ocr_available": ocr_available,
        "ocr_lines": ocr_lines,
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


def run_pipeline(image_path: Path, output_dir: Path, model_dir: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = synth.ensure_dir(output_dir)
    results_dir = synth.ensure_dir(output_dir / "results")
    model_dir = model_dir or Path("out/models")
    model_metadata = load_model_metadata(model_dir)
    image = Image.open(image_path).convert("RGB")

    candidates = [detect_candidate(image, orientation) for orientation in ORIENTATION_ORDER]
    best = max(candidates, key=lambda item: item["score"])
    barcode_status = "detected" if best["barcode_count"] else "not_visible"

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
        "orientation": best["orientation"],
        "orientation_candidates": [
            {
                "orientation": item["orientation"],
                "score": item["score"],
                "text_count": item["text_count"],
                "barcode_count": item["barcode_count"],
            }
            for item in candidates
        ],
        "barcode_status": barcode_status,
        "counts": {
            label: sum(1 for det in best["detections"] if det["label"] == label)
            for label in synth.REQUIRED_LABELS
        },
        "detections": best["detections"],
        "ocr_available": best["ocr_available"],
        "ocr_lines": best["ocr_lines"],
        "model": model_metadata,
        "latency_ms": round(elapsed_ms, 2),
        "note": "Real-image pipeline uses optional OCR scoring, trained/fallback model metadata, and deterministic visual detection.",
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"REAL {image_path.name}: orientation={result['orientation']} "
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_pipeline(Path(args.image), Path(args.output), Path(args.model_dir))
    return 0 if result["counts"]["medicine_box"] >= 1 and result["counts"]["text"] >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
