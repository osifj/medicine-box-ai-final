#!/usr/bin/env python3
"""Evaluate real-image deterministic pipeline against a YOLO-labelled dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_host_synthetic_demo as demo  # noqa: E402
import run_real_image_pipeline as real_pipeline  # noqa: E402
import yolo_dataset_utils as yolo  # noqa: E402


def labels_for_image(data_yaml: Path, split: str, image_path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    label_path = yolo.label_path_for_image(data_yaml, split, image_path, config)
    with Image.open(image_path) as image:
        width, height = image.size
    labels: list[dict[str, Any]] = []
    for box in yolo.read_yolo_boxes(label_path):
        if 0 <= box.class_id < len(yolo.CLASS_NAMES):
            labels.append(
                {
                    "label": yolo.CLASS_NAMES[box.class_id],
                    "bbox_xyxy": yolo.yolo_to_xyxy(box, width, height),
                }
            )
    return labels


def predict_candidate(
    image: Image.Image,
    yolo_model: Any | None,
    yolo_device: str,
    use_ocr: bool,
) -> tuple[dict[str, Any], str]:
    if yolo_model is not None:
        candidate = real_pipeline.detect_yolo_candidate(image, "normal", yolo_model, yolo_device=yolo_device, use_ocr=use_ocr)
        if (
            candidate.get("yolo_detection_count", 0) > 0
            and candidate["text_count"] >= 1
            and real_pipeline.label_count(candidate["detections"], "medicine_box") >= 1
        ):
            return candidate, "yolo_hybrid"
    return real_pipeline.detect_candidate(image, "normal", use_ocr=use_ocr), "deterministic_fallback"


def evaluate_images(
    data_yaml: Path,
    split: str,
    iou_threshold: float,
    use_ocr: bool = False,
    yolo_model_path: Path | None = None,
    yolo_device: str = "",
    no_yolo: bool = False,
    prefer_yolo: bool = True,
) -> dict[str, Any]:
    config = yolo.parse_data_yaml(data_yaml)
    per_class = {label: {"tp": 0, "fp": 0, "fn": 0, "ious": []} for label in yolo.CLASS_NAMES}
    latencies: list[float] = []
    image_reports: list[dict[str, Any]] = []
    yolo_model, yolo_status = real_pipeline.load_yolo_detector(yolo_model_path, disabled=no_yolo or not prefer_yolo)
    backend_counts = {"yolo_hybrid": 0, "deterministic_fallback": 0}

    for image_path in yolo.image_files(data_yaml, split, config):
        labels = labels_for_image(data_yaml, split, image_path, config)
        image = Image.open(image_path).convert("RGB")
        start = time.perf_counter()
        candidate, backend = predict_candidate(image, yolo_model, yolo_device, use_ocr)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        backend_counts[backend] += 1
        detections = candidate["detections"]
        counts = {name: sum(1 for det in detections if det["label"] == name) for name in yolo.CLASS_NAMES}
        expected = {name: sum(1 for item in labels if item["label"] == name) for name in yolo.CLASS_NAMES}

        for label in yolo.CLASS_NAMES:
            gt = [item for item in labels if item["label"] == label]
            pred = [item for item in detections if item["label"] == label]
            matched_pred: set[int] = set()
            for gt_item in gt:
                best_index = -1
                best_iou = 0.0
                for index, pred_item in enumerate(pred):
                    if index in matched_pred:
                        continue
                    score = demo.iou(gt_item["bbox_xyxy"], pred_item["bbox_xyxy"])
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
        image_reports.append(
            {
                "image": str(image_path),
                "counts": counts,
                "expected_counts": expected,
                "model_backend": backend,
                "latency_ms": round(latency_ms, 2),
            }
        )

    metrics: dict[str, Any] = {
        "split": split,
        "iou_threshold": iou_threshold,
        "use_ocr": use_ocr,
        "model_backend_counts": backend_counts,
        "yolo": yolo_status,
        "per_class": {},
        "overall": {},
        "images": image_reports,
    }
    f1_values: list[float] = []
    ap50_values: list[float] = []
    for label, raw in per_class.items():
        tp = raw["tp"]
        fp = raw["fp"]
        fn = raw["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ap50 = precision * recall
        metrics["per_class"][label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "miss_rate": round(fn / (tp + fn), 4) if tp + fn else 0.0,
            "mean_iou": round(float(np.mean(raw["ious"])), 4) if raw["ious"] else 0.0,
            "ap50_proxy": round(ap50, 4),
        }
        if tp + fn > 0:
            f1_values.append(f1)
            ap50_values.append(ap50)
    metrics["overall"] = {
        "macro_f1": round(float(np.mean(f1_values)), 4) if f1_values else 0.0,
        "mAP50_proxy": round(float(np.mean(ap50_values)), 4) if ap50_values else 0.0,
        "miss_rate": round(float(np.mean([item["miss_rate"] for item in metrics["per_class"].values()])), 4),
        "mean_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "images": len(image_reports),
    }
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate real-image pipeline on a YOLO dataset")
    parser.add_argument("--data", default="data/real_labelimg_yolo_prelabelled/data.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default="out/real_yolo_pipeline_eval/metrics.json")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--ocr", action="store_true", help="Enable OCR during evaluation; slower, not needed for box metrics")
    parser.add_argument("--yolo-model", default="", help="Path to YOLOv8 best.pt; defaults to final/smoke weight")
    parser.add_argument("--yolo-device", default="", help="YOLO device, for example mps/cpu/cuda")
    parser.add_argument("--prefer-yolo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO during evaluation")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = evaluate_images(
        Path(args.data),
        args.split,
        args.iou_threshold,
        use_ocr=args.ocr,
        yolo_model_path=Path(args.yolo_model) if args.yolo_model else None,
        yolo_device=args.yolo_device,
        no_yolo=args.no_yolo,
        prefer_yolo=args.prefer_yolo,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Real LabelImg eval: images={metrics['overall']['images']} "
        f"macro_f1={metrics['overall']['macro_f1']} "
        f"mAP50_proxy={metrics['overall']['mAP50_proxy']} "
        f"mean_latency_ms={metrics['overall']['mean_latency_ms']} "
        f"backend_counts={metrics['model_backend_counts']}"
    )
    print(f"  metrics: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
