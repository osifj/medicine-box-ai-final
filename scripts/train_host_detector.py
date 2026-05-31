#!/usr/bin/env python3
"""Train or prepare host-side detector assets.

Always creates synthetic data, YOLO-format labels, and a saved fallback profile.
If ultralytics is available and --try-yolo is passed, it also starts YOLOv8n
training and stores outputs under out/models/yolo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_host_synthetic_demo as demo  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare/train host detector")
    parser.add_argument("--output", default="out/models")
    parser.add_argument("--dataset-output", default="out/host_training")
    parser.add_argument("--train-count", type=int, default=80)
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--max-boxes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--try-yolo", action="store_true", help="Run YOLOv8n training if ultralytics is installed")
    parser.add_argument("--epochs", type=int, default=20)
    return parser


def try_yolo_train(data_yaml: Path, output_dir: Path, epochs: int) -> dict:
    if not shutil_which("yolo"):
        return {"status": "skipped", "reason": "ultralytics yolo command not found"}
    project = output_dir / "yolo"
    cmd = [
        "yolo",
        "detect",
        "train",
        "model=yolov8n.pt",
        f"data={data_yaml}",
        "imgsz=640",
        f"epochs={epochs}",
        f"project={project}",
        "name=medicine_box_synth",
    ]
    started = time.perf_counter()
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "status": "completed" if result.returncode == 0 else "failed",
        "command": cmd,
        "returncode": result.returncode,
        "latency_s": round(time.perf_counter() - started, 2),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def shutil_which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


def main() -> int:
    args = build_parser().parse_args()
    output_dir = demo.ensure_dir(Path(args.output))
    dataset_root = demo.ensure_dir(Path(args.dataset_output))
    model_path = output_dir / "host_detector_profile.json"

    train_records = demo.generate_dataset(dataset_root / "train", args.train_count, args.seed, "train", max_boxes=args.max_boxes)
    val_records = demo.generate_dataset(dataset_root / "val", args.val_count, args.seed + 2_000, "val", max_boxes=args.max_boxes)
    test_records = demo.generate_dataset(dataset_root / "test", args.test_count, args.seed + 4_000, "test", max_boxes=args.max_boxes)
    model = demo.train_detector(train_records, model_path)
    data_yaml = demo.export_yolo_dataset(
        {"train": train_records, "val": val_records, "test": test_records},
        dataset_root / "yolo_dataset",
    )
    metrics = demo.evaluate_records(test_records, model)
    metrics_path = output_dir / "host_detector_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    training_report = {
        "status": "completed",
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "yolo_data_yaml": str(data_yaml),
        "fallback_model": model,
        "metrics": metrics,
        "machine_learning_role": (
            "Fallback profile is learned/calibrated from generated labels. "
            "YOLOv8n training is supported as the stronger ML route when ultralytics is installed."
        ),
        "yolo": {"status": "not_requested"},
    }
    if args.try_yolo:
        training_report["yolo"] = try_yolo_train(data_yaml, output_dir, args.epochs)

    report_path = output_dir / "host_detector_training_report.json"
    report_path.write_text(json.dumps(training_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Model: {model_path}")
    print(f"YOLO data: {data_yaml}")
    print(f"Metrics: {metrics_path}")
    print(f"Training report: {report_path}")
    print(
        "Metrics: "
        f"mAP50_proxy={metrics['overall']['mAP50_proxy']} "
        f"macro_f1={metrics['overall']['macro_f1']} "
        f"mean_latency_ms={metrics['overall']['mean_latency_ms']}"
    )
    if args.try_yolo:
        print(f"YOLO: {training_report['yolo']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
