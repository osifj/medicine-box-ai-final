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
from typing import Any

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
    parser.add_argument("--try-yolo-real", action="store_true", help="Run YOLOv8n on imported real LabelImg data")
    parser.add_argument("--real-yolo", default="data/real_labelimg_yolo/data.yaml")
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


def read_simple_data_yaml(data_yaml: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def summarize_yolo_dataset(data_yaml: Path) -> dict[str, Any]:
    if not data_yaml.exists():
        return {"status": "missing", "path": str(data_yaml)}
    config = read_simple_data_yaml(data_yaml)
    root = Path(config.get("path") or ".")
    if not root.is_absolute():
        root = data_yaml.parent / root
    names = ["medicine_box", "barcode", "text"]
    summary: dict[str, Any] = {"status": "available", "path": str(data_yaml), "splits": {}}
    for split in ("train", "val", "test"):
        rel = config.get(split)
        if not rel:
            continue
        label_dir = root / rel.replace("images/", "labels/")
        txts = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
        class_counts = {name: 0 for name in names}
        boxes = 0
        for txt in txts:
            for line in txt.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                cls = int(float(parts[0]))
                if 0 <= cls < len(names):
                    class_counts[names[cls]] += 1
                boxes += 1
        summary["splits"][split] = {
            "labels": len(txts),
            "boxes": boxes,
            "class_counts": class_counts,
        }
    return summary


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
    real_yolo_path = Path(args.real_yolo)
    real_yolo_summary = summarize_yolo_dataset(real_yolo_path)
    training_report["real_labelimg_yolo"] = real_yolo_summary
    if args.try_yolo:
        training_report["yolo"] = try_yolo_train(data_yaml, output_dir, args.epochs)
    if args.try_yolo_real:
        training_report["yolo_real"] = try_yolo_train(real_yolo_path, output_dir, args.epochs)

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
    if real_yolo_summary["status"] == "available":
        print(f"Imported real LabelImg YOLO: {real_yolo_path}")
        for split, info in real_yolo_summary["splits"].items():
            print(f"  {split}: labels={info['labels']} boxes={info['boxes']} class_counts={info['class_counts']}")
    if args.try_yolo_real:
        print(f"YOLO real: {training_report['yolo_real']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
