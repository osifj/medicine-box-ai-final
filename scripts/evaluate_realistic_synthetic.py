#!/usr/bin/env python3
"""Evaluate on a fixed realistic synthetic medicine-box test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_host_synthetic_demo as demo  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate realistic synthetic fixed test set")
    parser.add_argument("--output", default="out/realistic_synthetic_eval")
    parser.add_argument("--model", default="out/models/host_detector_profile.json")
    parser.add_argument("--test-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = demo.ensure_dir(Path(args.output))
    model_path = Path(args.model)
    test_records = demo.generate_dataset(output / "test_fixed", args.test_count, args.seed + 4_000, "test", max_boxes=3)
    if model_path.exists():
        model = json.loads(model_path.read_text(encoding="utf-8"))
    else:
        train_records = demo.generate_dataset(output / "train_for_eval", 40, args.seed, "train", max_boxes=3)
        model = demo.train_detector(train_records, output / "host_detector_profile.json")
        model_path = output / "host_detector_profile.json"
    data_yaml = demo.export_yolo_dataset({"train": [], "val": [], "test": test_records}, output / "yolo_dataset")
    metrics = demo.evaluate_records(test_records, model, iou_threshold=args.iou_threshold)
    result = {
        "status": "completed",
        "model": str(model_path),
        "yolo_data_yaml": str(data_yaml),
        "manifest": str(output / "test_fixed" / "manifest.json"),
        "metrics": metrics,
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "Realistic synthetic metrics: "
        f"mAP50_proxy={metrics['overall']['mAP50_proxy']} "
        f"macro_f1={metrics['overall']['macro_f1']} "
        f"miss_rate_text={metrics['per_class']['text']['miss_rate']} "
        f"mean_latency_ms={metrics['overall']['mean_latency_ms']}"
    )
    print(f"Wrote: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
