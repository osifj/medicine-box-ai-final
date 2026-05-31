#!/usr/bin/env python3
"""Evaluate the host synthetic detector against a generated manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_host_synthetic_demo as demo  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate host synthetic detector")
    parser.add_argument("--manifest", default="out/host_synthetic_demo/test_fixed/manifest.json")
    parser.add_argument("--model", default="out/host_synthetic_demo/synthetic_detector_profile.json")
    parser.add_argument("--output", default="out/host_synthetic_demo/results/eval_metrics.json")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest)
    model_path = Path(args.model)
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = json.loads(model_path.read_text(encoding="utf-8"))
    metrics = demo.evaluate_records(records, model, iou_threshold=args.iou_threshold)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "Evaluation: "
        f"mAP50_proxy={metrics['overall']['mAP50_proxy']} "
        f"macro_f1={metrics['overall']['macro_f1']} "
        f"mean_latency_ms={metrics['overall']['mean_latency_ms']}"
    )
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
