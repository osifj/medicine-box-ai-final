#!/usr/bin/env python3
"""Build a mixed YOLO dataset from synthetic data plus imported real labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_yolo_dataset  # noqa: E402
import run_host_synthetic_demo as demo  # noqa: E402
import yolo_dataset_utils as yolo  # noqa: E402


def copy_yolo_source(
    source_yaml: Path,
    output: Path,
    prefix: str,
    image_mode: str,
    include_splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, Any]:
    config = yolo.parse_data_yaml(source_yaml)
    summary: dict[str, Any] = {"source": str(source_yaml), "prefix": prefix, "splits": {}}
    for split in include_splits:
        images = yolo.image_files(source_yaml, split, config)
        if not images:
            continue
        out_image_dir = output / "images" / split
        out_label_dir = output / "labels" / split
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        boxes = 0
        for image_path in images:
            label_path = yolo.label_path_for_image(source_yaml, split, image_path, config)
            if not label_path.exists():
                continue
            stem = f"{prefix}_{split}_{image_path.stem}"
            dst_image = out_image_dir / f"{stem}{image_path.suffix.lower()}"
            dst_label = out_label_dir / f"{stem}.txt"
            if image_mode == "symlink":
                if dst_image.exists() or dst_image.is_symlink():
                    dst_image.unlink()
                target = os.path.relpath(image_path.resolve(), dst_image.parent.resolve())
                dst_image.symlink_to(target)
            else:
                shutil.copy2(image_path, dst_image)
            label_text = label_path.read_text(encoding="utf-8")
            dst_label.write_text(label_text if label_text.endswith("\n") else label_text + "\n", encoding="utf-8")
            copied += 1
            boxes += len([line for line in label_text.splitlines() if line.strip()])
        summary["splits"][split] = {"images": copied, "boxes": boxes, "image_mode": image_mode}
    return summary


def build_mixed_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists() and args.clean:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    synthetic_root = output.parent / "synthetic_source"
    train_records = demo.generate_dataset(synthetic_root / "train", args.train_count, args.seed, "train", max_boxes=args.max_boxes)
    val_records = demo.generate_dataset(synthetic_root / "val", args.val_count, args.seed + 2_000, "val", max_boxes=args.max_boxes)
    test_records = demo.generate_dataset(synthetic_root / "test", args.test_count, args.seed + 4_000, "test", max_boxes=args.max_boxes)
    synthetic_yaml = demo.export_yolo_dataset(
        {"train": train_records, "val": val_records, "test": test_records},
        output.parent / "synthetic_yolo",
    )

    sources = [
        copy_yolo_source(synthetic_yaml, output, "synth", "copy"),
    ]
    real_yaml = Path(args.real).expanduser()
    if real_yaml.exists():
        sources.append(copy_yolo_source(real_yaml, output, "real", args.real_image_mode, include_splits=("train", "val")))
    data_yaml = yolo.write_data_yaml(output, include_test=True, absolute_path=True)
    audit = audit_yolo_dataset.audit_dataset(data_yaml)
    manifest = {
        "status": "completed",
        "data_yaml": str(data_yaml),
        "sources": sources,
        "audit": {
            "status": audit["status"],
            "total_boxes": audit["total_boxes"],
            "total_class_counts": audit["total_class_counts"],
            "total_problems": audit["total_problems"],
        },
        "note": "Synthetic data supplies all classes. Real prelabelled data improves real text/medicine_box appearance.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build mixed synthetic+real YOLO training dataset")
    parser.add_argument("--output", default="out/host_training_mixed/yolo_dataset")
    parser.add_argument("--real", default="data/real_labelimg_yolo_prelabelled/data.yaml")
    parser.add_argument("--train-count", type=int, default=80)
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--max-boxes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--real-image-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--clean", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_mixed_dataset(args)
    print(f"Mixed YOLO: {manifest['data_yaml']}")
    for source in manifest["sources"]:
        print(f"  source={source['prefix']} {source['splits']}")
    print(f"  audit={manifest['audit']}")
    return 0 if manifest["audit"]["total_problems"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
