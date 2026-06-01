#!/usr/bin/env python3
"""Small helpers for this repo's YOLO-format medicine-box datasets."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Any


CLASS_NAMES = ["medicine_box", "barcode", "text"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    cx: float
    cy: float
    width: float
    height: float

    @property
    def values(self) -> tuple[float, float, float, float]:
        return self.cx, self.cy, self.width, self.height


def parse_data_yaml(data_yaml: Path) -> dict[str, Any]:
    config: dict[str, Any] = {"names": {}}
    in_names = False
    for raw in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith(" ") or line.startswith("\t"):
            if in_names and ":" in line:
                key, value = line.split(":", 1)
                try:
                    config["names"][int(key.strip())] = value.strip().strip("\"'")
                except ValueError:
                    continue
            continue
        in_names = False
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key == "names":
            in_names = True
            if value.startswith("[") and value.endswith("]"):
                names = [item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip()]
                config["names"] = {index: name for index, name in enumerate(names)}
        else:
            config[key] = value
    if not config["names"]:
        config["names"] = {index: name for index, name in enumerate(CLASS_NAMES)}
    return config


def dataset_root(data_yaml: Path, config: dict[str, Any] | None = None) -> Path:
    config = config or parse_data_yaml(data_yaml)
    root = Path(config.get("path") or ".").expanduser()
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def split_image_dir(data_yaml: Path, split: str, config: dict[str, Any] | None = None) -> Path | None:
    config = config or parse_data_yaml(data_yaml)
    rel = config.get(split)
    if not rel:
        return None
    return dataset_root(data_yaml, config) / str(rel)


def split_label_dir(data_yaml: Path, split: str, config: dict[str, Any] | None = None) -> Path | None:
    image_dir = split_image_dir(data_yaml, split, config)
    if image_dir is None:
        return None
    parts = list(image_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts)
    return image_dir.parent.parent / "labels" / image_dir.name


def image_files(data_yaml: Path, split: str, config: dict[str, Any] | None = None) -> list[Path]:
    image_dir = split_image_dir(data_yaml, split, config)
    if image_dir is None or not image_dir.exists():
        return []
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def label_files(data_yaml: Path, split: str, config: dict[str, Any] | None = None) -> list[Path]:
    label_dir = split_label_dir(data_yaml, split, config)
    if label_dir is None or not label_dir.exists():
        return []
    return sorted(label_dir.glob("*.txt"))


def label_path_for_image(data_yaml: Path, split: str, image_path: Path, config: dict[str, Any] | None = None) -> Path:
    label_dir = split_label_dir(data_yaml, split, config)
    if label_dir is None:
        raise ValueError(f"split has no label dir: {split}")
    return label_dir / f"{image_path.stem}.txt"


def parse_yolo_line(line: str) -> YoloBox:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError("expected 5 YOLO fields")
    class_id = int(float(parts[0]))
    cx, cy, width, height = (float(value) for value in parts[1:])
    return YoloBox(class_id, cx, cy, width, height)


def read_yolo_boxes(label_path: Path) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            boxes.append(parse_yolo_line(line))
    return boxes


def format_yolo_line(box: YoloBox) -> str:
    return f"{box.class_id} {box.cx:.6f} {box.cy:.6f} {box.width:.6f} {box.height:.6f}"


def xyxy_to_yolo(box: list[int] | tuple[int, int, int, int], image_width: int, image_height: int, class_id: int) -> YoloBox:
    x1, y1, x2, y2 = box
    x1 = max(0, min(image_width - 1, int(round(x1))))
    y1 = max(0, min(image_height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(image_width, int(round(x2))))
    y2 = max(y1 + 1, min(image_height, int(round(y2))))
    return YoloBox(
        class_id=class_id,
        cx=((x1 + x2) / 2) / image_width,
        cy=((y1 + y2) / 2) / image_height,
        width=(x2 - x1) / image_width,
        height=(y2 - y1) / image_height,
    )


def yolo_to_xyxy(box: YoloBox, image_width: int, image_height: int) -> list[int]:
    bw = box.width * image_width
    bh = box.height * image_height
    x1 = int(round(box.cx * image_width - bw / 2))
    y1 = int(round(box.cy * image_height - bh / 2))
    x2 = int(round(box.cx * image_width + bw / 2))
    y2 = int(round(box.cy * image_height + bh / 2))
    return [
        max(0, min(image_width - 1, x1)),
        max(0, min(image_height - 1, y1)),
        max(1, min(image_width, x2)),
        max(1, min(image_height, y2)),
    ]


def write_data_yaml(output: Path, include_test: bool = True, absolute_path: bool = False) -> Path:
    path_value = str(output.resolve()) if absolute_path else "."
    lines = [
        f"path: {path_value}",
        "train: images/train",
        "val: images/val",
    ]
    if include_test:
        lines.append("test: images/test")
    lines.extend(
        [
            "names:",
            "  0: medicine_box",
            "  1: barcode",
            "  2: text",
            "",
        ]
    )
    path = output / "data.yaml"
    path.write_text("\n".join(lines), encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    return path


def copy_or_link_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        target = os.path.relpath(src.resolve(), dst.parent.resolve())
        dst.symlink_to(target)
    else:
        shutil.copy2(src, dst)
