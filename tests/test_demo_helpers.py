from __future__ import annotations

import importlib.util
import argparse
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class HybridParsingTests(unittest.TestCase):
    def test_extract_boxes_from_nested_grove_json(self) -> None:
        hybrid = load_script("run_hybrid_demo")
        text = 'noise {"type":1,"name":"INVOKE","data":{"boxes":[[10,20,30,40,91,0]]}} tail'
        payloads = hybrid.extract_json_objects(text)
        boxes = hybrid.boxes_from_payloads(payloads, min_score=50)
        self.assertEqual(boxes, [{"x": 10, "y": 20, "w": 30, "h": 40, "score": 91, "target": 0}])


class SyntheticDemoTests(unittest.TestCase):
    def test_import_labelimg_class_mapping(self) -> None:
        importer = load_script("import_labelimg_dataset")
        old_classes = ["barcode", "text"]
        self.assertEqual(
            importer.map_label_line("0 0.5 0.5 0.2 0.1", old_classes),
            "1 0.5 0.5 0.2 0.1",
        )
        self.assertEqual(
            importer.map_label_line("1 0.5 0.5 0.2 0.1", old_classes),
            "2 0.5 0.5 0.2 0.1",
        )

    def test_synthetic_demo_detects_required_labels(self) -> None:
        demo = load_script("run_host_synthetic_demo")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = demo.generate_dataset(root / "train", count=3, seed=123, prefix="train")
            model = demo.train_detector(records, root / "model.json")
            image_record = demo.generate_dataset(root / "demo", count=1, seed=999, prefix="demo")[0]
            detections = demo.raw_detect(Path(image_record["image"]), model=model)
            labels = {item["label"] for item in detections}
            self.assertIn("medicine_box", labels)
            self.assertIn("text", labels)
            for item in detections:
                self.assertIn("label", item)
                self.assertIn("score", item)
                self.assertIn("bbox_xyxy", item)
                self.assertIn("bbox_xywh", item)
                self.assertIn("text_hint", item)

    def test_yolo_export_and_metrics(self) -> None:
        demo = load_script("run_host_synthetic_demo")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = demo.generate_dataset(root / "train", count=2, seed=1, prefix="train")
            val = demo.generate_dataset(root / "val", count=1, seed=2, prefix="val")
            test = demo.generate_dataset(root / "test", count=1, seed=3, prefix="test")
            model = demo.train_detector(train, root / "model.json")
            data_yaml = demo.export_yolo_dataset({"train": train, "val": val, "test": test}, root / "yolo")
            metrics = demo.evaluate_records(test, model)
            self.assertTrue(data_yaml.exists())
            self.assertIn("mAP50_proxy", metrics["overall"])
            self.assertIn("macro_f1", metrics["overall"])
            self.assertIn("mean_latency_ms", metrics["overall"])

    def test_real_pipeline_on_generated_handheld_like_image(self) -> None:
        demo = load_script("run_host_synthetic_demo")
        real = load_script("run_real_image_pipeline")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = demo.generate_dataset(root / "demo", count=1, seed=20260601, prefix="demo")[0]
            result = real.run_pipeline(Path(record["image"]), root / "real_check", root / "models")
            self.assertGreaterEqual(result["counts"]["medicine_box"], 1)
            self.assertGreaterEqual(result["counts"]["text"], 1)
            self.assertIn(result["barcode_status"], {"detected", "not_visible"})
            self.assertIn(result["orientation"], set(real.ORIENTATION_ORDER))
            self.assertIn("quality_warnings", result)

    def test_audit_yolo_dataset_valid(self) -> None:
        audit = load_script("audit_yolo_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images" / "train").mkdir(parents=True)
            (root / "labels" / "train").mkdir(parents=True)
            Image.new("RGB", (64, 48), (240, 210, 220)).save(root / "images" / "train" / "sample.jpg")
            (root / "labels" / "train" / "sample.txt").write_text("2 0.500000 0.500000 0.400000 0.200000\n", encoding="utf-8")
            data_yaml = root / "data.yaml"
            data_yaml.write_text(
                "\n".join(
                    [
                        f"path: {root}",
                        "train: images/train",
                        "names:",
                        "  0: medicine_box",
                        "  1: barcode",
                        "  2: text",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            report = audit.audit_dataset(data_yaml)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["total_class_counts"]["text"], 1)

    def test_prelabel_adds_medicine_box(self) -> None:
        prelabel = load_script("prelabel_real_yolo_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "real"
            (root / "images" / "train").mkdir(parents=True)
            (root / "labels" / "train").mkdir(parents=True)
            image = Image.new("RGB", (160, 100), (230, 230, 225))
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 20, 145, 78), fill=(235, 160, 205))
            draw.rectangle((20, 20, 145, 38), fill=(120, 25, 75))
            image.save(root / "images" / "train" / "box.jpg")
            (root / "labels" / "train" / "box.txt").write_text("2 0.550000 0.320000 0.420000 0.120000\n", encoding="utf-8")
            data_yaml = root / "data.yaml"
            data_yaml.write_text(
                f"path: {root}\ntrain: images/train\nnames:\n  0: medicine_box\n  1: barcode\n  2: text\n",
                encoding="utf-8",
            )
            output = Path(tmp) / "prelabelled"
            manifest = prelabel.prelabel_dataset(
                argparse.Namespace(
                    input=str(data_yaml),
                    output=str(output),
                    clean=True,
                    image_mode="copy",
                    overwrite_medicine_box=False,
                    add_barcode=False,
                    min_medicine_score=0.18,
                )
            )
            labels = (output / "labels" / "train" / "box.txt").read_text(encoding="utf-8").splitlines()
            self.assertTrue(any(line.startswith("0 ") for line in labels))
            self.assertTrue(any(line.startswith("2 ") for line in labels))
            self.assertEqual(manifest["splits"][0]["added_medicine_box"], 1)

    def test_build_mixed_dataset(self) -> None:
        builder = load_script("build_host_training_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real"
            (real / "images" / "train").mkdir(parents=True)
            (real / "images" / "val").mkdir(parents=True)
            (real / "labels" / "train").mkdir(parents=True)
            (real / "labels" / "val").mkdir(parents=True)
            Image.new("RGB", (64, 48), (240, 190, 215)).save(real / "images" / "train" / "r0.jpg")
            Image.new("RGB", (64, 48), (240, 190, 215)).save(real / "images" / "val" / "r1.jpg")
            (real / "labels" / "train" / "r0.txt").write_text("0 0.5 0.5 0.8 0.6\n2 0.5 0.4 0.3 0.1\n", encoding="utf-8")
            (real / "labels" / "val" / "r1.txt").write_text("0 0.5 0.5 0.8 0.6\n2 0.5 0.4 0.3 0.1\n", encoding="utf-8")
            real_yaml = real / "data.yaml"
            real_yaml.write_text(
                f"path: {real}\ntrain: images/train\nval: images/val\nnames:\n  0: medicine_box\n  1: barcode\n  2: text\n",
                encoding="utf-8",
            )
            manifest = builder.build_mixed_dataset(
                argparse.Namespace(
                    output=str(root / "mixed" / "yolo_dataset"),
                    real=str(real_yaml),
                    train_count=1,
                    val_count=1,
                    test_count=1,
                    max_boxes=1,
                    seed=55,
                    real_image_mode="copy",
                    clean=True,
                )
            )
            self.assertEqual(manifest["audit"]["status"], "ok")
            self.assertGreaterEqual(manifest["audit"]["total_class_counts"]["medicine_box"], 3)


if __name__ == "__main__":
    unittest.main()
