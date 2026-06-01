from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


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


if __name__ == "__main__":
    unittest.main()
