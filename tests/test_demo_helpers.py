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
    def test_synthetic_demo_detects_required_labels(self) -> None:
        demo = load_script("run_host_synthetic_demo")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = demo.generate_dataset(root / "train", count=3, seed=123, prefix="train")
            model = demo.train_detector(records, root / "model.json")
            image_record = demo.generate_dataset(root / "demo", count=1, seed=999, prefix="demo")[0]
            detections = demo.raw_detect(Path(image_record["image"]), model=model)
            labels = {item["label"] for item in detections}
            self.assertTrue(set(demo.REQUIRED_LABELS).issubset(labels))


if __name__ == "__main__":
    unittest.main()
