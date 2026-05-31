#!/usr/bin/env python3
"""Hybrid demo: Grove edge detection + host camera barcode/OCR recognition.

Flow:
1. Auto-lower Grove detection threshold for demo sensitivity
2. Wait/loop invoke Grove until medicine_box detected
3. If detected → trigger host camera capture + barcode/OCR pipeline
4. Pretty-print result with medicine name + confidence
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import serial
except ImportError:
    serial = None


GROVE_PROJECT = Path(__file__).resolve().parent.parent
HOST_PROJECT = Path("/Users/dep/Projects/New_project")
OUTPUT_DIR = GROVE_PROJECT / "out" / "hybrid_demo"
GROVE_SCRIPT = GROVE_PROJECT / "scripts" / "grovevision_at.py"
GROVE_PORT = "/dev/cu.usbmodem5B420573151"
GROVE_BAUD = 921600


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    """Extract complete JSON objects from mixed stdout/stderr text."""
    objects: list[dict[str, Any]] = []
    start = -1
    depth = 0
    in_string = False
    escaped = False

    for index, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = index
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    objects.append(json.loads(text[start : index + 1]))
                except json.JSONDecodeError:
                    pass
                start = -1

    return objects


def boxes_from_payloads(payloads: list[dict[str, Any]], min_score: int) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for payload in payloads:
        data = payload.get("data", payload)
        raw_boxes = data.get("boxes") if isinstance(data, dict) else None
        if not isinstance(raw_boxes, list):
            continue
        for box in raw_boxes:
            if isinstance(box, dict):
                candidate = {
                    "x": box.get("x"),
                    "y": box.get("y"),
                    "w": box.get("w"),
                    "h": box.get("h"),
                    "score": box.get("score"),
                    "target": box.get("target"),
                }
            elif isinstance(box, list) and len(box) >= 6:
                candidate = {
                    "x": box[0],
                    "y": box[1],
                    "w": box[2],
                    "h": box[3],
                    "score": box[4],
                    "target": box[5],
                }
            else:
                continue

            score = candidate.get("score")
            if isinstance(score, (int, float)) and score >= min_score:
                boxes.append(candidate)
    return boxes


def load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def send_at_command(port: str, cmd: str) -> str:
    """Send a raw AT command to the Grove device and return response."""
    if serial is None:
        return ""
    try:
        s = serial.Serial(port, GROVE_BAUD, timeout=2)
        time.sleep(0.5)
        s.write(f"{cmd}\r\n".encode())
        time.sleep(0.3)
        resp = s.read(4096).decode(errors="replace")
        s.close()
        return resp
    except Exception:
        return ""


def set_grove_tscore(port: str, score: int) -> bool:
    """Lower Grove detection threshold for better sensitivity."""
    print(f"[Grove] Setting TSCORE={score} (lower = more sensitive)...")
    resp = send_at_command(port, f"AT+TSCORE={score}")
    if resp:
        print(f"  Response: {resp.strip()[:120]}")
    return bool(resp)


def run_grove_probe(port: str = GROVE_PORT) -> dict[str, Any]:
    """Probe Grove device and return parsed model info."""
    print("[Grove] Probing device...")
    result = subprocess.run(
        [sys.executable, str(GROVE_SCRIPT), "--port", port, "probe"],
        capture_output=True, text=True, timeout=30,
        cwd=str(GROVE_PROJECT),
    )
    output = result.stdout + result.stderr

    model_name = "unknown"
    classes: list[str] = []
    for line in output.split("\n"):
        if line.startswith("Model:"):
            model_name = line.replace("Model:", "").strip()
        if line.startswith("Classes:"):
            parts = line.replace("Classes:", "").strip()
            for p in parts.split(","):
                p = p.strip()
                if "=" in p:
                    classes.append(p.split("=", 1)[1].strip())
                elif p:
                    classes.append(p)

    return {"model": model_name, "classes": classes, "raw": output}


def run_grove_invoke(port: str = GROVE_PORT, min_score: int = 20) -> dict[str, Any]:
    """Invoke Grove inference and return boxes."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_prefix = str(ensure_dir(OUTPUT_DIR) / f"grove_{timestamp}")

    result = subprocess.run(
        [sys.executable, str(GROVE_SCRIPT), "--port", port,
         "invoke", "--image", "--min-score", str(min_score), "--out", out_prefix],
        capture_output=True, text=True, timeout=60,
        cwd=str(GROVE_PROJECT),
    )
    output = result.stdout + result.stderr

    boxes = boxes_from_payloads(extract_json_objects(output), min_score)
    detected = bool(boxes)

    return {
        "detected": detected,
        "boxes": boxes,
        "image_path": str(Path(out_prefix) / "grovevision_frame.jpg"),
        "html_path": str(Path(out_prefix) / "grovevision_result.html"),
        "raw": output,
    }


def run_host_pipeline(image_path: str | None = None) -> dict[str, Any]:
    """Run host camera barcode + OCR pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_prefix = str(ensure_dir(OUTPUT_DIR) / f"host_{timestamp}")
    host_scripts = HOST_PROJECT / "scripts"

    if image_path:
        print(f"[Host] Running pipeline on image: {image_path}")
        script = host_scripts / "run_image_pipeline.py"
        if script.exists():
            result = subprocess.run(
                [sys.executable, str(script), "--image", image_path,
                 "--output", out_prefix + ".json", "--enable-detector"],
                capture_output=True, text=True, timeout=120,
                cwd=str(HOST_PROJECT),
            )
        else:
            result = subprocess.run(
                [sys.executable, "-c", f"""
import sys; sys.path.insert(0, '{HOST_PROJECT}')
from medicine_box_vision.pipeline import MedicineBoxPipeline, PipelineConfig
from pathlib import Path
pipeline = MedicineBoxPipeline(PipelineConfig(
    detector_model=Path('artifacts/detector_finetune_v4/detector_best.keras'),
    detector_meta=Path('artifacts/detector_finetune_v4/detector_meta.json'),
    ocr_model=Path('artifacts/ocr_synth_v2/ocr_predictor.keras'),
    ocr_meta=Path('artifacts/ocr_synth_v2/ocr_meta.json'),
    charset_file=Path('configs/charset_generated.txt'),
    enable_detector=True,
))
result = pipeline.analyze_image_path('{image_path}')
import json; print(json.dumps(result, ensure_ascii=False, indent=2))
"""],
                capture_output=True, text=True, timeout=120,
                cwd=str(HOST_PROJECT),
            )
    else:
        print("[Host] Capturing from host camera...")
        script = host_scripts / "run_host_camera.py"
        if script.exists():
            result = subprocess.run(
                [sys.executable, str(script),
                 "--save-dir", out_prefix,
                 "--max-frames", "1", "--enable-detector", "--save-preview"],
                capture_output=True, text=True, timeout=60,
                cwd=str(HOST_PROJECT),
            )
        else:
            return {"error": "Host camera script not found", "status": "skipped"}

    output = result.stdout + result.stderr

    parsed = load_json_file(Path(out_prefix + ".json"))
    if parsed is None:
        for obj in extract_json_objects(output):
            if "resolved_medicine_name" in obj or "detections" in obj:
                parsed = obj
                break

    return {
        "status": "completed" if result.returncode == 0 else "error",
        "output_prefix": out_prefix,
        "raw": output,
        "parsed": parsed,
    }


def pretty_print_result(result: dict[str, Any]) -> None:
    """Print a human-friendly summary of the demo result."""
    print("\n" + "=" * 70)
    print("  Medicine Box AI  -  Demo Result")
    print("=" * 70)

    grove = result.get("grove", {})
    host = result.get("host_camera", {})

    print(f"\n  Grove Edge:  {'DETECTED' if grove.get('detected') else 'no detection'}")
    boxes = grove.get("boxes", [])
    if boxes:
        for b in boxes:
            print(f"    - medicine_box  score={b['score']}  center=({b['x']},{b['y']})")

    print(f"\n  Host Camera: {host.get('status', 'unknown')}")
    parsed = host.get("parsed")
    if parsed and parsed.get("resolved_medicine_name"):
        med = parsed["resolved_medicine_name"]
        print(f"    Medicine:   {med.get('medicine_name', '?')}")
        print(f"    Evidence:   {med.get('reason', 'unknown')}")
        print(f"    Confidence: {med.get('score', 0):.2f}")

    detections = parsed.get("detections", []) if parsed else []
    texts = parsed.get("texts", []) if parsed else []
    if detections:
        print(f"    Detections: {len(detections)} regions")
    if texts:
        for t in texts:
            print(f"    OCR: \"{t.get('text', '?')}\"")

    barcodes = parsed.get("barcodes", []) if parsed else []
    barcode_regions = parsed.get("barcode_regions", []) if parsed else []
    if barcodes:
        print(f"    Barcodes: {len(barcodes)} decoded")
    elif barcode_regions:
        print(f"    Barcode regions: {len(barcode_regions)} found (not decoded)")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Hybrid demo: Grove edge + host camera")
    parser.add_argument("--grove-port", default=GROVE_PORT)
    parser.add_argument("--tscore", type=int, default=25,
                        help="Grove detection threshold (lower=more sensitive, default 25)")
    parser.add_argument("--min-score", type=int, default=10,
                        help="Minimum score to display boxes (default 10)")
    parser.add_argument("--host-image", help="Use existing image instead of capturing from camera")
    parser.add_argument("--wait", action="store_true",
                        help="Keep polling Grove until a medicine_box is detected")
    parser.add_argument("--wait-timeout", type=int, default=30,
                        help="Max seconds to wait for detection (default 30)")
    parser.add_argument("--skip-host", action="store_true", help="Skip host camera step")
    parser.add_argument("--output", default=str(ensure_dir(OUTPUT_DIR)))
    args = parser.parse_args()

    port = args.grove_port

    result: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "grove": {},
        "host_camera": {},
        "final_status": "unknown",
    }

    # Step 0: Lower Grove threshold for demo sensitivity
    print("=" * 60)
    print("Step 0: Configure Grove detection sensitivity")
    print("=" * 60)
    try:
        set_grove_tscore(port, args.tscore)
    except Exception as e:
        print(f"  TSCORE set failed: {e} (continuing anyway)")

    # Step 1: Probe Grove
    print("\n" + "=" * 60)
    print("Step 1: Grove probe")
    print("=" * 60)
    try:
        probe = run_grove_probe(port=port)
        result["grove"]["model"] = probe["model"]
        result["grove"]["classes"] = probe["classes"]
        print(f"  Model: {probe['model']}")
        print(f"  Classes: {probe['classes']}")
    except Exception as e:
        print(f"  Probe failed: {e}")
        result["final_status"] = "grove_probe_failed"
        pretty_print_result(result)
        return 1

    # Step 2: Invoke Grove (with optional wait loop)
    invoke = None
    start_time = time.time()

    while True:
        print("\n" + "=" * 60)
        elapsed = time.time() - start_time
        print(f"Step 2: Grove invoke [{elapsed:.0f}s elapsed]")
        print("=" * 60)
        try:
            invoke = run_grove_invoke(port=port, min_score=args.min_score)
            result["grove"]["detected"] = invoke["detected"]
            result["grove"]["boxes"] = invoke["boxes"]
            result["grove"]["image_path"] = invoke["image_path"]
            if invoke["detected"]:
                print(f"  DETECTED! {len(invoke['boxes'])} medicine_box(es)")
                break
            else:
                print(f"  No medicine_box detected.")
                if args.wait and (time.time() - start_time) < args.wait_timeout:
                    print("  Waiting... (place medicine box in front of Grove camera)")
                    time.sleep(1.0)
                    continue
                else:
                    break
        except Exception as e:
            print(f"  Invoke failed: {e}")
            if args.wait and (time.time() - start_time) < args.wait_timeout:
                time.sleep(2.0)
                continue
            result["final_status"] = "grove_invoke_failed"
            pretty_print_result(result)
            return 1

    # Step 3: Decide whether to trigger host
    if not invoke or not invoke["detected"]:
        print("\n" + "=" * 60)
        print("Step 3: No medicine_box detected")
        print("=" * 60)
        print("  Tip: Use --wait to keep polling until a box appears.")
        print("  Tip: Use --tscore 15 for even higher sensitivity.")
        result["final_status"] = "no_medicine_box_detected"
        result["host_camera"] = {"status": "skipped", "reason": "Grove did not detect medicine_box"}
        pretty_print_result(result)
        return 0

    # Step 4: Run host camera pipeline
    if args.skip_host:
        print("\n  Skipping host camera (--skip-host)")
        result["host_camera"] = {"status": "skipped"}
        result["final_status"] = "grove_only"
    else:
        print("\n" + "=" * 60)
        print("Step 4: Host camera -- barcode + OCR pipeline")
        print("=" * 60)
        try:
            host = run_host_pipeline(image_path=args.host_image)
            result["host_camera"] = host
            if host.get("parsed"):
                result["host_camera"]["parsed"] = host["parsed"]
            print(f"  Pipeline status: {host.get('status', 'unknown')}")
        except Exception as e:
            print(f"  Host pipeline failed: {e}")
            result["host_camera"] = {"status": "error", "error": str(e)}

    # Step 5: Final result
    if result["grove"]["detected"] and result["host_camera"].get("status") == "completed":
        result["final_status"] = "success"
    elif result["grove"]["detected"]:
        result["final_status"] = "grove_detected_host_pending"
    else:
        result["final_status"] = "incomplete"

    # Save result (strip raw output to keep file small)
    report = copy.deepcopy(result)
    report.get("grove", {}).pop("raw", None)
    report.get("host_camera", {}).pop("raw", None)
    output_path = Path(args.output) / f"hybrid_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved: {output_path}")

    pretty_print_result(result)

    return 0 if result["final_status"] in ("success", "grove_only", "no_medicine_box_detected") else 1


if __name__ == "__main__":
    sys.exit(main())
