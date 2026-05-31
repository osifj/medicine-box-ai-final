#!/usr/bin/env python3
"""Small stdlib-only AT helper for Seeed Grove Vision AI / SSCMA devices."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import glob
import html
import json
import os
import pathlib
import platform
import select
import subprocess
import sys
import time
from typing import Any

try:
    import serial
except ImportError:  # pragma: no cover - fallback path for bare Python installs.
    serial = None


DEFAULT_BAUD = 921600
PROBE_COMMANDS = [
    "AT+ID?",
    "AT+NAME?",
    "AT+STAT?",
    "AT+VER?",
    "AT+MODEL?",
    "AT+INFO?",
    "AT+SENSOR?",
    "AT+TSCORE?",
    "AT+TIOU?",
]


def auto_port() -> str | None:
    patterns = [
        "/dev/cu.usbmodem*",
        "/dev/cu.wchusbserial*",
        "/dev/cu.usbserial*",
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
    ]
    ports: list[str] = []
    for pattern in patterns:
        ports.extend(glob.glob(pattern))
    ignore = ("Bluetooth", "FreeBuds", "debug-console", "wlan-debug")
    ports = [p for p in sorted(set(ports)) if not any(x in p for x in ignore)]
    return ports[0] if ports else None


def print_busy_hint(port: str) -> None:
    try:
        result = subprocess.run(
            ["lsof", port], text=True, capture_output=True, check=False
        )
    except FileNotFoundError:
        result = None
    if result and result.stdout.strip():
        print("\nPort is busy. Process using it:")
        print(result.stdout.rstrip())
        print(
            "\nClose/disconnect that serial monitor, SenseCraft WebSerial tab, "
            "or Arduino Serial Monitor, then retry."
        )


def configure_port(port: str, baud: int) -> None:
    stty_flag = "-f" if platform.system() == "Darwin" else "-F"
    cmd = [
        "stty",
        stty_flag,
        port,
        str(baud),
        "cs8",
        "-cstopb",
        "-parenb",
        "-ixon",
        "-ixoff",
        "raw",
        "-echo",
        "min",
        "0",
        "time",
        "1",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        if "Resource busy" in result.stderr:
            print_busy_hint(port)
        raise RuntimeError(result.stderr.strip() or "stty failed")


def open_serial(port: str, baud: int) -> Any:
    if serial is not None:
        try:
            ser = serial.Serial()
            ser.port = port
            ser.baudrate = baud
            ser.bytesize = 8
            ser.parity = "N"
            ser.stopbits = 1
            ser.timeout = 0
            ser.write_timeout = 1
            ser.dtr = False
            ser.rts = False
            ser.open()
            time.sleep(1.0)
            ser.reset_input_buffer()
            return ser
        except Exception as exc:
            if "busy" in str(exc).lower() or "resource" in str(exc).lower():
                print_busy_hint(port)
            raise RuntimeError(str(exc)) from exc

    configure_port(port, baud)
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    time.sleep(0.1)
    drain(fd)
    return fd


def close_serial(link: Any) -> None:
    if hasattr(link, "close"):
        link.close()
    else:
        os.close(link)


def serial_write(link: Any, data: bytes) -> None:
    if hasattr(link, "write"):
        link.write(data)
        link.flush()
    else:
        os.write(link, data)


def serial_read(link: Any, timeout: float) -> bytes:
    if hasattr(link, "read"):
        deadline = time.time() + timeout
        while time.time() < deadline:
            waiting = link.in_waiting
            if waiting:
                return link.read(waiting)
            time.sleep(0.02)
        return b""

    ready, _, _ = select.select([link], [], [], timeout)
    if not ready:
        return b""
    try:
        return os.read(link, 65536)
    except BlockingIOError:
        return b""


def drain(link: Any) -> None:
    if hasattr(link, "reset_input_buffer"):
        link.reset_input_buffer()
        return

    while True:
        ready, _, _ = select.select([link], [], [], 0)
        if not ready:
            return
        try:
            if not os.read(link, 65536):
                return
        except BlockingIOError:
            return


def send_command(link: Any, command: str, timeout: float, want_event: str | None = None) -> str:
    wire = command if command.endswith(("\r", "\n")) else command + "\r\n"
    serial_write(link, wire.encode("ascii"))

    chunks: list[bytes] = []
    deadline = time.time() + timeout
    last_read = time.time()
    saw_expected = False

    while time.time() < deadline:
        chunk = serial_read(link, 0.05)
        if chunk:
            if chunk:
                chunks.append(chunk)
                last_read = time.time()
                text = b"".join(chunks).decode("utf-8", errors="replace")
                for obj in extract_json_objects(text):
                    if want_event and obj.get("name") == want_event and obj.get("type") == 1:
                        saw_expected = True
                    elif not want_event and obj.get("type") == 0:
                        saw_expected = True
        elif chunks and saw_expected and time.time() - last_read > 0.25:
            break

    return b"".join(chunks).decode("utf-8", errors="replace")


def extract_json_objects(text: str) -> list[dict[str, Any]]:
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
                raw = text[start : index + 1]
                try:
                    objects.append(json.loads(raw))
                except json.JSONDecodeError:
                    pass
                start = -1

    return objects


def iter_box_lists(payloads: list[dict[str, Any]]) -> list[list[Any]]:
    boxes: list[list[Any]] = []
    for payload in payloads:
        data = payload.get("data", payload)
        raw_boxes = data.get("boxes") if isinstance(data, dict) else None
        if not isinstance(raw_boxes, list):
            continue
        for box in raw_boxes:
            if isinstance(box, dict):
                boxes.append(
                    [
                        box.get("x"),
                        box.get("y"),
                        box.get("w"),
                        box.get("h"),
                        box.get("score"),
                        box.get("target"),
                    ]
                )
            elif isinstance(box, list):
                boxes.append(box)
    return boxes


def find_image(payloads: list[dict[str, Any]]) -> str | None:
    for payload in reversed(payloads):
        data = payload.get("data", payload)
        if isinstance(data, dict):
            image = data.get("image")
            if isinstance(image, str) and len(image) > 100:
                return image
    return None


def class_names_from_payloads(payloads: list[dict[str, Any]]) -> list[str] | None:
    for payload in payloads:
        data = payload.get("data")
        if not isinstance(data, dict):
            continue
        info = data.get("info")
        if isinstance(info, str):
            decoded: Any | None = None
            try:
                decoded = json.loads(info)
            except json.JSONDecodeError:
                try:
                    padded = info + "=" * (-len(info) % 4)
                    decoded = json.loads(base64.b64decode(padded).decode("utf-8"))
                except Exception:
                    continue
            classes = decoded.get("classes") if isinstance(decoded, dict) else None
            if isinstance(classes, list):
                return [str(item) for item in classes]
    return None


def model_name_from_payloads(payloads: list[dict[str, Any]]) -> str | None:
    for payload in payloads:
        data = payload.get("data")
        if not isinstance(data, dict):
            continue
        info = data.get("info")
        if not isinstance(info, str):
            continue
        for candidate in (info,):
            try:
                decoded = json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    padded = candidate + "=" * (-len(candidate) % 4)
                    decoded = json.loads(base64.b64decode(padded).decode("utf-8"))
                except Exception:
                    continue
            if isinstance(decoded, dict) and isinstance(decoded.get("model_name"), str):
                return decoded["model_name"]
    return None


def print_json_payloads(text: str, raw: bool) -> list[dict[str, Any]]:
    payloads = extract_json_objects(text)
    if raw:
        print(text.rstrip())
        return payloads

    if not payloads:
        print(text.rstrip() or "(no response)")
        return payloads

    for payload in payloads:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payloads


def run_probe(link: Any, raw: bool) -> list[dict[str, Any]]:
    all_payloads: list[dict[str, Any]] = []
    for command in PROBE_COMMANDS:
        print(f"\n>>> {command}")
        text = send_command(link, command, timeout=1.5)
        payloads = print_json_payloads(text, raw)
        all_payloads.extend(payloads)
    return all_payloads


def summarize_boxes(boxes: list[list[Any]], classes: list[str] | None, min_score: int) -> None:
    if not boxes:
        print("\nNo boxes returned.")
        return

    print("\nDetected boxes:")
    for index, box in enumerate(boxes):
        if len(box) < 6:
            print(f"  [{index}] malformed box: {box}")
            continue
        x, y, w, h, score, target = box[:6]
        if isinstance(score, (int, float)) and score < min_score:
            continue
        label = str(target)
        if isinstance(target, int) and classes and 0 <= target < len(classes):
            label = classes[target]
        left = float(x) - float(w) / 2
        top = float(y) - float(h) / 2
        right = float(x) + float(w) / 2
        bottom = float(y) + float(h) / 2
        print(
            f"  [{index}] label={label} target={target} score={score} "
            f"center=({x},{y}) size=({w}x{h}) "
            f"rect=({left:.1f},{top:.1f})-({right:.1f},{bottom:.1f})"
        )


def write_overlay_html(
    out_dir: pathlib.Path,
    image_b64: str,
    boxes: list[list[Any]],
    classes: list[str] | None,
) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "grovevision_frame.jpg"
    html_path = out_dir / "grovevision_result.html"

    image_bytes = base64.b64decode(image_b64)
    image_path.write_bytes(image_bytes)

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Grove Vision Result</title>
<style>
  body {{ margin: 0; background: #111; color: #eee; font: 14px system-ui, sans-serif; }}
  main {{ min-height: 100vh; display: grid; place-items: center; padding: 24px; box-sizing: border-box; }}
  canvas {{ max-width: 100%; height: auto; box-shadow: 0 12px 40px #0008; }}
</style>
<main><canvas id="canvas"></canvas></main>
<script>
const imageSrc = {json.dumps(image_path.name)};
const boxes = {json.dumps(boxes)};
const classes = {json.dumps(classes or [])};
const colors = ["#00e5ff", "#ffca28", "#ff7043", "#66bb6a", "#ab47bc", "#ec407a"];
const img = new Image();
img.onload = () => {{
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);
  ctx.lineWidth = Math.max(2, Math.round(canvas.width / 160));
  ctx.font = `bold ${{Math.max(12, Math.round(canvas.width / 24))}}px system-ui, sans-serif`;
  boxes.forEach((box, index) => {{
    if (!Array.isArray(box) || box.length < 6) return;
    const [x, y, w, h, score, target] = box;
    const left = x - w / 2;
    const top = y - h / 2;
    const color = colors[Math.abs(target || index) % colors.length];
    const label = `${{classes[target] || target}}: ${{score}}`;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.strokeRect(left, top, w, h);
    const metrics = ctx.measureText(label);
    const textH = Math.max(16, Math.round(canvas.width / 18));
    ctx.fillRect(left, Math.max(0, top - textH), metrics.width + 12, textH);
    ctx.fillStyle = "#111";
    ctx.fillText(label, left + 6, Math.max(textH - 4, top - 5));
  }});
}};
img.src = imageSrc;
</script>
</html>
"""
    html_path.write_text(page, encoding="utf-8")
    return html_path


def run_invoke(args: argparse.Namespace, link: Any, known_classes: list[str] | None) -> None:
    result_only = 0 if args.image else 1
    command = f"AT+INVOKE=1,0,{result_only}"
    text = send_command(link, command, timeout=args.timeout, want_event="INVOKE")
    payloads = print_json_payloads(text, args.raw)
    classes = known_classes or class_names_from_payloads(payloads)
    boxes = iter_box_lists(payloads)
    summarize_boxes(boxes, classes, args.min_score)

    image_b64 = find_image(payloads)
    if args.image and image_b64:
        html_path = write_overlay_html(pathlib.Path(args.out), image_b64, boxes, classes)
        print(f"\nWrote overlay: {html_path}")
    elif args.image:
        print("\nImage was requested, but the device did not include image data.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe and invoke Grove Vision AI through SSCMA AT commands."
    )
    parser.add_argument("--port", default=auto_port(), help="Serial port path")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--raw", action="store_true", help="Print raw serial text")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe", help="Read ID, version, model, sensor and thresholds")

    invoke = sub.add_parser("invoke", help="Run one inference and print boxes")
    invoke.add_argument("--image", action="store_true", help="Request JPEG image too")
    invoke.add_argument("--timeout", type=float, default=8.0)
    invoke.add_argument("--min-score", type=int, default=0)
    invoke.add_argument("--out", default="out")
    invoke.add_argument("--loop", action="store_true", help="Invoke repeatedly")
    invoke.add_argument("--interval", type=float, default=0.5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.port:
        print("No serial port found. Plug in Grove Vision or pass --port /dev/...", file=sys.stderr)
        return 2

    print(f"Opening {args.port} at {args.baud} baud")
    try:
        link = open_serial(args.port, args.baud)
    except Exception as exc:
        print(f"Failed to open serial port: {exc}", file=sys.stderr)
        return 1

    known_classes: list[str] | None = None
    try:
        if args.cmd == "probe":
            payloads = run_probe(link, args.raw)
            classes = class_names_from_payloads(payloads)
            model_name = model_name_from_payloads(payloads)
            if model_name:
                print(f"\nModel: {model_name}")
            if classes:
                print("\nClasses:", ", ".join(f"{i}={name}" for i, name in enumerate(classes)))
        elif args.cmd == "invoke":
            info_text = send_command(link, "AT+INFO?", timeout=1.5)
            known_classes = class_names_from_payloads(extract_json_objects(info_text))
            while True:
                print(f"\n[{dt.datetime.now().strftime('%H:%M:%S')}]")
                run_invoke(args, link, known_classes)
                if not args.loop:
                    break
                time.sleep(args.interval)
        return 0
    finally:
        close_serial(link)


if __name__ == "__main__":
    raise SystemExit(main())
