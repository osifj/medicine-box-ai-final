#!/usr/bin/env python3
"""Deploy a Grove Vision AI V2 / Grove AI WE2 model over USB serial.

This mirrors the Seeed SenseCraft Web Toolkit WE2 flow:
1. reset into the Himax bootloader,
2. XMODEM-send an offset config block,
3. XMODEM-send the model bytes to 0x400000,
4. reboot and write model metadata with AT+INFO.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
from typing import Any

try:
    import serial
except ImportError:  # pragma: no cover
    serial = None


DEFAULT_PORT = "/dev/cu.usbmodem5B420573151"
DEFAULT_BAUD = 921600
DEFAULT_ADDRESS = 0x400000

SOH = 0x01
EOT = 0x04
ACK = 0x06
NAK = 0x15
CAN = 0x18
CRC_MODE = 0x43
FILLER = 0x1A


def crc16_xmodem(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def auto_port() -> str:
    for pattern in ("/dev/cu.usbmodem*", "/dev/cu.wchusbserial*", "/dev/cu.usbserial*"):
        import glob

        ports = sorted(glob.glob(pattern))
        if ports:
            return ports[0]
    return DEFAULT_PORT


def parse_address(value: str) -> int:
    return int(value, 0)


def open_link(port: str, baud: int) -> serial.Serial:
    if serial is None:
        raise SystemExit("pyserial is required for flashing: python3 -m pip install pyserial")
    ser = serial.Serial(
        port=port,
        baudrate=baud,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=0,
        write_timeout=2,
    )
    ser.rts = True
    ser.dtr = False
    return ser


def hard_reset(ser: serial.Serial) -> None:
    ser.rts = False
    time.sleep(0.1)
    ser.rts = True


def read_available(ser: serial.Serial) -> bytes:
    waiting = ser.in_waiting
    return ser.read(waiting) if waiting else b""


def read_until(ser: serial.Serial, needle: bytes, timeout: float) -> bytes:
    deadline = time.time() + timeout
    buffer = bytearray()
    while time.time() < deadline:
        chunk = read_available(ser)
        if chunk:
            buffer.extend(chunk)
            if needle in buffer:
                return bytes(buffer)
        time.sleep(0.01)
    return bytes(buffer)


def wait_byte(ser: serial.Serial, timeout: float) -> int | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = ser.read(1)
        if data:
            return data[0]
        time.sleep(0.01)
    return None


def enter_bootloader(ser: serial.Serial) -> None:
    hard_reset(ser)
    ser.reset_input_buffer()
    phrase = b"Xmodem download and burn FW image"
    deadline = time.time() + 8
    buffer = bytearray()
    while time.time() < deadline:
        ser.write(b"1")
        time.sleep(0.01)
        chunk = read_available(ser)
        if chunk:
            buffer.extend(chunk)
            if phrase in buffer:
                ser.write(b"1")
                time.sleep(0.1)
                ser.reset_input_buffer()
                return
    raise RuntimeError(
        "Could not enter bootloader. Check that no browser/serial monitor is using the port."
    )


def xmodem_send(ser: serial.Serial, payload: bytes, label: str = "data") -> None:
    # Check if C already in buffer; if so, use it
    existing = read_available(ser)
    start = CRC_MODE if (existing and CRC_MODE in existing) else wait_byte(ser, 60)
    if start != CRC_MODE:
        raise RuntimeError(f"XMODEM did not start for {label}; got {start!r}")

    total = (len(payload) + 127) // 128
    offset = 0
    block = 1
    while offset < len(payload):
        chunk = payload[offset : offset + 128].ljust(128, bytes([FILLER]))
        crc = crc16_xmodem(chunk)
        packet = bytes([SOH, block & 0xFF, 0xFF - (block & 0xFF)]) + chunk + bytes(
            [crc >> 8, crc & 0xFF]
        )

        for attempt in range(30):
            ser.write(packet)
            code = wait_byte(ser, 60)
            if code == ACK:
                break
            if code == NAK:
                continue
            if code == CAN:
                raise RuntimeError(f"XMODEM transfer cancelled while sending {label}")
        else:
            raise RuntimeError(f"Too many XMODEM errors while sending block {block}")

        time.sleep(0.02)

        if block == 1 or block == total or block % max(1, total // 20) == 0:
            percent = int(block / total * 100)
            print(f"{label}: {block}/{total} blocks ({percent}%)")

        block = (block + 1) & 0xFF
        offset += 128

    # Send EOT; this device doesn't ACK EOT in standard XMODEM
    # Just send it and proceed
    ser.write(bytes([EOT]))
    time.sleep(0.5)


def flash_complete(ser: serial.Serial, reboot: bool) -> None:
    """Handle the post-XMODEM menu.
    
    After each XMODEM transfer, the device may show:
        [0] Reboot system
        [1] Xmodem download and burn FW image
    
    If reboot=True: send '0' to reboot.
    If reboot=False: send '1' to re-enter XMODEM for next transfer.
    """
    answer = b"0" if reboot else b"1"
    # Wait up to 5s for menu text, then send answer
    deadline = time.time() + 5
    phrase = b"Reboot system"
    buffer = bytearray()
    while time.time() < deadline:
        chunk = read_available(ser)
        if chunk:
            buffer.extend(chunk)
            if phrase in buffer:
                ser.write(answer)
                time.sleep(0.5)
                return
        time.sleep(0.02)
    # No menu found; try sending answer anyway
    ser.write(answer)
    time.sleep(2.0)


def flash_model(ser: serial.Serial, data: bytes, address: int) -> None:
    enter_bootloader(ser)

    if address:
        config = bytearray([0xFF] * 128)
        config[0] = 0xC0
        config[1] = 0x5A
        config[2] = address & 0xFF
        config[3] = (address >> 8) & 0xFF
        config[4] = (address >> 16) & 0xFF
        config[5] = (address >> 24) & 0xFF
        config[6] = 0
        config[7] = 0
        config[8] = 0
        config[9] = 0
        config[10] = 0x5A
        config[11] = 0xC0
        ser.reset_input_buffer()
        xmodem_send(ser, bytes(config), "offset-config")
        # After config, device is silent; re-trigger with '1' to enter XMODEM
        for _ in range(20):
            ser.write(b"1")
            time.sleep(0.02)
        time.sleep(0.3)

    xmodem_send(ser, data, "model")
    # After data, device shows the menu. Send '0' to reboot.
    time.sleep(1)
    ser.write(b"0")
    time.sleep(0.5)
    # Read and discard any remaining output
    _ = read_available(ser)


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    depth = 0
    start = -1
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
                    out.append(json.loads(text[start : index + 1]))
                except json.JSONDecodeError:
                    pass
    return out


def at_command(ser: serial.Serial, command: str, timeout: float = 3) -> list[dict[str, Any]]:
    ser.reset_input_buffer()
    wire = command if command.endswith(("\r", "\n")) else command + "\r\n"
    ser.write(wire.encode("ascii"))
    deadline = time.time() + timeout
    chunks: list[bytes] = []
    while time.time() < deadline:
        chunk = read_available(ser)
        if chunk:
            chunks.append(chunk)
            payloads = extract_json_objects(b"".join(chunks).decode(errors="replace"))
            if payloads:
                return payloads
        time.sleep(0.02)
    return extract_json_objects(b"".join(chunks).decode(errors="replace"))


def write_metadata(ser: serial.Serial, metadata: dict[str, Any]) -> None:
    encoded = base64.b64encode(json.dumps(metadata, separators=(",", ":")).encode()).decode()
    print("Writing AT+INFO metadata")
    response = at_command(ser, f'AT+INFO="{encoded}"', timeout=5)
    if not response or response[-1].get("code") != 0:
        raise RuntimeError(f"AT+INFO failed: {response}")

    action_response = at_command(ser, 'AT+ACTION="","",""', timeout=5)
    if action_response and action_response[-1].get("code") not in (0, 5):
        print(f"Warning: AT+ACTION cleanup response: {action_response}")


def fetch_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode())


def download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        target.write_bytes(response.read())
    return target


def model_from_sensecraft(model_id: str, cache_dir: Path) -> tuple[Path, dict[str, Any]]:
    detail = fetch_json(f"https://sensecraft.seeed.cc/aiserverapi/model/view_model?model_id={model_id}")
    apply = fetch_json(f"https://sensecraft.seeed.cc/aiserverapi/model/apply_model?model_id={model_id}")
    if detail.get("code") != "0" or apply.get("code") != "0":
        raise RuntimeError(f"SenseCraft API failed: detail={detail} apply={apply}")

    snapshot = json.loads(apply["data"]["model_snapshot"])
    model_url = snapshot["arguments"]["url"]
    name = snapshot.get("model_name") or detail["data"].get("name") or f"SenseCraft {model_id}"
    classes = []
    for algorithm in snapshot.get("algorithm", []):
        for target in algorithm.get("display_ui", {}).get("targets", []):
            index = int(target.get("target_id", len(classes)))
            while len(classes) <= index:
                classes.append("")
            classes[index] = target.get("target_name") or target.get("target_name_cn") or str(index)
    if not classes:
        labels = detail["data"].get("labels") or []
        for item in labels:
            index = int(item.get("object_id", len(classes)))
            while len(classes) <= index:
                classes.append("")
            classes[index] = item.get("object_name") or str(index)

    model_file = download(model_url, cache_dir / Path(model_url).name)
    metadata = {
        "model_id": str(model_id),
        "version": snapshot.get("version", "1.0.0"),
        "arguments": snapshot.get("arguments", {}),
        "algorithm": snapshot.get("algorithm", []),
        "model_name": name,
        "model_format": snapshot.get("model_format", "tfLite"),
        "task": str(detail["data"].get("task", snapshot.get("arguments", {}).get("task", "1"))),
        "author": detail["data"].get("author_name", "SenseCraft AI"),
        "classes": classes,
        "checksum": snapshot.get("checksum") or hashlib.md5(model_file.read_bytes()).hexdigest(),
    }
    return model_file, metadata


def model_from_file(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    path = Path(args.model_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    if not classes:
        raise ValueError("--classes is required for --model-file, e.g. --classes medicine_box")
    metadata = {
        "model_id": "local",
        "version": args.version,
        "arguments": {
            "size": round(path.stat().st_size / 1024, 2),
            "url": str(path),
            "task": "detect",
        },
        "algorithm": [
            {
                "algorithm_id": 1,
                "algorithm_name": "Object Detection",
                "display_ui": {
                    "display_ui_type": 2,
                    "preview_txt_none": "Nothing",
                    "preview_txt": "${count} ${target_name} Detected",
                    "targets": [
                        {"target_id": index, "target_name": name, "target_name_cn": name}
                        for index, name in enumerate(classes)
                    ],
                },
            }
        ],
        "model_name": args.name,
        "model_format": "tfLite",
        "task": "1",
        "author": "local",
        "classes": classes,
        "checksum": hashlib.md5(path.read_bytes()).hexdigest(),
    }
    return path, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy model to Grove Vision AI V2 / WE2")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sensecraft-model-id", help="Public SenseCraft model id, e.g. 60261")
    source.add_argument("--model-file", help="Local .tflite/.lite file")
    parser.add_argument("--classes", default="", help="Comma-separated class names for local model")
    parser.add_argument("--name", default="Medicine Box Detection")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--port", default=auto_port())
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--address", type=parse_address, default=DEFAULT_ADDRESS)
    parser.add_argument("--cache-dir", default="models")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--metadata-out", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cache_dir = Path(args.cache_dir)
    if args.sensecraft_model_id:
        model_path, metadata = model_from_sensecraft(args.sensecraft_model_id, cache_dir)
    else:
        model_path, metadata = model_from_file(args)

    if args.metadata_out:
        Path(args.metadata_out).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    data = model_path.read_bytes()
    print(f"Model: {metadata['model_name']}")
    print(f"Classes: {', '.join(metadata.get('classes') or [])}")
    print(f"File: {model_path} ({len(data)} bytes)")
    print(f"Flash address: 0x{args.address:X}")

    if args.download_only:
        print("Download-only mode; not flashing.")
        return 0

    print(f"Opening {args.port} at {args.baud}")
    with open_link(args.port, args.baud) as ser:
        flash_model(ser, data, args.address)
        print("Waiting for reboot...")
        time.sleep(3)
        ser.baudrate = args.baud
        ser.reset_input_buffer()
        write_metadata(ser, metadata)
        probe = at_command(ser, "AT+INFO?", timeout=5)
        print(json.dumps(probe[-1] if probe else probe, ensure_ascii=False, indent=2))

    print("Deploy complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
