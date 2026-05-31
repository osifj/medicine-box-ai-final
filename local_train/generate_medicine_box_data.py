#!/usr/bin/env python3
"""Generate synthetic medicine_box single-class training data.

Outputs YOLO-format dataset with only class 0 = medicine_box.
Each label is the bounding box of the entire medicine box in the scene.

Based on the data generator from /Users/dep/Projects/New_project.
"""

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# --- Medicine box generation (simplified from old project) ---

COLORS = [
    ((244, 248, 255), (34, 82, 160)),
    ((255, 246, 239), (193, 77, 26)),
    ((243, 252, 244), (36, 120, 70)),
    ((251, 244, 248), (163, 54, 92)),
    ((252, 250, 244), (148, 112, 52)),
]

BACKGROUNDS = ["desk", "wood", "clinic", "marble", "fabric", "gradient"]


def make_background(width, height):
    theme = random.choice(BACKGROUNDS)
    base = Image.new("RGB", (width, height), (245, 244, 240))
    draw = ImageDraw.Draw(base)
    if theme == "desk":
        base = Image.new("RGB", (width, height), random.choice([(236, 232, 227), (230, 234, 239)]))
        draw = ImageDraw.Draw(base)
        for _ in range(8):
            c = tuple(random.randint(220, 248) for _ in range(3))
            x1 = random.randint(-width//4, width)
            y1 = random.randint(-height//4, height)
            draw.ellipse((x1, y1, x1+random.randint(width//6, width//3), y1+random.randint(height//6, height//3)), fill=c)
    elif theme == "wood":
        base = Image.new("RGB", (width, height), (182, 150, 118))
        draw = ImageDraw.Draw(base)
        for row in range(0, height, random.randint(14, 28)):
            tone = random.randint(-20, 20)
            c = (max(80, min(220, 182+tone)), max(70, min(210, 150+tone)), max(60, min(200, 118+tone)))
            draw.rectangle((0, row, width, row+random.randint(10, 20)), fill=c)
    elif theme == "marble":
        base = Image.new("RGB", (width, height), (238, 236, 232))
        draw = ImageDraw.Draw(base)
        for _ in range(20):
            g = random.randint(210, 240)
            draw.line((random.randint(-width//4, width), random.randint(-height//4, height),
                       random.randint(-width//4, width)+random.randint(60, 300),
                       random.randint(-height//4, height)+random.randint(-20, 20)),
                      fill=(g, g, g), width=random.randint(1, 4))
    elif theme == "fabric":
        base = Image.new("RGB", (width, height), (232, 228, 220))
        draw = ImageDraw.Draw(base)
        for i in range(0, width, random.randint(24, 48)):
            g = max(180, min(240, 228+random.randint(-8, 8)))
            draw.line((i, 0, i, height), fill=(g, g-4, g-12), width=random.randint(1, 3))
    elif theme == "gradient":
        c1 = random.choice([(235,240,245), (245,238,230), (240,245,238)])
        c2 = random.choice([(218,224,232), (228,218,210), (225,232,220)])
        arr = np.zeros((height, width, 3), dtype=np.float32)
        for y in range(height):
            t = y / height
            arr[y,:,0] = c1[0]*(1-t)+c2[0]*t
            arr[y,:,1] = c1[1]*(1-t)+c2[1]*t
            arr[y,:,2] = c1[2]*(1-t)+c2[2]*t
        arr += np.random.normal(0, 4, arr.shape)
        base = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    else:
        base = Image.new("RGB", (width, height), (228, 238, 242))
        draw = ImageDraw.Draw(base)
        for _ in range(4):
            px = random.randint(0, width-90)
            py = random.randint(0, height-140)
            draw.rounded_rectangle((px, py, px+70, py+120), radius=12, fill=(228, 234, 240))
    arr = np.clip(np.asarray(base, dtype=np.float32)+np.random.normal(0, 6, (height, width, 3)), 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def make_medicine_box():
    width = random.randint(380, 580)
    height = random.randint(260, 380)
    base_color, accent = random.choice(COLORS)
    box = Image.new("RGBA", (width, height), base_color+(255,))
    draw = ImageDraw.Draw(box)
    draw.rounded_rectangle((0, 0, width-1, height-1), radius=16, outline=(190, 190, 190), width=3)
    draw.rectangle((0, 0, width, random.randint(36, 56)), fill=accent+(255,))
    draw.rectangle((0, height-random.randint(22, 36), width, height), fill=accent+(255,))
    # 3D side panel effect
    side_w = random.randint(30, 50)
    side_c = tuple(max(0, ch-random.randint(18, 32)) for ch in accent)
    draw.polygon([(width-side_w, 0), (width, 14), (width, height), (width-side_w, height-8)], fill=side_c+(255,))
    # barcode placeholder
    bar_y = height - 72
    draw.rectangle((width-190, bar_y, width-14, height-10), fill=(255, 255, 255))
    for i in range(0, 170, random.randint(2, 5)):
        if random.random() > 0.4:
            draw.rectangle((width-188+i, bar_y+4, width-188+i+random.randint(1, 3), height-22), fill=(30, 30, 30))
    # label area
    label_top = random.randint(50, 70)
    draw.rectangle((12, label_top, width-14, label_top+random.randint(80, 140)), fill=(252, 252, 252), outline=(210, 210, 210))
    # glare
    glare = Image.new("RGBA", box.size, (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glare)
    gx1 = random.randint(-width//5, width//3)
    gdraw.polygon([(gx1, 0), (gx1+60, 0), (gx1+width//3, height), (gx1+width//3-80, height)],
                  fill=(255, 255, 255, random.randint(14, 32)))
    glare = glare.filter(ImageFilter.GaussianBlur(radius=12))
    box = Image.alpha_composite(box, glare)
    return box


def rotate_points(points, center, angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    cx, cy = center
    return [(cx+(x-cx)*ca-(y-cy)*sa, cy+(x-cx)*sa+(y-cy)*ca) for x, y in points]


def get_transformed_box_bbox(box_size, angle_deg, offset):
    w, h = box_size
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    cx, cy = w/2, h/2
    rotated = rotate_points(corners, (cx, cy), angle_deg)
    xs = [p[0]-min(r[0] for r in rotated)+offset[0] for p in rotated]
    ys = [p[1]-min(r[1] for r in rotated)+offset[1] for p in rotated]
    return min(xs), min(ys), max(xs), max(ys)


def to_yolo_line(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    cx = ((x1+x2)/2)/img_w
    cy = ((y1+y2)/2)/img_h
    bw = (x2-x1)/img_w
    bh = (y2-y1)/img_h
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def render_scene(out_dir, split, index):
    img_w = random.randint(800, 1200)
    img_h = random.randint(600, 900)
    bg = make_background(img_w, img_h).convert("RGBA")

    pkg = make_medicine_box()
    angle = random.uniform(-12, 12)
    rotated = pkg.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

    # shadow
    shadow = Image.new("RGBA", rotated.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (12, 12, rotated.size[0]-4, rotated.size[1]-4), radius=16, fill=(0, 0, 0, 60))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=8))

    max_x = max(10, img_w-rotated.size[0]-10)
    max_y = max(10, img_h-rotated.size[1]-10)
    offset = (random.randint(10, max_x), random.randint(10, max_y))
    bg.alpha_composite(shadow, dest=(offset[0]+5, offset[1]+6))
    bg.alpha_composite(rotated, dest=offset)

    # medicine_box label
    box_bbox = get_transformed_box_bbox(pkg.size, angle, offset)
    label_line = to_yolo_line(box_bbox, img_w, img_h)

    # noise and save
    merged = bg.convert("RGB")
    arr = np.clip(np.asarray(merged, dtype=np.float32)+np.random.normal(0, 3, (img_h, img_w, 3)), 0, 255)
    merged = Image.fromarray(arr.astype(np.uint8))
    if random.random() < 0.4:
        merged = merged.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.9)))

    img_dir = out_dir / "images" / split
    lab_dir = out_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    fname = f"medbox_{index:05d}"
    merged.save(img_dir / f"{fname}.jpg", quality=92)
    (lab_dir / f"{fname}.txt").write_text(label_line + "\n", encoding="utf-8")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate medicine_box single-class training data")
    p.add_argument("--output", default="local_train/medicine_box", help="Output dataset root")
    p.add_argument("--train", type=int, default=200, help="Training images")
    p.add_argument("--val", type=int, default=40, help="Validation images")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.output)
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    for i in range(args.train):
        render_scene(out, "train", i)
        if (i+1) % 50 == 0:
            print(f"  train: {i+1}/{args.train}")
    for i in range(args.val):
        render_scene(out, "val", i)
        if (i+1) % 20 == 0:
            print(f"  val: {i+1}/{args.val}")

    # data.yaml
    yaml_content = f"""path: {out.resolve()}
train: images/train
val: images/val
names:
  0: medicine_box
"""
    (out / "data.yaml").write_text(yaml_content)
    print(f"\nDone. {args.train} train + {args.val} val images at {out.resolve()}")
    print(f"data.yaml written with classes: medicine_box")


if __name__ == "__main__":
    main()
