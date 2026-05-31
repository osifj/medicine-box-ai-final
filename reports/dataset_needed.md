# Medicine Box Dataset Status

## Search Results

**Current project**: No `.txt` label files found.

**Old project 1** (`/Users/dep/Projects/New_project`):
- 96 images in `captures/host_session_001/images/`
- Labels exist in YOLO format
- **Classes**: `barcode` (0), `text` (1)
- All annotations are for barcode regions and text regions ON medicine boxes, not the whole medicine box

**Old project 2** (`/Users/dep/Documents/New project`):
- Copy of same dataset
- Classes: `barcode`, `text`

## Verdict

There are **zero** medicine_box whole-box annotations. The existing 96 images show medicine boxes but are labeled for sub-regions (barcodes and text), not the entire box outline. These cannot be directly reused for medicine_box detection without re-annotation.

## What's Needed

1. A new dataset of 50-100+ images of medicine boxes
2. YOLO-format bounding boxes around the **entire medicine box** (not just barcodes/text)
3. Single class: `medicine_box` (class id 0)

## Reusing Existing Images

The 96 existing images from `host_session_001` could potentially be re-annotated with whole-box labels. They already show medicine boxes from various angles. This would save collection time but still requires manual annotation work.
