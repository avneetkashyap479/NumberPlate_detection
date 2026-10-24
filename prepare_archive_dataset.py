"""
Dataset converter for Indian License Plate Dataset (datasets/archive) into YOLOv8 format.
"""

import os
import glob
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
ARCHIVE_DIR = ROOT / "datasets" / "archive"
OUTPUT_DIR = ROOT / "datasets" / "plates"
SAMPLES_DIR = ROOT / "samples"

def find_image_for_xml(xml_path):
    dir_name = os.path.dirname(xml_path)
    base = os.path.splitext(os.path.basename(xml_path))[0]

    # Check direct name matches
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        cand = os.path.join(dir_name, base + ext)
        if os.path.isfile(cand):
            return cand

    # Check XML filename tag
    try:
        tree = ET.parse(xml_path)
        fn = tree.findtext("filename")
        if fn:
            cand = os.path.join(dir_name, fn)
            if os.path.isfile(cand):
                return cand
    except Exception:
        pass

    return None

def main():
    if not ARCHIVE_DIR.is_dir():
        raise FileNotFoundError(f"Archive directory {ARCHIVE_DIR} does not exist.")

    xml_files = sorted(glob.glob(str(ARCHIVE_DIR / "**" / "*.xml"), recursive=True))
    print(f"Found {len(xml_files)} XML annotation files in {ARCHIVE_DIR}")

    # Clean existing plates dataset images/labels
    for split in ("train", "val"):
        shutil.rmtree(OUTPUT_DIR / "images" / split, ignore_errors=True)
        shutil.rmtree(OUTPUT_DIR / "labels" / split, ignore_errors=True)
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    ground_truth = {}

    for xf in xml_files:
        img_path = find_image_for_xml(xf)
        if not img_path:
            continue

        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            continue

        if img_w <= 0 or img_h <= 0:
            continue

        try:
            tree = ET.parse(xf)
            root = tree.getroot()
        except Exception:
            continue

        boxes = []
        plate_names = []

        for obj in root.findall("object"):
            bnd = obj.find("bndbox")
            if bnd is None:
                continue

            try:
                xmin = float(bnd.findtext("xmin", "0"))
                ymin = float(bnd.findtext("ymin", "0"))
                xmax = float(bnd.findtext("xmax", "0"))
                ymax = float(bnd.findtext("ymax", "0"))
            except ValueError:
                continue

            # Ensure coordinates are within image boundaries
            xmin = max(0.0, min(float(img_w), xmin))
            ymin = max(0.0, min(float(img_h), ymin))
            xmax = max(0.0, min(float(img_w), xmax))
            ymax = max(0.0, min(float(img_h), ymax))

            if xmax <= xmin or ymax <= ymin:
                continue

            # Normalized YOLO coordinates
            x_center = ((xmin + xmax) / 2.0) / img_w
            y_center = ((ymin + ymax) / 2.0) / img_h
            norm_w = (xmax - xmin) / img_w
            norm_h = (ymax - ymin) / img_h

            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            name_tag = obj.findtext("name", "").strip().upper()

            boxes.append((0, x_center, y_center, norm_w, norm_h, [int(xmin), int(ymin), int(xmax), int(ymax)]))
            if name_tag:
                plate_names.append(name_tag)

        if not boxes:
            continue

        records.append({
            "img_path": img_path,
            "img_w": img_w,
            "img_h": img_h,
            "boxes": boxes,
            "plates": plate_names
        })

    print(f"Successfully processed {len(records)} valid images with annotations.")

    # 85% train, 15% validation split
    split_index = int(len(records) * 0.85)
    train_records = records[:split_index]
    val_records = records[split_index:]

    def export_subset(subset, split_name):
        for idx, rec in enumerate(subset):
            src_img = Path(rec["img_path"])
            file_stem = f"in_plate_{split_name}_{idx:04d}"
            ext = src_img.suffix.lower()
            if ext not in (".jpg", ".jpeg", ".png"):
                ext = ".jpg"
            
            dst_img_name = f"{file_stem}{ext}"
            dst_img_path = OUTPUT_DIR / "images" / split_name / dst_img_name
            dst_lbl_path = OUTPUT_DIR / "labels" / split_name / f"{file_stem}.txt"

            shutil.copy2(src_img, dst_img_path)

            with open(dst_lbl_path, "w", encoding="utf-8") as lf:
                for cls_id, xc, yc, nw, nh, _ in rec["boxes"]:
                    lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

            ground_truth[dst_img_name] = {
                "split": split_name,
                "width": rec["img_w"],
                "height": rec["img_h"],
                "plates": rec["plates"],
                "boxes": [b[5] for b in rec["boxes"]]
            }

    print(f"Exporting {len(train_records)} train images...")
    export_subset(train_records, "train")

    print(f"Exporting {len(val_records)} validation images...")
    export_subset(val_records, "val")

    # Save ground truth mapping
    gt_path = OUTPUT_DIR / "ground_truth.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Saved ground truth mapping to {gt_path}")

    # Write data.yaml
    data_yaml_content = f"""path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val
names:
  0: license_plate
"""
    (OUTPUT_DIR / "data.yaml").write_text(data_yaml_content, encoding="utf-8")
    print(f"Wrote data.yaml to {OUTPUT_DIR / 'data.yaml'}")

    # Copy diverse Indian vehicle samples to samples/
    sample_indices = [5, 25, 45, 65, 85, 105]
    for i, idx in enumerate(sample_indices, 1):
        if idx < len(val_records):
            val_rec = val_records[idx]
            dst = SAMPLES_DIR / f"indian_sample_{i}.jpg"
            shutil.copy2(val_rec["img_path"], dst)
    print("Exported Indian vehicle test samples to samples/ directory.")

if __name__ == "__main__":
    main()
