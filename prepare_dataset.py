import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "datasets" / "raw_test"
COCO_PATH = RAW_DIR / "_annotations.coco.json"
TARGET_DIR = ROOT / "datasets" / "plates"
SAMPLES_DIR = ROOT / "samples"

def main():
    if not COCO_PATH.is_file():
        raise FileNotFoundError(f"Missing {COCO_PATH}")

    with open(COCO_PATH, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    img_to_boxes = {img["id"]: [] for img in coco["images"]}

    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in images:
            continue
        img = images[img_id]
        img_w, img_h = img["width"], img["height"]
        x, y, w, h = ann["bbox"]
        
        # Convert to YOLO normalized format: class x_center y_center width height
        x_center = (x + w / 2.0) / img_w
        y_center = (y + h / 2.0) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        
        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        norm_w = max(0.0, min(1.0, norm_w))
        norm_h = max(0.0, min(1.0, norm_h))

        img_to_boxes[img_id].append((0, x_center, y_center, norm_w, norm_h))

    # Setup directories
    for split in ("train", "val"):
        (TARGET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (TARGET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    sorted_img_ids = sorted(images.keys())
    split_idx = int(len(sorted_img_ids) * 0.8) # 80% train, 20% val
    train_ids = sorted_img_ids[:split_idx]
    val_ids = sorted_img_ids[split_idx:]

    def export_split(img_ids, split):
        for img_id in img_ids:
            img_info = images[img_id]
            file_name = img_info["file_name"]
            src_img = RAW_DIR / file_name
            if not src_img.is_file():
                continue

            dst_img = TARGET_DIR / "images" / split / file_name
            shutil.copy2(src_img, dst_img)

            label_name = Path(file_name).stem + ".txt"
            dst_label = TARGET_DIR / "labels" / split / label_name
            with open(dst_label, "w", encoding="utf-8") as lf:
                for cls_id, xc, yc, nw, nh in img_to_boxes[img_id]:
                    lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

    print(f"Exporting {len(train_ids)} train samples...")
    export_split(train_ids, "train")
    print(f"Exporting {len(val_ids)} validation samples...")
    export_split(val_ids, "val")

    # Copy 5 clean samples for demo testing
    sample_indices = [10, 25, 45, 80, 120]
    for i, idx in enumerate(sample_indices, 1):
        if idx < len(sorted_img_ids):
            src = RAW_DIR / images[sorted_img_ids[idx]]["file_name"]
            if src.is_file():
                shutil.copy2(src, SAMPLES_DIR / f"car_sample_{i}.jpg")
    print("Exported demo samples into samples/ directory.")

    # Write data.yaml with relative path configuration
    data_yaml_content = f"""path: {TARGET_DIR.resolve()}
train: images/train
val: images/val
names:
  0: license_plate
"""
    (TARGET_DIR / "data.yaml").write_text(data_yaml_content, encoding="utf-8")
    print(f"Wrote data.yaml to {TARGET_DIR / 'data.yaml'}")

if __name__ == "__main__":
    main()
