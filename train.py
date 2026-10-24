"""
Model training pipeline for License Plate Detection using Ultralytics YOLOv8.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Ensure SSL certificate bundle is available on macOS
import certifi
if "SSL_CERT_FILE" not in os.environ:
    os.environ["SSL_CERT_FILE"] = certifi.where()

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def detect_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def train(data_yaml=None, base_model="yolov8n.pt", epochs=10, imgsz=640, batch=16, device=None, output_path=None):
    data_yaml = Path(data_yaml or (ROOT / "datasets" / "plates" / "data.yaml"))
    output_path = Path(output_path or (ROOT / "models" / "best.pt"))
    device = device or detect_device()

    if not data_yaml.is_file():
        print(f"Dataset configuration not found at {data_yaml}.")
        print("Running prepare_dataset.py...")
        prepare_script = ROOT / "prepare_dataset.py"
        if prepare_script.is_file():
            import subprocess
            subprocess.run([sys.executable, str(prepare_script)], check=True)
        else:
            raise FileNotFoundError(f"Could not find dataset config at {data_yaml} and {prepare_script} is missing.")

    print(f"=== Starting License Plate Detector Training ===")
    print(f"Data configuration : {data_yaml}")
    print(f"Base model         : {base_model}")
    print(f"Epochs             : {epochs}")
    print(f"Image Size         : {imgsz}")
    print(f"Batch Size         : {batch}")
    print(f"Compute Device     : {device}")
    print(f"Target Output Path : {output_path}")

    # Initialize model
    model = YOLO(str(base_model))

    # Train model
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(ROOT / "runs" / "plate"),
        name="train_run",
        exist_ok=True,
        plots=False,
        verbose=True,
    )

    # Locate and copy best weights
    best_weights = Path(model.trainer.best) if hasattr(model, "trainer") and hasattr(model.trainer, "best") else None
    if not best_weights or not best_weights.is_file():
        best_weights = ROOT / "runs" / "plate" / "train_run" / "weights" / "best.pt"

    val_results = None
    if best_weights.is_file():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, output_path)
        print(f"\nSuccessfully trained and updated model weights at: {output_path}")

        print("\n=== Validating Trained Model ===")
        trained_model = YOLO(str(output_path))
        val_results = trained_model.val(data=str(data_yaml), device=device, plots=False)
        print(f"Validation mAP50: {val_results.box.map50:.4f}, mAP50-95: {val_results.box.map:.4f}")
    else:
        print(f"\nWarning: Expected trained checkpoint at {best_weights} was not found.")

    return results, val_results

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO License Plate Detection Model")
    parser.add_argument("--data", type=str, default=str(ROOT / "datasets" / "plates" / "data.yaml"), help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Initial weights/architecture (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution size (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size (default: 16)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('mps', 'cuda', 'cpu')")
    parser.add_argument("--output", type=str, default=str(ROOT / "models" / "best.pt"), help="Target destination for best.pt")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        output_path=args.output,
    )
