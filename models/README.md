# Plate detector weights

No checkpoint is bundled. A public model candidate could not be independently reviewed because the model listing endpoint returned HTTP 401 during project setup; its license, files, and intended class therefore remain unverified. Do not download a checkpoint solely based on an unverified name.

Place a trusted Ultralytics-compatible YOLO detection checkpoint at `models/best.pt`, or configure another path with `PLATE_MODEL_PATH` in `.env`. The checkpoint must have been trained for a license-plate class. The application never substitutes a generic COCO checkpoint. Review the checkpoint's source, intended classes, license, and redistribution terms before use. A local `.pt` file alone does not prove its provenance or accuracy.

After configuring it, open `/api/health` and confirm `plate_model_available` is true. Then process a real, locally sourced image and review results manually. A loaded file is not evidence that the model is accurate.

## Train your own detector

Use a dataset whose terms allow your intended academic use. Dataset terms are independent of Ultralytics' code license. Start with images you have permission to use and annotate all visible plates consistently.

Expected structure:

```text
datasets/plates/
  images/train/
  images/val/
  labels/train/
  labels/val/
  data.yaml
```

Each label file shares its image filename and contains one row per plate:

```text
0 x_center y_center width height
```

Coordinates are normalized to `[0, 1]`; class IDs start at zero. Use empty/missing label files for images without plates, following the dataset convention. `data.yaml`:

```yaml
path: C:/absolute/path/to/MinorProject/datasets/plates
train: images/train
val: images/val
names:
  0: license_plate
```

Install dependencies in the active virtual environment, then use a base detection checkpoint as initialization only (it is not used for application plate inference):

```powershell
 yolo detect train model=yolov8n.pt data=datasets/plates/data.yaml epochs=100 imgsz=640 project=runs/plate name=exp
 yolo detect val model=runs/plate/exp/weights/best.pt data=datasets/plates/data.yaml
 Copy-Item runs/plate/exp/weights/best.pt models/best.pt
```

Set `PLATE_MODEL_PATH=models/best.pt` in `.env` (the default). Review validation metrics and predictions against held-out data before reporting results. Ultralytics YOLOv8 package/model licensing may impose AGPL-3.0 obligations or require a separate license depending on the use; check the current official license terms and your specific use before distribution or deployment.
