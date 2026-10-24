# Platewatch ANPR 🚗🔍

> **Automatic Number Plate Recognition (ANPR)** system optimized for Indian and international vehicle license plates using **Ultralytics YOLOv8** and **EasyOCR**, featuring a modern **Flask** dashboard, real-time analytics, and SQLite detection history.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-orange.svg)](https://github.com/JaidedAI/EasyOCR)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 Key Features

- **Trained YOLOv8 License Plate Detector**: Pre-trained on over 1,600+ real-world Indian vehicle images (OLX, Google, and video sources) across various lighting and tilt conditions.
  - **Detection Precision**: `95.8%`
  - **Detection Recall**: `98.8%`
  - **mAP@50**: `0.980`
- **EasyOCR Integration**: Enhanced with CLAHE adaptive contrast and bicubic scaling for crisp character recognition.
- **Still Image & Video Processing**:
  - Image detection with bounding box annotations, cropped license plates, and confidence metrics.
  - Video processing with frame sampling (`VIDEO_FRAME_INTERVAL`), timestamp tracking, and web-ready H.264 preview generation.
- **Interactive Web Dashboard**:
  - **Overview**: System health, active model status, and processed metrics.
  - **Image Detection**: Drag-and-drop or select an image to inspect detections and extracted text.
  - **Video Processing**: Sampled video analysis with playback and frame-by-frame plate logs.
  - **Detection History**: Paginated, filterable, and searchable records with plate crop previews.
  - **Analytics**: Daily processing counts and OCR confidence distribution charts powered by Chart.js.
- **RESTful API**: Clean JSON endpoints for programmatic integration.
- **Zero-Setup Quickstart**: Pre-packaged weights (`models/best.pt`) and sample media in `samples/` so any user who clones the repository can run it immediately out of the box.

---

## 🚀 Quick Start (Works Out of the Box)

### 1. Clone the repository
```bash
git clone https://github.com/avneetkashyap479/NumberPlate_detection.git
cd NumberPlate_detection
```

### 2. Set up a Python Virtual Environment
**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```
Open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

> **Note**: You don't need to manually configure `.env` or create database tables. The app automatically initializes the database, creates required folders, and uses the pre-trained weights in `models/best.pt`.

---

## 🧪 Testing with Included Samples

The repository includes ready-to-test sample images and a test video in the `samples/` directory:
- `samples/indian_sample_1.jpg` through `samples/indian_sample_6.jpg`: Real Indian vehicles with standard and high-security registration plates (HSRP).
- `samples/car_sample_1.jpg` through `samples/car_sample_5.jpg`: Additional vehicle angles.
- `samples/sample_traffic.mp4`: Sample traffic video clip.

**To test via UI:**
1. Navigate to **Image Detection** (`/image`).
2. Upload any image from the `samples/` folder.
3. Click **Run plate detection** to view the detected bounding box, confidence score, and extracted number plate text.

---

## 🛠️ Model Training & Evaluation

If you wish to retrain or fine-tune the YOLOv8 model on custom datasets:

### Prepare Dataset
Convert Pascal VOC XML or COCO annotations into YOLO format:
```bash
# For Indian vehicle archive dataset:
python prepare_archive_dataset.py

# Or for generic COCO datasets:
python prepare_dataset.py
```

### Train Model
Train using Apple Silicon MPS, Nvidia CUDA, or CPU (automatically detected):
```bash
python train.py --epochs 10 --batch 16 --imgsz 640
```
Trained weights are automatically saved to `models/best.pt`.

### Run Benchmark Evaluation
Evaluate detection IoU, Precision, Recall, F1 score, and OCR character accuracy:
```bash
python evaluate_model.py
```
Detailed metrics are exported to [`reports/evaluation_results.json`](file:///Users/sakshisingh/Desktop/vehicleNumberPlate/reports/evaluation_results.json).

---

## 🧪 Running Automated Tests

Run the complete pytest test suite:
```bash
pytest -v
```

---

## 📡 REST API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/health` | `GET` | Service status and model availability. |
| `/api/stats` | `GET` | Processed images, videos, and detection event totals. |
| `/api/detect/image` | `POST` | Upload an image file (`multipart/form-data`) for instant detection. |
| `/api/detect/video` | `POST` | Upload a video file for sampled frame detection. |
| `/api/history` | `GET` | Retrieve paginated detection jobs with optional filters (`search`, `type`, `date`). |
| `/api/history/<id>` | `DELETE`| Remove a job and its associated media files. |
| `/api/analytics` | `GET` | Daily trends and confidence distribution data. |
| `/api/file/<job_id>/<kind>/<det_id>` | `GET` | Fetch annotated media or cropped plate image (`?download=1` to download). |

---

## ⚙️ Configuration (Optional)

You can customize runtime behavior by copying `.env.example` to `.env`:
```bash
cp .env.example .env
```

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `PLATE_MODEL_PATH` | `models/best.pt` | Path to trained YOLO weights. |
| `DETECTION_CONFIDENCE` | `0.25` | Minimum bounding box confidence threshold. |
| `VIDEO_FRAME_INTERVAL` | `15` | Sample every Nth video frame. |
| `MAX_VIDEO_SECONDS` | `180` | Maximum video duration allowed (seconds). |
| `MAX_UPLOAD_MB` | `100` | Upload file size limit in megabytes. |
| `DATABASE_PATH` | `database/anpr.sqlite3` | SQLite database file location. |
| `OCR_LANGUAGES` | `en` | Comma-separated list of OCR languages. |

---

## 📁 Project Structure

```
vehicleNumberPlate/
├── app.py                      # Application entrypoint & factory
├── requirements.txt            # Dependency specifications
├── train.py                    # YOLO training pipeline
├── evaluate_model.py           # Quantitative validation runner
├── prepare_archive_dataset.py  # Dataset converter for Indian plates
├── config/
│   └── settings.py             # App configuration & path resolution
├── database/
│   ├── db.py                   # SQLite connection & query helpers
│   └── schema.sql              # Database schema
├── detection/
│   ├── plate_detector.py       # YOLOv8 license plate detector
│   ├── ocr_engine.py           # EasyOCR text extractor & pre-processor
│   ├── video_processor.py      # Video frame sampler & H.264 encoder
│   └── evaluation.py           # IoU matching & character accuracy metrics
├── models/
│   └── best.pt                 # Pre-trained license plate detection weights
├── routes/
│   ├── main_routes.py          # Dashboard UI routes
│   └── api_routes.py           # JSON REST API routes
├── samples/                    # Ready-to-use sample images & video
├── templates/                  # Bootstrap 5 HTML templates
├── static/                     # CSS, JS, and UI assets
└── tests/                      # Automated pytest suite
```

---

## 📄 License
This project is licensed under the MIT License.
