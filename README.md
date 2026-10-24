# Platewatch ANPR

A local academic prototype for **Automatic Number Plate Recognition using YOLO and EasyOCR**. It provides a Flask dashboard, SQLite history, image and sampled-video workflows, analytics, and evaluation utilities.

## Current verified state

The application expects `models/best.pt` by default. The checkpoint must be YOLO-compatible and trained specifically for license plates. Its existence alone does not establish its class labels, license, provenance, or accuracy. There is no generic COCO fallback, fabricated detection, or fabricated metric. EasyOCR and OpenCV inference paths require actual input media to be tested.

## Features

- Local responsive Bootstrap-based dashboard with actual SQLite metrics.
- Validated image/video uploads, generated filenames, size caps, and decoded-content checks.
- YOLO plate detector integration that requires configured weights.
- EasyOCR OCR with raw and display text kept separate and OCR/detection confidence independently stored.
- Annotated outputs and plate crops; sampled video with frame number and timestamp.
- Searchable, filterable, paginated job history; managed-file deletion.
- Analytics charts from database values; empty states when no real values exist.
- IoU-based one-to-one detection evaluation and character edit accuracy helper.

## Stack and structure

Python 3.10+ is recommended. Flask, OpenCV, Ultralytics YOLO, EasyOCR, NumPy, Pillow, SQLite, Bootstrap 5, Chart.js, and pytest are used. The application is split among `config/`, `detection/`, `database/`, `routes/`, `templates/`, `static/`, `models/`, `datasets/`, `tests/`, and `reports/`.

## Windows setup (PowerShell)

From the opened project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, use the interpreter directly: `\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `\.venv\Scripts\python.exe app.py`. PyTorch/EasyOCR installation size and compatibility depend on Python, Windows, and CPU/GPU drivers; consult the PyTorch install selector if the pinned Ultralytics/EasyOCR dependencies cannot resolve in your environment. The default OCR reader runs on CPU.

Edit `.env` with a local secret and, after reviewing a checkpoint, the path to plate detector weights. The default is `models/best.pt`. `.env` and weights are ignored by Git.

## Model setup and training

No public model has been approved or downloaded in this workspace. Review candidate source, intended classes, compatible Ultralytics format, and license before use; public availability does not automatically grant reuse rights. The current setup review failed with HTTP 401 for one candidate, leaving license and download availability unverified.

For a locally trained model, use the annotated YOLO detection layout and single class ID `0` as described in `models/README.md`. It documents directory structure, annotation row format, dataset YAML, train/validation commands, and the resulting `best.pt` path. A base `yolov8n.pt` may initialize training; it is never used directly by this app as plate detection. Check Ultralytics licensing terms (including AGPL-3.0 obligations or alternative license) before distributing or deploying.

## Run the application

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Open `http://127.0.0.1:5000`. The Flask development server binds to localhost with debug disabled. It is intended for local demonstration only, not public deployment.

Upload still images at **Image detection**. Supported extensions: JPG/JPEG, PNG, BMP, WEBP, TIFF. Video accepts MP4, AVI, MOV, MKV, M4V, with synchronous processing and a configurable duration cap. Video frames are sampled every `VIDEO_FRAME_INTERVAL` frames (default 15); repeated plate appearances are separate detection events and are not unique vehicles. Inspect records in **Detection history** and open a job to review the annotated output/crops.

## API

- `GET /api/health`: app status and whether the configured weights path exists.
- `GET /api/stats`: actual job/event/confidence summary.
- `POST /api/detect/image`: multipart field `file`.
- `POST /api/detect/video`: multipart field `file`.
- `GET /api/history`: optional `search`, `type`, `date`, `page` query parameters.
- `GET /api/history/<id>` and `DELETE /api/history/<id>`.
- `GET /api/analytics`: daily and OCR-confidence aggregates.
- Managed downloads are provided by `/api/file/<job_id>/<kind>/<detection_id>`.

JSON errors use an `error` field and appropriate HTTP status. Technical failures are logged server-side; upload routes return no stack traces.

## Tests

```powershell
pytest -q
```

Tests cover database operations, upload validation, missing model behavior, routes, and metric helpers. They do not validate model accuracy, OCR quality, or video inference against real weights/data.

## Evaluation and report evidence

`detection/evaluation.py` calculates greedy one-to-one IoU matching at a declared threshold, with each prediction/ground-truth box matched at most once; it returns TP/FP/FN, precision, recall, and F1. `character_accuracy` uses case-insensitive Levenshtein similarity without punctuation removal. Exact plate recognition accuracy should compare matched predicted crops with manually transcribed ground truth. Do not use these metrics without ground-truth labels. No detector/OCR metrics are available before an annotated held-out test set is run.

`reports/evaluation_results.json` is intentionally marked `not_run`. Run trials for clear daylight, low-light, tilted, blurry, distant, occluded, no-plate, and video samples. Record the actual plate result, OCR match, confidences, duration, sample interval, and source. Do not add invented values. See `reports/screenshots/README.md` for a checklist. Use **Win+Shift+S** to save genuine browser screenshots in `reports/screenshots/` after running real inputs; no screenshots are pre-generated.

## Configuration

`MAX_UPLOAD_MB` limits request body size (default 100). `DETECTION_CONFIDENCE` sets detector confidence (0.25). `VIDEO_FRAME_INTERVAL` controls frame sampling (15). `MAX_VIDEO_SECONDS` limits duration (180). `DATABASE_PATH`, `PLATE_MODEL_PATH`, and `OCR_LANGUAGES` are configurable in `.env`.

## Limitations

- No detector weights, dataset, sample input, or ground-truth evaluation is included.
- The app does not certify model provenance, license, accuracy, or registration validity.
- Synchronous video processing can block a local request; no real-time performance is claimed.
- OpenCV mp4v output depends on available codec support.
- EasyOCR may download language assets on first initialization and recognition may be incorrect.
- Production use requires a production WSGI server, authentication/authorization, CSRF protection, deployment hardening, retention policy, and legal/privacy review.

## Common issues

- **Plate detector weights not found:** add reviewed plate-specific YOLO weights and set `PLATE_MODEL_PATH`.
- **Weights load but detections are wrong:** verify model class and training labels; file existence alone does not establish suitability.
- **EasyOCR initialization/download fails:** verify package compatibility and network access for its language assets, or pre-cache the supported English assets.
- **Video output has no picture:** verify OpenCV codec support; try another input format or codec.
- **Upload too large:** lower media resolution/size or increase `MAX_UPLOAD_MB` for local use.
- **PowerShell script activation blocked:** use `\.venv\Scripts\python.exe` directly, or follow your organization’s execution-policy rules.
