# FILE: plate-ocr/README.md
# License Plate Recognition & Privacy Filter System

An AI-powered License Plate Recognition (LPR) tool that:

- Detects license plates using **YOLOv8 (Ultralytics)**
- Recognizes plate text using a **CRNN OCR model (PyTorch)**
- Optionally **blurs plates** for privacy
- Supports both **image and video** input via a Streamlit UI

---

## Key Features

- **License Plate Detection** — YOLOv8 model (`ultralytics`)
- **OCR** — CRNN model in PyTorch + CTC-style decoding
- **Privacy Mode** — blur plates in images/videos
- **Image Support** — upload an image and get annotated output + export JSON/CSV
- **Video Support** — upload a video, process with frame skipping, download processed MP4

---

## Tech Stack

- **Ultralytics YOLOv8** – plate detection
- **PyTorch** – CRNN OCR model
- **OpenCV** – image/video processing + drawing + blurring
- **Streamlit** – UI
- **Torchvision** – image transforms

---

## Project Structure (current)

plate-ocr/
- app/
  - streamlit_app.py
- src/
  - inference.py
  - model.py
  - dataset.py
  - process_video.py
  - paths.py
- model/
  - plate_model_v1.pth
  - char_to_idx.json
- example/
  - test_image.jpg
  - test_video.mp4
- LP-detection.pt
- requirements.txt
- README.md

---

## Notes on model files

This repo expects these files to exist:

- `LP-detection.pt` (YOLOv8 weights)
- `model/plate_model_v1.pth` (CRNN weights)
- `model/char_to_idx.json` (vocab mapping)

The code resolves paths robustly, so it works from different working directories (useful for Docker/CI).

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
