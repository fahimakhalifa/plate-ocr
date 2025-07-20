# License Plate Recognition & Privacy Filter System

An AI-powered license plate recognition (LPR) tool that:

- Detects license plates using YOLOv8
- Recognizes plate text using a CRNN model (PyTorch)
- Allows privacy blurring for detected plates
- Supports both image and video input via a clean Streamlit UI

---

## Key Features

- **License Plate Detection** — using a YOLOv8 model fine-tuned for plates
- **OCR** — custom-trained CRNN model to recognize plate text
- **Privacy Mode** — blurs plates in images/videos if enabled
- **Video Support** — upload and process full driving videos
- **Image Support** — test on static images instantly

---

## Tech Stack

- **YOLOv8** – for plate detection (`ultralytics`)
- **PyTorch** – CRNN plate recognition model
- **Streamlit** – interactive web interface
- **OpenCV** – for image/video processing
- **Torchvision** – data transformations

---

## YOLOv5 License Plate Detector
Download the pre-trained YOLOv5 model for license plate detection:

[Download LP-detection.pt](https://example.com/LP-detection.pt)

Place it in the `models/` folder before running the app.

---


## Project Structure

plate-ocr/
│
├── app/
│ └── streamlit_app.py # Streamlit UI code
│
├── src/
│ ├── inference.py # CRNN prediction logic
│ ├── model.py # CRNN model architecture
│ ├── dataset.py # Dataset class & decoding
│ └── process_video.py # Video frame processing
│
├── model/
│ ├── plate_model_v1.pth # Trained CRNN model
│ └── char_to_idx.json # Vocabulary mapping
│
├── examples/ # Sample media for testing
│ ├── test_image.jpg
│ └── sample_video.mp4
│
├── requirements.txt
└── README.md