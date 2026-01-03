# FILE: plate-ocr/src/process_video.py
import os
import sys

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from dataset import decode
from inference import load_crnn_model
from paths import model_asset
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# ✅ LOAD MODELS ONCE at startup (NOT per frame)
print("⏳ Loading YOLO model...")
yolo_model = YOLO(model_asset("LP-detection.pt"))
print("⏳ Loading CRNN model...")
crnn_model, idx_to_char, char_to_idx = load_crnn_model()
print("✅ Models loaded.")

transform = transforms.Compose(
    [
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def process_frame(frame, blur_enabled=False):
    """Process a single frame for plate detection and OCR."""
    results = yolo_model(frame)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate = frame[y1:y2, x1:x2]
        if plate.size == 0:
            continue

        if blur_enabled:
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(plate, (23, 23), 30)

        # OCR processing
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(plate_gray).convert("L")
        input_tensor = transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            logits = crnn_model(input_tensor)
            pred = logits.argmax(2).squeeze(1)

        plate_text = decode(pred, idx_to_char)

        # Draw results
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            plate_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    return frame


def process_video(video_path, output_path="processed_video.mp4", blur=False, frame_skip=3):
    """
    Process video with performance optimizations.

    Args:
        video_path: Input video file
        output_path: Output video file
        blur: Enable privacy blurring
        frame_skip: Process every Nth frame (1=all, 3=process 1/3 of frames)
    """
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25  # fallback

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if w == 0 or h == 0:
        print("❌ Invalid video dimensions. Exiting.")
        return

    print(f"📹 Video Info: {w}x{h} @ {fps:.1f} FPS, {total_frames} frames total")
    print(f"⚡ Optimization: Processing 1 of every {frame_skip} frames")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps / frame_skip, (w, h))

    frame_count = 0
    processed_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ FRAME SKIPPING: Only process every Nth frame
        if frame_count % frame_skip == 0:
            processed = process_frame(frame, blur_enabled=blur)
            out.write(processed)
            processed_count += 1

            # Show progress every 10 processed frames
            if processed_count % 10 == 0:
                print(f"⏳ Processed {processed_count}/{max(1, total_frames//frame_skip)} frames...")

        frame_count += 1

    cap.release()
    out.release()

    print(f"✅ Processed {processed_count} of {frame_count} frames.")
    print(f"✅ Video saved as: {output_path}")
    print(f"✅ Speedup: ~{frame_skip}x faster than original")
