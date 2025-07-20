import cv2
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from PIL import Image
import torch
from torchvision import transforms
from inference import load_crnn_model
from dataset import decode

crnn_model, idx_to_char, char_to_idx = load_crnn_model()

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def process_frame(frame, blur_enabled=False):
    from ultralytics import YOLO
    yolo_model = YOLO("LP-detection.pt")
    results = yolo_model(frame)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate = frame[y1:y2, x1:x2]
        if plate.size == 0: continue
        if blur_enabled:
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(plate, (23, 23), 30)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(plate_gray).convert("L")
        input_tensor = transform(pil_image).unsqueeze(0)
        with torch.no_grad():
            logits = crnn_model(input_tensor)
            pred = logits.argmax(2).squeeze(1)
        plate_text = decode(pred, idx_to_char)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, plate_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame


def process_video(video_path, output_path="processed_video.avi", blur=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25  # fallback

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if w == 0 or h == 0:
        print("❌ Invalid video dimensions. Exiting.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use XVID for .avi
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed = process_frame(frame, blur_enabled=blur)
        out.write(processed)
        frame_count += 1

    cap.release()
    out.release()
    print(f"✅ Processed {frame_count} frames.")
    print(f"✅ Video saved as: {output_path}")
