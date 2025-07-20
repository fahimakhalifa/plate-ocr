import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from ultralytics import YOLO
from PIL import Image
import torch
from torchvision import transforms
from inference import predict_plate
from process_video import process_video

yolo_model = YOLO("LP-detection.pt")

transform = transforms.Compose([
    transforms.Resize((32, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Streamlit UI ===
st.set_page_config(page_title="License Plate Detector", layout="wide")
st.title("🚗 License Plate Recognition & Privacy Filter")

option = st.sidebar.selectbox("Choose Input Type", ("Image", "Video"))
privacy = st.sidebar.checkbox("🔒 Enable Privacy Blur", value=True)

if option == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        results = yolo_model(image_np)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = image_np[y1:y2, x1:x2]
            plate_resized = cv2.resize(plate_crop, (160, 32))
            plate_gray = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2GRAY)
            pil_plate = Image.fromarray(plate_gray)

            input_tensor = transform(pil_plate).unsqueeze(0)
            pred_text = predict_plate(input_tensor)

            if privacy:
                image_np[y1:y2, x1:x2] = cv2.GaussianBlur(image_np[y1:y2, x1:x2], (25, 25), 30)

            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image_np, pred_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        st.image(image_np, caption="Processed Image", channels="RGB")

elif option == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    blur_opt = st.sidebar.checkbox("🔒 Blur Plates", value=True)

    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
            input_tmp.write(uploaded_video.read())
            input_path = input_tmp.name

        output_path = os.path.join("processed_video.mp4")

        st.info("🔄 Processing video, please wait...")
        process_video(input_path, output_path, blur=blur_opt)

        st.success("✅ Done! Click below to download:")

        with open(output_path, "rb") as file:
            btn = st.download_button(
                label="📥 Download Processed Video",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )





