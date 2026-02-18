# FILE: plate-ocr/app/streamlit_app.py
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# Keep import behavior stable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inference import predict_plate
from paths import model_asset
from process_video import process_video

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Plate OCR Demo — License Plate Recognition & Privacy Filter",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================
# Theme (teal + slate) — NO "EMPTY BOX" WRAPPERS
# =========================
st.markdown(
    """
<style>
/* Layout */
.main .block-container { max-width: 1200px; padding-top: 1.1rem; padding-bottom: 2.2rem; }
section[data-testid="stSidebar"] { display: none; }

/* Palette */
:root{
  --bg: #F7FAFC;
  --card: #FFFFFF;
  --text: #0F172A;
  --muted: #64748B;
  --border: rgba(15, 23, 42, .10);
  --shadow: 0 12px 28px rgba(2, 6, 23, .06);
  --primary: #0F766E;
  --primaryHover: #0B5F59;
  --soft: rgba(15, 118, 110, .10);
}

/* Background */
.stApp { background: var(--bg); }

/* Critical visibility fix */
.stApp, .stApp * { color: var(--text); }
.stApp .stCaption, .stApp caption, .stApp small { color: var(--muted) !important; }
.stApp label, .stApp [data-testid="stWidgetLabel"] { color: var(--text) !important; }

/* Typography */
.h1 { font-size: 2.05rem; font-weight: 900; letter-spacing: -0.02em; margin: 0; color: var(--text) !important; }
.sub { margin-top: .35rem; color: var(--muted) !important; font-size: 1rem; }
.h2 { font-size: 1.15rem; font-weight: 900; margin: 0 0 .25rem 0; }
.p { margin: 0; color: var(--muted) !important; }

/* Section card (single layer only) */
.section {
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 16px;
  background: var(--card);
  box-shadow: var(--shadow);
}

/* Step cards (no nested boxes) */

.step {
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px 14px;
  background: var(--card);
  box-shadow: 0 10px 22px rgba(2, 6, 23, .05);
}
.badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: var(--soft);
  color: var(--primary) !important;
  font-weight: 900;
  font-size: .78rem;
  letter-spacing: .03em;
  text-transform: uppercase;
}
.divider { height: 1px; background: var(--border); margin: 12px 0; }

/* KPI tiles */
.kpi {
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 12px;
  background: #FFFFFF;
}
.kpi-k { color: var(--muted) !important; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }
.kpi-v { font-size: 1.35rem; font-weight: 900; margin-top: 6px; color: var(--text) !important; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 14px !important;
  padding: 10px 14px !important;
  font-weight: 850 !important;
}
.stButton > button[kind="primary"] {
  background: var(--primary) !important;
  color: white !important;
  border: 1px solid var(--primary) !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--primaryHover) !important;
  border: 1px solid var(--primaryHover) !important;
}
.stButton > button[kind="secondary"] {
  background: white !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}
.stDownloadButton > button {
  background: white !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}
.stDownloadButton > button:hover {
  border-color: rgba(15,118,110,.35) !important;
}

/* File uploader skin */
[data-testid="stFileUploader"] {
  background: #F8FAFC !important;
  border: 1px solid rgba(15, 118, 110, .18) !important;
  border-radius: 16px !important;
  padding: 10px !important;
}
[data-testid="stFileUploader"] section {
  background: #FFFFFF !important;
  border: 2px dashed rgba(15, 118, 110, .30) !important;
  border-radius: 14px !important;
}
[data-testid="stFileUploader"] button {
  background: #FFFFFF !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  font-weight: 900 !important;
}

/* Selectbox */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background: #FFFFFF !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
}

/* Tabs (clean pill row) */
.stTabs [data-baseweb="tab-list"] {
  gap: 10px;
  padding: 6px 6px;
  background: #F1F5F9;
  border: 1px solid var(--border);
  border-radius: 999px;
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  border-radius: 999px;
  padding: 10px 14px;
  font-weight: 900;
  color: var(--text) !important;
}
.stTabs [aria-selected="true"] {
  background: var(--primary) !important;
  color: white !important;
}
.stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* Logs */
.logbox {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: .88rem;
  white-space: pre-wrap;
  line-height: 1.35;
  color: var(--text) !important;
}

/* Footer */
.footer {
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px 16px;
  background: #FFFFFF;
}
.footer strong { font-weight: 900; }
.footer .muted { color: var(--muted) !important; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Session State
# =========================
def _init_state() -> None:
    defaults = {
        "processing_stats": {"total_processed": 0, "total_time": 0.0, "plates_detected": 0},
        "last_image_result": None,
        "last_video_result": None,
        "logs": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{ts}] {msg}")


_init_state()

# =========================
# Load models (cached)
# =========================
@st.cache_resource
def load_models():
    yolo = YOLO(model_asset("last.pt"))
    tfm = transforms.Compose(
        [
            transforms.Resize((32, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    return yolo, tfm


yolo_model, transform = load_models()

# =========================
# Helpers
# =========================
def image_details(img: Image.Image, raw: bytes) -> Dict[str, str]:
    return {
        "Dimensions": f"{img.size[0]} × {img.size[1]}",
        "Format": (img.format or "Unknown"),
        "Mode": img.mode,
        "File size": f"{len(raw)/1024:.1f} KB",
    }


def video_details(filename: str, raw: bytes, mime: str) -> Dict[str, str]:
    fmt = mime.split("/")[-1].upper() if mime and "/" in mime else "Unknown"
    return {"Filename": filename, "Format": fmt, "File size": f"{len(raw)/(1024*1024):.1f} MB"}


def run_image_pipeline(image_np: np.ndarray, confidence_threshold: float, privacy: bool) -> Dict[str, Any]:
    start_time = time.time()
    log("Running YOLOv8 detection on image.")
    results = yolo_model(image_np)

    plates_found = 0
    dets: List[Dict[str, Any]] = []
    total_conf = 0.0

    processed = image_np.copy()
    crops: List[np.ndarray] = []

    log("Running OCR on detected crops.")
    for box in results[0].boxes:
        confidence = float(box.conf[0])
        if confidence < confidence_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = processed[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        plate_resized = cv2.resize(crop, (160, 32))
        plate_gray = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2GRAY)
        pil_plate = Image.fromarray(plate_gray)

        input_tensor = transform(pil_plate).unsqueeze(0)
        pred_text = predict_plate(input_tensor)
        ocr_conf = 0.0  

        if not pred_text:
            pred_text = "N/A"



        if privacy:
            processed[y1:y2, x1:x2] = cv2.GaussianBlur(processed[y1:y2, x1:x2], (25, 25), 30)

        cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{pred_text} ({ocr_conf:.2f})" if pred_text != "N/A" else "N/A"
        cv2.putText(processed, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        plates_found += 1
        total_conf += confidence
        crops.append(crop.copy())
        dets.append(
            {
                "Plate": plates_found,
                "Location": f"({x1}, {y1})",
                "Text": pred_text,
                "Confidence": confidence,
                "OCR_Confidence": float(ocr_conf),
                "Box": [x1, y1, x2, y2],
            }
        )

    dt = time.time() - start_time
    avg_conf = (total_conf / plates_found) if plates_found > 0 else 0.0
    log(f"Done. Plates={plates_found}, time={dt:.2f}s, avg_conf={avg_conf:.1%}")

    st.session_state.processing_stats["total_processed"] += 1
    st.session_state.processing_stats["total_time"] += float(dt)
    st.session_state.processing_stats["plates_detected"] += int(plates_found)

    return {
        "original_np": image_np,
        "processed_np": processed,
        "detections": dets,
        "crops": crops,
        "stats": {"plates_found": plates_found, "avg_conf": avg_conf, "processing_time": dt},
    }


def run_video_pipeline(video_bytes: bytes, blur: bool, frame_skip: int, output_quality: str) -> Dict[str, Any]:
    cache_key = f"{hash(video_bytes)}_{blur}_{frame_skip}_{output_quality}"
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", f"processed_video_{cache_key}.mp4")

    if os.path.exists(output_path):
        log("Using cached processed video output.")
        return {"output_path": output_path, "stats": {"processing_time": None}}

    log("Writing uploaded video to temp file.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_tmp:
        input_tmp.write(video_bytes)
        input_path = input_tmp.name

    log(f"Calling process_video(blur={blur}, frame_skip={frame_skip}).")
    start_time = time.time()
    process_video(input_path, output_path, blur=blur, frame_skip=frame_skip)
    dt = time.time() - start_time
    output_path = ensure_web_mp4(output_path)
    os.unlink(input_path)

    st.session_state.processing_stats["total_processed"] += 1
    st.session_state.processing_stats["total_time"] += float(dt)

    log(f"Done. Video processed in {dt:.1f}s -> {output_path}")
    return {"output_path": output_path, "stats": {"processing_time": dt}}

def ensure_web_mp4(input_path: str) -> str:
    """
    Convert output to a browser-friendly MP4 (H.264 + AAC) so Streamlit preview works.
    If ffmpeg is not installed, we keep the original file.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        log("ffmpeg not found; skipping web transcode (preview may be black).")
        return input_path

    out_path = os.path.splitext(input_path)[0] + "_web.mp4"
    if os.path.exists(out_path):
        return out_path

    cmd = [
        ffmpeg, "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "128k",
        out_path,
    ]

    try:
        log("Transcoding processed video to web-safe MP4 (H.264/AAC)…")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out_path
    except Exception:
        log("Web transcode failed; using original processed output.")
        return input_path


# =========================
# Header (single block, no empty wrapper)
# =========================
st.markdown(
    """
<div class="section">
  <div class="h1">Plate OCR — Product Demo</div>
  <div class="sub">Upload an image or video, run license plate detection + OCR, optionally blur plates, then export results.</div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# =========================
# Controls (Steps are back — but only 1 layer of cards)
# =========================
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown(
        """
<div class="step">
  <span class="badge">Step 1</span>
  <div style="height:10px"></div>
  <div class="h2">Upload</div>
  <div class="p">Choose input type and upload a file.</div>
  <div class="divider"></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("**Input type**")
    mode = st.radio("mode", ["Image", "Video"], horizontal=True, label_visibility="collapsed")

    uploaded_img = None
    uploaded_vid = None

    st.markdown("")

    if mode == "Image":
        st.markdown("**Image file**")
        uploaded_img = st.file_uploader("image_file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        st.caption("Recommended: clear plates, good lighting.")
    else:
        st.markdown("**Video file**")
        uploaded_vid = st.file_uploader("video_file", type=["mp4", "mov", "avi"], label_visibility="collapsed")
        st.caption("Tip: shorter videos process faster.")

with c2:
    st.markdown(
        """
<div class="step">
  <span class="badge">Step 2</span>
  <div style="height:10px"></div>
  <div class="h2">Options</div>
  <div class="p">Privacy + threshold controls.</div>
  <div class="divider"></div>
</div>
""",
        unsafe_allow_html=True,
    )

    privacy = st.checkbox("Privacy blur", value=True)
    confidence_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.5, 0.05)

    processing_mode = st.selectbox(
        "Processing mode",
        ["Balanced", "Fast", "High Accuracy"],
        index=0,
        help="UI placeholder (no behavior change in v1).",
    )

    frame_skip = 3
    output_quality = "Medium"
    if mode == "Video":
        st.markdown("")
        frame_skip = st.slider("Frame skip (video)", 1, 10, 3)
        output_quality = st.select_slider(
            "Output quality",
            ["Low", "Medium", "High"],
            value="Medium",
            help="UI placeholder (no behavior change in v1).",
        )

with c3:
    st.markdown(
        """
<div class="step">
  <span class="badge">Step 3</span>
  <div style="height:10px"></div>
  <div class="h2">Run</div>
  <div class="p">Start processing and review results.</div>
  <div class="divider"></div>
</div>
""",
        unsafe_allow_html=True,
    )

    run_disabled = (uploaded_img is None) if mode == "Image" else (uploaded_vid is None)

    # spacing so buttons don't stick to the step header
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        start = st.button("Start processing", type="primary", use_container_width=True, disabled=run_disabled)
    with col_b:
        reset = st.button("Reset results", type="secondary", use_container_width=True)

    if reset:
        st.session_state.last_image_result = None
        st.session_state.last_video_result = None
        st.session_state.logs = []
        st.success("Cleared.")

    if start:
        st.session_state.logs = []
        log("Run requested.")

        if mode == "Image":
            img_bytes = uploaded_img.getvalue()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(img)

            with st.status("Processing image…", expanded=True) as status:
                status.write("Running detection + OCR…")
                res = run_image_pipeline(img_np, float(confidence_threshold), bool(privacy))
                status.update(label="Done", state="complete")

            res["meta"] = {
                "filename": uploaded_img.name,
                "details": image_details(img, img_bytes),
                "settings": {
                    "privacy": bool(privacy),
                    "confidence_threshold": float(confidence_threshold),
                    "processing_mode": processing_mode,
                },
            }
            st.session_state.last_image_result = res
            st.session_state.last_video_result = None
            st.success("Image processed.")

        else:
            vid_bytes = uploaded_vid.getvalue()

            with st.status("Processing video…", expanded=True) as status:
                status.write("Running video processing…")
                res = run_video_pipeline(vid_bytes, bool(privacy), int(frame_skip), str(output_quality))
                status.update(label="Done", state="complete")

            res["meta"] = {
                "filename": uploaded_vid.name,
                "details": video_details(uploaded_vid.name, vid_bytes, uploaded_vid.type),
                "settings": {
                    "blur": bool(privacy),
                    "frame_skip": int(frame_skip),
                    "processing_mode": processing_mode,
                    "output_quality": output_quality,
                },
            }
            res["original_bytes"] = vid_bytes
            st.session_state.last_video_result = res
            st.session_state.last_image_result = None
            st.success("Video processed.")

    with st.expander("Model & pipeline", expanded=False):
        st.markdown(
            """
**Detection**: YOLOv8 (Ultralytics)  
**OCR**: CRNN (PyTorch)  

**Flow**: Detect → Crop → OCR → Overlay → Optional Blur → Export
"""
        )

st.write("")

# =========================
# Results (single section card; no nested cards)
# =========================
st.markdown(
    """
<div class="section">
  <div class="h2">Results</div>
  <div class="p">Preview, detections, crops, exports, run info, and logs.</div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

if st.session_state.last_image_result is None and st.session_state.last_video_result is None:
    st.info("Run processing to see results here.")
else:
    if st.session_state.last_image_result is not None:
        res = st.session_state.last_image_result
        stats = res["stats"]
        meta = res["meta"]

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(
                f"<div class='kpi'><div class='kpi-k'>Plates Found</div><div class='kpi-v'>{stats['plates_found']}</div></div>",
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"<div class='kpi'><div class='kpi-k'>Avg Confidence</div><div class='kpi-v'>{stats['avg_conf']:.1%}</div></div>",
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"<div class='kpi'><div class='kpi-k'>Processing Time</div><div class='kpi-v'>{stats['processing_time']:.2f}s</div></div>",
                unsafe_allow_html=True,
            )

        tabs = st.tabs(["Preview", "Detections", "Crops", "Exports", "Run info", "Logs"])

        with tabs[0]:
            a, b = st.columns(2, gap="large")
            with a:
                st.markdown("**Original**")
                st.image(res["original_np"], use_column_width=True)  # ✅ streamlit 1.35.0 compatible
            with b:
                st.markdown("**Processed**")
                st.image(res["processed_np"], use_column_width=True)  # ✅

        with tabs[1]:
            dets = res["detections"]
            if dets:
                df = pd.DataFrame(
                    [{"Plate": d["Plate"], "Location": d["Location"], "Text": d["Text"], "Det_Confidence": d["Confidence"], "OCR_Confidence": d.get("OCR_Confidence", 0.0)} for d in dets]

                )
                st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Det_Confidence": st.column_config.ProgressColumn("Det_Confidence", format="%.1f%%", min_value=0, max_value=1)
                    },

                )
            else:
                st.warning("No plates detected at the current threshold.")

        with tabs[2]:
            crops = res.get("crops", [])
            if not crops:
                st.info("No crops available.")
            else:
                cols = st.columns(4)
                for i, crop in enumerate(crops):
                    with cols[i % 4]:
                        st.image(crop, caption=f"Crop #{i+1}", use_column_width=True)

        with tabs[3]:
            dets = res["detections"]
            json_data = json.dumps(
                [{"plate_index": d["Plate"], "location": d["Location"], "text": d["Text"], "confidence": d["Confidence"], "box": d["Box"]} for d in dets],
                indent=2,
            )
            st.download_button("Download detections (JSON)", data=json_data, file_name="detection_results.json", mime="application/json", use_container_width=True)

            csv_data = pd.DataFrame(dets).drop(columns=["Box"], errors="ignore").to_csv(index=False)
            st.download_button("Download detections (CSV)", data=csv_data, file_name="detection_results.csv", mime="text/csv", use_container_width=True)

            _, buffer = cv2.imencode(".jpg", cv2.cvtColor(res["processed_np"], cv2.COLOR_RGB2BGR))
            st.download_button(
                "Download processed image (JPG)",
                data=buffer.tobytes(),
                file_name=f"detected_plates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )

        with tabs[4]:
            st.markdown("**File**")
            st.write(f"- **Name:** {meta['filename']}")
            st.markdown("**Image details**")
            for k, v in meta["details"].items():
                st.write(f"- **{k}:** {v}")
            st.markdown("**Settings**")
            for k, v in meta["settings"].items():
                st.write(f"- **{k}:** {v}")

        with tabs[5]:
            st.markdown("<div class='logbox'>" + "\n".join(st.session_state.logs[-250:]) + "</div>", unsafe_allow_html=True)

    else:
        res = st.session_state.last_video_result
        meta = res["meta"]
        out_path = res["output_path"]
        original_bytes = res.get("original_bytes", b"")

        tabs = st.tabs(["Preview", "Exports", "Run info", "Logs"])

        with tabs[0]:
            a, b = st.columns(2, gap="large")
            with a:
                st.markdown("**Original**")
                if original_bytes:
                    st.video(original_bytes, format="video/mp4")
                else:
                    st.info("Original preview unavailable.")
            with b:
                st.markdown("**Processed**")
                if os.path.exists(out_path):
                    with open(out_path, "rb") as f:
                        st.video(f.read(), format="video/mp4")
                else:
                    st.error("Processed output file missing. Re-run processing.")

        with tabs[1]:
            if os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    st.download_button(
                        "Download processed video (MP4)",
                        data=f,
                        file_name=f"processed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )

        with tabs[2]:
            st.markdown("**File**")
            st.write(f"- **Name:** {meta['filename']}")
            st.markdown("**Video details**")
            for k, v in meta["details"].items():
                st.write(f"- **{k}:** {v}")
            st.markdown("**Settings**")
            for k, v in meta["settings"].items():
                st.write(f"- **{k}:** {v}")

        with tabs[3]:
            st.markdown("<div class='logbox'>" + "\n".join(st.session_state.logs[-250:]) + "</div>", unsafe_allow_html=True)

# =========================
# Footer (clean + useful, no ugly blocks)
# =========================
st.write("")
st.markdown(
    f"""
<div class="footer">
  <div style="display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap;">
    <div>
      <strong>Session:</strong>
      <span class="muted">Processed {st.session_state.processing_stats['total_processed']} files · {st.session_state.processing_stats['plates_detected']} plates · {st.session_state.processing_stats['total_time']:.1f}s total</span>
    </div>
    <div>
      <strong>Plate OCR · v1.1</strong>
      <span class="muted">· Product demo · Built with Streamlit · {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
