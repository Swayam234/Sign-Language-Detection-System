"""
app.py  –  Streamlit Web Interface for Sign Language Detection
==============================================================
Run with:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import os
import time
import urllib.request
import threading

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf

# ─────────────── PAGE CONFIG ───────────────
st.set_page_config(
    page_title="Sign Language Detector",
    page_icon="🤟",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 18px 24px;
    text-align: center;
    border: 1px solid #0f3460;
    margin-bottom: 10px;
}
.metric-label { color: #8899aa; font-size: 0.78rem; letter-spacing: 0.05em; text-transform: uppercase; }
.metric-value { color: #00d4ff; font-size: 2.2rem; font-weight: 800; line-height: 1.1; }

.sentence-box {
    background: linear-gradient(135deg, #0d1117, #161b22);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px 20px;
    font-size: 1.15rem;
    color: #e6edf3;
    min-height: 54px;
    word-break: break-word;
}

.stButton>button {
    border-radius: 10px;
    font-weight: 600;
    border: none;
    padding: 8px 18px;
    transition: all 0.2s;
}
</style>
""", unsafe_allow_html=True)

# ─────────────── PATHS & CONSTANTS ───────────────
MODEL_PATH       = "model_lstm.h5"
CLASSES_PATH     = "src/data/classes.npy"
LANDMARKER_PATH  = "hand_landmarker.task"
SEQ_LEN          = 10
PRED_THRESHOLD   = 0.75
DEBOUNCE_FRAMES  = 12

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ─────────────── CACHED RESOURCES ───────────────
@st.cache_resource
def load_model_and_classes():
    model   = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH)
    return model, classes

@st.cache_resource
def load_detector():
    if not os.path.exists(LANDMARKER_PATH):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task",
            LANDMARKER_PATH
        )
    base     = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    opts     = vision.HandLandmarkerOptions(base_options=base, num_hands=1)
    return vision.HandLandmarker.create_from_options(opts)

# ─────────────── HELPERS ───────────────
def extract_features(det_result):
    if not det_result.hand_landmarks:
        return None
    lms = det_result.hand_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in lms]).flatten()

def draw_hand(frame, det_result):
    if not det_result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    lms  = det_result.hand_landmarks[0]
    pts  = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 212, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 100, 0), -1)

# ─────────────── SESSION STATE ───────────────
for key, val in [("sentence", []), ("last_appended", ""), ("debounce_hist", []),
                 ("frame_buffer", []), ("running", False),
                 ("pred_label", "---"), ("confidence", 0.0)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────── SIDEBAR ───────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hand-with-pen--v1.png", width=64)
    st.title("🤟 Sign Language\nDetector")
    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown(
        "1. Click **Start Camera**\n"
        "2. Hold your hand in front of the webcam\n"
        "3. Letters are detected and built into a sentence\n"
        "4. Click **🔊 Speak** to hear the sentence\n"
        "5. Click **🗑 Clear** to reset"
    )
    st.markdown("---")
    threshold = st.slider("Confidence Threshold", 0.50, 0.99, PRED_THRESHOLD, 0.01)
    st.markdown("---")
    st.caption("Powered by MediaPipe · TensorFlow · Streamlit")

# ─────────────── MAIN UI ───────────────
st.title("Real‑Time Sign Language Detection")
st.markdown("Uses **MediaPipe** hand landmarks → **LSTM** inference → NLP sentence builder")

col_start, col_speak, col_clear = st.columns([2, 1, 1])

with col_start:
    if not st.session_state.running:
        if st.button("▶  Start Camera", use_container_width=True, type="primary"):
            st.session_state.running = True
            st.rerun()
    else:
        if st.button("⏹  Stop Camera", use_container_width=True):
            st.session_state.running = False
            st.rerun()

with col_speak:
    if st.button("🔊 Speak", use_container_width=True):
        sentence_str = " ".join(st.session_state.sentence)
        if sentence_str:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 150)
                engine.say(sentence_str)
                engine.runAndWait()
            except Exception as e:
                st.warning(f"TTS error: {e}")

with col_clear:
    if st.button("🗑 Clear", use_container_width=True):
        st.session_state.sentence       = []
        st.session_state.last_appended  = ""
        st.session_state.debounce_hist  = []
        st.rerun()

st.markdown("---")

# Metrics row
m1, m2 = st.columns(2)
with m1:
    st.markdown(f"""<div class="metric-box">
        <div class="metric-label">Current Sign</div>
        <div class="metric-value">{st.session_state.pred_label}</div>
    </div>""", unsafe_allow_html=True)
with m2:
    conf_pct = f"{st.session_state.confidence * 100:.1f}%"
    st.markdown(f"""<div class="metric-box">
        <div class="metric-label">Confidence</div>
        <div class="metric-value">{conf_pct}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("**Sentence**")
sentence_display = " ".join(st.session_state.sentence) or "_waiting for gestures…_"
st.markdown(f'<div class="sentence-box">{sentence_display}</div>', unsafe_allow_html=True)
st.markdown("")

# Camera feed placeholder
frame_placeholder = st.empty()

# ─────────────── CAMERA LOOP ───────────────
if st.session_state.running:
    model, classes = load_model_and_classes()
    detector       = load_detector()
    cap            = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open webcam. Make sure it's connected and not in use.")
        st.session_state.running = False
    else:
        stop_flag   = False
        frame_count = 0

        while st.session_state.running and not stop_flag:
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            detection = detector.detect(mp_image)
            draw_hand(frame, detection)

            feat = extract_features(detection)
            if feat is not None:
                st.session_state.frame_buffer.append(feat)
                if len(st.session_state.frame_buffer) > SEQ_LEN:
                    st.session_state.frame_buffer.pop(0)

            pred_label     = "---"
            confidence_val = 0.0

            if len(st.session_state.frame_buffer) == SEQ_LEN:
                seq   = np.expand_dims(np.array(st.session_state.frame_buffer), 0)
                preds = model.predict(seq, verbose=0)[0]
                idx   = int(np.argmax(preds))
                confidence_val = float(preds[idx])

                if confidence_val >= threshold:
                    pred_label = classes[idx]
                    st.session_state.debounce_hist.append(pred_label)
                    if len(st.session_state.debounce_hist) > DEBOUNCE_FRAMES:
                        st.session_state.debounce_hist.pop(0)

                    if (len(st.session_state.debounce_hist) == DEBOUNCE_FRAMES
                            and all(x == pred_label for x in st.session_state.debounce_hist)
                            and pred_label != st.session_state.last_appended):
                        st.session_state.sentence.append(pred_label)
                        st.session_state.last_appended  = pred_label
                        st.session_state.debounce_hist  = []
                else:
                    st.session_state.debounce_hist = []

            st.session_state.pred_label = pred_label
            st.session_state.confidence = confidence_val

            # Overlay HUD
            disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = disp.shape[:2]
            cv2.rectangle(disp, (0, 0), (w, 50), (15, 15, 30), -1)
            cv2.putText(disp, f"{pred_label}  {confidence_val:.2f}",
                        (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 212, 100), 2)
            frame_placeholder.image(disp, channels="RGB", use_container_width=True)

            frame_count += 1
            # Rerun to refresh metrics every 30 frames
            if frame_count % 30 == 0:
                cap.release()
                st.rerun()

        cap.release()
