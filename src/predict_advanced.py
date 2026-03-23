"""
predict_advanced.py
===================
Advanced real-time sign language detection using:
  - MediaPipe hand landmark detection (replaces fixed ROI)
  - LSTM-based consecutive prediction smoothing
  - NLP-style sentence builder with debounce logic
  - Text-to-Speech via pyttsx3 on a background thread
"""

import cv2
import numpy as np
import os
import threading
import time
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
import pyttsx3

# ───────────────────────── CONFIG ─────────────────────────
MODEL_PATH       = "model_lstm.h5"
CLASSES_PATH     = "src/data/classes.npy"
LANDMARKER_PATH  = "hand_landmarker.task"
SEQ_LEN          = 10       # frames to buffer before inference
PRED_THRESHOLD   = 0.75     # minimum confidence to accept a letter
DEBOUNCE_FRAMES  = 12       # frames a letter must stay stable before appending
MAX_SENTENCE_LEN = 60       # character cap for sentence buffer


# ───────────────── DOWNLOAD LANDMARKER IF MISSING ─────────────────
def ensure_landmarker(path: str):
    if not os.path.exists(path):
        print("Downloading hand_landmarker.task …")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task",
            path
        )
        print("Done.")

ensure_landmarker(LANDMARKER_PATH)

# ───────────────── LOAD MODEL & CLASSES ─────────────────
print("Loading LSTM model …")
model   = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH)
print(f"Loaded {len(classes)} classes: {list(classes)}")

# ───────────────── MEDIAPIPE TASKS SETUP ─────────────────
_base   = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
_opts   = vision.HandLandmarkerOptions(base_options=_base, num_hands=1)
detector = vision.HandLandmarker.create_from_options(_opts)

# ───────────────── TEXT-TO-SPEECH THREAD ─────────────────
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)

_tts_lock   = threading.Lock()
_tts_queue  = []

def _tts_worker():
    while True:
        with _tts_lock:
            if _tts_queue:
                text = _tts_queue.pop(0)
                tts_engine.say(text)
                tts_engine.runAndWait()
        time.sleep(0.1)

tts_thread = threading.Thread(target=_tts_worker, daemon=True)
tts_thread.start()

def speak(text: str):
    with _tts_lock:
        _tts_queue.append(text)


# ───────────────── HELPER: Extract 63-d feature vector ─────────────────
def extract_features(detection_result) -> np.ndarray | None:
    if not detection_result.hand_landmarks:
        return None
    lms = detection_result.hand_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in lms]).flatten()   # (63,)


# ───────────────── HELPER: Draw hand skeleton ─────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def draw_hand(frame, detection_result):
    if not detection_result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    lms = detection_result.hand_landmarks[0]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 80, 0), -1)


# ────────────────────────── MAIN LOOP ──────────────────────────
frame_buffer   = []          # rolling window of feature vectors
debounce_hist  = []          # recent stable predictions
sentence       = []          # accumulated words / letters
last_appended  = ""

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("\n[START] Press  Q / ESC  to quit | SPACE to speak sentence | BACKSPACE to clear\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── Detect hand landmarks ──
    img_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection  = detector.detect(mp_image)
    draw_hand(frame, detection)

    # ── Build sequence buffer ──
    feat = extract_features(detection)
    if feat is not None:
        frame_buffer.append(feat)
        if len(frame_buffer) > SEQ_LEN:
            frame_buffer.pop(0)

    pred_label      = "---"
    confidence_val  = 0.0

    if len(frame_buffer) == SEQ_LEN:
        seq   = np.expand_dims(np.array(frame_buffer), axis=0)   # (1, SEQ_LEN, 63)
        preds = model.predict(seq, verbose=0)[0]
        idx   = int(np.argmax(preds))
        confidence_val = float(preds[idx])

        if confidence_val >= PRED_THRESHOLD:
            pred_label = classes[idx]
            debounce_hist.append(pred_label)
            if len(debounce_hist) > DEBOUNCE_FRAMES:
                debounce_hist.pop(0)

            # NLP debounce: append letter only when it's stable
            if (len(debounce_hist) == DEBOUNCE_FRAMES
                    and all(x == pred_label for x in debounce_hist)
                    and pred_label != last_appended):
                sentence.append(pred_label)
                last_appended  = pred_label
                debounce_hist  = []
                if len(" ".join(sentence)) > MAX_SENTENCE_LEN:
                    sentence = sentence[-20:]   # keep last 20 chars worth
        else:
            debounce_hist = []

    # ── Build display sentence ──
    sentence_str = " ".join(sentence)

    # ── HUD Overlay ──
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Sign: {pred_label}  ({confidence_val:.2f})",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 120), 2)
    cv2.putText(frame, f"Sentence: {sentence_str}",
                (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, "SPACE=Speak  BACKSPACE=Clear  Q=Quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow("Sign Language Detection — Advanced", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:          # ESC / Q → quit
        break
    elif key == 32:                    # SPACE → speak current sentence
        if sentence_str:
            speak(sentence_str)
    elif key == 8:                     # BACKSPACE → clear
        sentence.clear()
        last_appended = ""

cap.release()
cv2.destroyAllWindows()
print("Session ended.")
