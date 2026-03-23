import os
import cv2
import mediapipe as mp
import numpy as np
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATASET_DIR = "C:\\Sign language detector\\dataset\\asl_alphabet_train\\asl_alphabet_train"
if not os.path.exists(DATASET_DIR):
    DATASET_DIR = "C:\\Sign language detector\\asl_alphabet_train\\asl_alphabet_train"

MAX_IMAGES_PER_CLASS = 400

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading hand_landmarker.task...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

X = []
y = []
classes = sorted(os.listdir(DATASET_DIR))

print(f"Extracting MediaPipe landmarks from {len(classes)} classes using Tasks API...")

for label_idx, label in enumerate(classes):
    class_dir = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(class_dir):
        continue
    
    images = os.listdir(class_dir)[:MAX_IMAGES_PER_CLASS]
    valid_count = 0
    for img_name in images:
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        detection_result = detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]
            # Extract x, y, z for 21 points = 63 features
            features = []
            for lm in landmarks:
                features.extend([lm.x, lm.y, lm.z])
            
            X.append(features)
            y.append(label_idx)
            valid_count += 1
            
    print(f"Class {label}: Extracted landmarks from {valid_count}/{len(images)} images.")

X = np.array(X)
y = np.array(y)

print(f"Total extracted features: {X.shape}")
print(f"Total labels: {y.shape}")

os.makedirs("src/data", exist_ok=True)
np.save("src/data/X.npy", X)
np.save("src/data/y.npy", y)
np.save("src/data/classes.npy", np.array(classes))

print("Saved landmarks to src/data/")
