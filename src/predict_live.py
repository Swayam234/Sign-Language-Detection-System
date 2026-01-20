import cv2
import numpy as np
import tensorflow as tf
import os


# LOAD TRAINED MODEL
model = tf.keras.models.load_model("model.h5")


# AUTO LOAD LABELS (NO JSON)
DATASET_DIR = r"C:\Sign language detector\asl_alphabet_train\asl_alphabet_train"
labels = sorted(os.listdir(DATASET_DIR))

print("Loaded labels:", labels)

IMG_SIZE = 224


# WEBCAM
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define ROI (Region of Interest)
    x1, y1, x2, y2 = 100, 100, 400, 400
    roi = frame[y1:y2, x1:x2]

    # Preprocess image
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img, verbose=0)
    class_id = np.argmax(preds)
    confidence = np.max(preds)

    # Get label safely
    if confidence >= 0.70:
        label = labels[class_id]
    else:
        label = "Detecting..."

    # Draw ROI + Prediction
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"{label} ({confidence:.2f})",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
