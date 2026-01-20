import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os


# CONFIG
LABEL = "A"  # CHANGE THIS FOR EACH SIGN
DATASET_PATH = "dataset"
CSV_PATH = os.path.join(DATASET_PATH, f"{LABEL}.csv")

os.makedirs(DATASET_PATH, exist_ok=True)


# MEDIAPIPE SETUP
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils


# WEBCAM
cap = cv2.VideoCapture(0)

data = []

print(" Press 'S' to save landmarks")
print(" Press 'Q' to quit")


# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                cv2.putText(frame, "Press S to Save",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and result.multi_hand_landmarks:
        data.append(landmarks)
        print(f"✅ Saved sample {len(data)}")

    elif key == ord('q'):
        break


# SAVE CSV
if data:
    df = pd.DataFrame(data)
    if os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_PATH, index=False)

cap.release()
cv2.destroyAllWindows()
