Sign Language Detection System (Real-Time)

A real-time Sign Language Detection system using Deep Learning and Computer Vision that recognizes hand gestures through a webcam and predicts corresponding sign language alphabets.
This project demonstrates the application of CNN-based image classification with live video inference.

Features

1) Real-time sign language detection using webcam

2) Deep Learning model trained on ASL dataset

3) Uses TensorFlow / Keras for model training & inference

4) ROI-based hand detection for better accuracy

5) Confidence thresholding to avoid false predictions

6) Runs fully on CPU (no GPU required)

7) Modular project structure (training + live prediction)

Tech Stack

Python 3.10+

TensorFlow / Keras

OpenCV

NumPy

ASL Alphabet Dataset

VS Code / Command Line


Project Structure

Sign-Language-Detector/
│
├── src/
│   ├── train_model.py        # Model training script
│   ├── predict_live.py       # Real-time webcam prediction
│
├── dataset/
│   └── asl_alphabet_train/   # ASL training dataset (A–Z)
│
├── model.h5                  # Trained CNN model (not pushed to GitHub)
├── requirements.txt          # Project dependencies
├── .gitignore
└── README.md


Accuracy & Performance

Trained on ~87,000 images

Supports A–Z alphabets

Accuracy improves with:

Proper lighting

Centered hand inside ROI

Reduced background noise


Limitations

Detects single hand only

Background clutter can affect accuracy

Not yet optimized for dynamic sentence-level gestures


Future Enhancements

MediaPipe-based hand landmark detection

CNN + LSTM for motion-based gestures

Sign-to-speech conversion

Web app using Streamlit / Flask

Sentence-level sign interpretation