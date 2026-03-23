An advanced real-time Sign Language Detection System built using Deep Learning, Computer Vision, and NLP, capable of recognizing hand gestures (A–Z) and converting them into meaningful text and speech.

This project leverages MediaPipe + LSTM architecture for efficient, CPU-optimized inference and provides both a desktop application and a web-based interface.



🚀 Features
🖐️ Hand Landmark Detection
Uses MediaPipe HandLandmarker API
Extracts 21 3D keypoints (63 features)
Robust to:
Rotation
Scale variations
Background noise
Eliminates need for heavy CNN inference

🧠 LSTM-Based Gesture Recognition
Sequence-based prediction using LSTM neural network
Input: 10 frames × 63 features

Architecture:

LSTM(64) → Dropout → LSTM(128) → BatchNorm → Dense → Softmax
Achieved:
✅ 96.30% Test Accuracy
✅ Early stopping for optimal performance

📝 NLP Sentence Formation
Smart debounce logic prevents noisy predictions
Letters added only after stable detection
Supports:
Sentence building
Duplicate filtering

🔊 Text-to-Speech (TTS)
Converts predicted text into speech using pyttsx3
Runs in a background thread (non-blocking)
Controls:
Press SPACE → Speak sentence
Press BACKSPACE → Clear sentence

🌐 Streamlit Web Application
Modern dark-themed UI
Features:
Live webcam feed
Hand skeleton visualization
Real-time prediction + confidence
Sentence display
Controls: Start, Stop, Speak, Clear
Adjustable confidence threshold




📊 Model Performance

Metric	Value

Test Accuracy	96.30%

Dataset Size	9,871 images

Classes	29 (A–Z + del, nothing, space)

Training Split	80/20




⚙️ Installation & Setup

1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Extract Landmarks (One-time)
python src/extract_landmarks.py

3️⃣ Train Model (One-time)
python src/train_lstm.py

4️⃣ Run Application

🌐 Web App (Recommended)

streamlit run app.py

💻 Desktop Mode

python src/predict_advanced.py



🧠 System Architecture
Webcam → MediaPipe → Hand Landmarks (63 features)
       → LSTM Model → Prediction
       → NLP Processing → Sentence Formation
       → Text-to-Speech Output



🔥 Key Highlights

⚡ Real-time performance on CPU (no GPU required)

🎯 High accuracy with lightweight architecture

🧩 Modular and scalable design

🌍 Deployable via web interface

🧠 Combines CV + DL + NLP + Speech




⚠️ Limitations
Supports single-hand gestures only
Works best with:
Good lighting
Minimal background noise
Limited to alphabet-level recognition (A–Z)



🚀 Future Enhancements
Dynamic gesture recognition using CNN + LSTM
Sentence-level interpretation using advanced NLP
Sign-to-speech with multilingual support
Mobile/Web deployment with real-time streaming
Gesture recognition for words and phrases




👨‍💻 Author
Swayam Sandeep Karapurkar
Aspiring AI & ML Engineer



⭐ If you like this project

Give it a ⭐ on GitHub and share it!
