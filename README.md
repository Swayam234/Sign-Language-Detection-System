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







Accuracy & Performance

1) Trained on ~87,000 images

2) Supports A–Z alphabets

3) Accuracy improves with:

4) Proper lighting

5) Centered hand inside ROI

6) Reduced background noise






Limitations

1) Detects single hand only

2) Background clutter can affect accuracy

3) Not yet optimized for dynamic sentence-level gestures






Future Enhancements

1) MediaPipe-based hand landmark detection

2) CNN + LSTM for motion-based gestures

3) Sign-to-speech conversion

4) Web app using Streamlit / Flask

5) Sentence-level sign interpretation