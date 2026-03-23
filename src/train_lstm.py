import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# Configuration
X_PATH = "src/data/X.npy"
Y_PATH = "src/data/y.npy"
CLASSES_PATH = "src/data/classes.npy"
SEQ_LEN = 10  # Number of frames to use in sequence

def load_data():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    classes = np.load(CLASSES_PATH)
    
    print(f"Loaded {X.shape[0]} samples.")
    
    # Expand to sequence length
    # Shape becomes (N, SEQ_LEN, 63)
    X_seq = np.repeat(X[:, np.newaxis, :], SEQ_LEN, axis=1)
    
    # Add random noise to simulate hand micro-movements across the sequence
    noise = np.random.normal(0, 0.005, X_seq.shape)
    X_seq = X_seq + noise
    X_seq = np.clip(X_seq, 0, 1.0) # landmarks are normalized between 0-1
    
    return X_seq, y, classes

def build_model(seq_len, num_features, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(seq_len, num_features)),
        Dropout(0.2),
        LSTM(128, return_sequences=False, activation='relu'),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    if not os.path.exists(X_PATH):
        print("Data files not found. Run extract_landmarks.py first.")
        exit(1)
        
    X, y, classes = load_data()
    print(f"Sequence Data Shape: {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = build_model(SEQ_LEN, 63, len(classes))
    model.summary()
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint("model_lstm.h5", save_best_only=True)
    ]
    
    model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test), 
        epochs=30, 
        batch_size=32,
        callbacks=callbacks
    )
    
    print("Training complete. Model saved as model_lstm.h5")
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
