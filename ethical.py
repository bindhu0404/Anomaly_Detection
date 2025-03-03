import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

MODEL_PATH = "anomaly_classifier.h5"
FRAME_SIZE = (64, 64)
MAX_FRAMES = 50
CATEGORIES = ["Normal", "Abuse", "Assault", "Arrest", "Arson"]  # 5 categories

# Load videos and labels
def load_videos(directory, label):
    data, labels = [], []
    for file in os.listdir(directory):
        cap = cv2.VideoCapture(os.path.join(directory, file))
        frames = []
        while len(frames) < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, FRAME_SIZE)
            frames.append(frame)
        cap.release()
        if len(frames) == MAX_FRAMES:
            data.append(np.array(frames))
            labels.append(label)
    return np.array(data), np.array(labels)

# Train and Save Model
def train_model():
    print("🔄 Training model...")
    dataset_path = r'C:\Users\DELL\Desktop\Ethical_Anomaly\dataset'

    # Load dataset for all categories
    normal_videos, normal_labels = load_videos(os.path.join(dataset_path, r'C:\Users\DELL\Desktop\Ethical_Anomaly\dataset\Normal_Videos'), 0)
    abuse_videos, abuse_labels = load_videos(os.path.join(dataset_path, r'C:\Users\DELL\Desktop\Ethical_Anomaly\dataset\Anomaly_Videos\Abuse'), 1)
    assault_videos, assault_labels = load_videos(os.path.join(dataset_path, r'C:\Users\DELL\Desktop\Ethical_Anomaly\dataset\Anomaly_Videos\Assault'), 2)
    arrest_videos, arrest_labels = load_videos(os.path.join(dataset_path, r'C:\Users\DELL\Desktop\Ethical_Anomaly\dataset\Anomaly_Videos\Arrest'), 3)
    arson_videos, arson_labels = load_videos(os.path.join(dataset_path, r'C:\Users\DELL\Desktop\Ethical_Anomaly\dataset\Anomaly_Videos\Arson'), 4)

    # Combine and preprocess data
    data = np.vstack((normal_videos, abuse_videos, assault_videos, arrest_videos, arson_videos)) / 255.0
    labels = to_categorical(np.hstack((normal_labels, abuse_labels, assault_labels, arrest_labels, arson_labels)), len(CATEGORIES))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Model Architecture
    model = Sequential([
        TimeDistributed(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3))),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        LSTM(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(len(CATEGORIES), activation='softmax')  # 5 output classes
    ])

    # Compile and train
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=4)

    # Save trained model
    model.save(MODEL_PATH)
    print("✅ Model trained and saved successfully!")

# Load or Train Model
if os.path.exists(MODEL_PATH):
    print("🔄 Loading saved model...")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
else:
    train_model()
    model = load_model(MODEL_PATH)

# Predict Video
def predict_video(video_path):
    if not os.path.exists(video_path):
        return None, "❌ Error: File not found!"
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_SIZE)
        frames.append(frame)
    
    cap.release()
    
    if len(frames) < MAX_FRAMES:
        return None, "⚠️ Warning: Not enough frames in the video. Try a longer video."

    frames = np.array(frames) / 255.0
    frames = np.expand_dims(frames, axis=0)

    prediction = model.predict(frames)[0]
    predicted_label = np.argmax(prediction)
    confidence = prediction[predicted_label]  # Confidence score

    result = CATEGORIES[predicted_label]
    return result, confidence
