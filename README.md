
# 🏋️‍♂️ AI Fitness Tracker & Form Analyzer

An end-to-end Machine Learning pipeline that performs real-time exercise classification, rep counting, and form correction using Computer Vision.

## 🌟 Project Overview
Unlike standard hardcoded fitness trackers, this project features a **Two-Stage Scalable Architecture**:
1. **Machine Learning Classifier:** A trained lightweight model (Random Forest) that automatically detects which exercise the user is performing based on their skeletal posture.
2. **Biomechanical Form Analyzer:** A set of modular algorithms that track specific joint angles (using geometry and trigonometry) to count reps and provide real-time form feedback.

**Current Supported Exercises:**
- Squats
- Push-ups
- Bicep Curls
*(Easily scalable to hundreds of exercises due to Object-Oriented Design).*

## 🛠️ Tech Stack
- **Python 3.12**
- **MediaPipe:** For robust, real-time 33-point human pose estimation.
- **OpenCV:** For video processing, frame manipulation, and rendering the HUD overlay.
- **Scikit-Learn:** For training the exercise classification models.
- **Pandas & NumPy:** For feature engineering and mathematical operations.

## 🏗️ System Architecture 

The pipeline is divided into three distinct stages:

### Stage 1: Data Collection & Feature Extraction (`data_collector.py`)
Captures video streams (webcam or pre-recorded), extracts normalized (X, Y, Z) coordinates of relevant joints using MediaPipe, and saves them into a labeled CSV dataset.

### Stage 2: Model Training (`train_model.py`)
Loads the CSV dataset, applies `StandardScaler` for normalization, and trains a lightweight classifier (Random Forest by default) to predict the exercise type. Saves the trained model, scaler, and label encoder as `.pkl` artifacts.

### Stage 3: Real-Time Inference (`main_tracker.py`)
The production script. It reads a video feed, predicts the exercise in real-time, applies a `PredictionSmoother` (sliding window mode) to prevent flickering, and routes the data to the specific exercise class to count reps and evaluate the form.

## 🚀 How to Run

**1. Install Dependencies:**
```bash
pip install opencv-python mediapipe scikit-learn pandas numpy
```

**2. Collect Data:**
```bash
python data_collector.py --label squat --source your_video.mp4 --output data/landmarks.csv
```

**3. Train the Classifier:**
```bash
python train_model.py --data data/landmarks.csv --output models/
```

**4. Run the Tracker:**
```bash
python main_tracker.py --source your_video.mp4 --models models/
```

## 🧠 Future Improvements
- Upgrade the ML classifier to an LSTM or Transformer model to classify exercises based on sequential time-series data rather than single-frame posture.
- Deploy the backend as a FastAPI microservice.
```

