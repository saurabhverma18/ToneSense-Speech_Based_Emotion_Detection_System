# ToneSense: Speech Based Emotion Detection System

**ToneSense** is a speech-based emotion detection system that analyzes audio input to identify emotions from speech using machine learning techniques. The system processes audio features such as Mel Frequency Cepstral Coefficients (MFCCs) and Mel Spectrograms, and uses a Convolutional Neural Network (CNN) for emotion classification. It is deployed as a **Streamlit** application for an interactive web interface.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)

## Features
- **Emotion Detection**: Predicts emotions such as happy, sad, angry, neutral, etc., from audio data.
- **Data Augmentation**: Enhances the dataset through techniques like noise addition, pitch shifting, and time-stretching.
- **Feature Extraction**: Utilizes Mel Frequency Cepstral Coefficients (MFCCs) and Mel Spectrograms for extracting meaningful features from audio.
- **CNN Model**: A Convolutional Neural Network (CNN) is trained for emotion classification.
- **Streamlit Interface**: A simple, interactive web interface built with Streamlit for real-time emotion prediction.

## Technologies Used
- **Python**: Core programming language for the system.
- **Librosa**: For audio feature extraction (MFCCs and Mel Spectrograms).
- **TensorFlow/Keras**: For training the CNN model.
- **Streamlit**: For building the interactive web interface.
- **NumPy**: For numerical operations.
- **Matplotlib**: For data visualization.
- **scikit-learn**: For machine learning utilities and metrics.

## Installation
To set up **ToneSense** on your local machine, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/saurabhverma18/ToneSense-Speech_Based_Emotion_Detection_System.git
    cd ToneSense-Speech_Based_Emotion_Detection_System
    ```

2. **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the dataset** (if necessary) and make sure it’s placed in the appropriate folder.

## Usage
1. **Preprocess Data**: Run the script to preprocess the audio files and extract features like MFCCs or Mel Spectrograms.
    ```bash
    python preprocess_data.py
    ```

2. **Train the Model**: Train the emotion detection model using the prepared dataset.
    ```bash
    python train_model.py
    ```

3. **Run the Streamlit App**: Start the Streamlit app to deploy the model and interact with it through a web interface.
    ```bash
    streamlit run app.py
    ```

    After running the above command, open your browser and go to `http://localhost:8501` to use the app.

4. **Predict Emotion**: Upload an audio file through the Streamlit interface, and the system will predict the emotion based on the speech.

## Folder Structure
```bash
ToneSense-Speech_Based_Emotion_Detection_System/
│
├── app.py                # Streamlit application for interaction
├── preprocess_data.py    # Data preprocessing and feature extraction script
├── train_model.py        # Model training script
├── requirements.txt      # Python dependencies
├── dataset/              # Folder containing the audio dataset
│   └── <dataset_files>   # Audio files (e.g., .wav files)
├── model/                # Folder to save the trained model
├── utils/                # Helper functions for data and model handling
└── README.md             # This README file
