import pandas as pd
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import StandardScaler
import streamlit as st
import openai
import os
from dotenv import load_dotenv

load_dotenv()

# Function to guess the musical key from chroma features
def guess_key_from_chroma(chroma):
    key_strengths = np.sum(chroma, axis=1)
    key = np.argmax(key_strengths)
    return key  # 0-11, corresponds to C-B in chromatic scale

# Function to calculate danceability
def calculate_danceability(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, backtrack=True)
    beats = librosa.beat.beat_track(y=y, sr=sr)[1]
    danceability = np.sqrt(len(onsets)) / (len(beats) + 1e-6)
    return danceability

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, duration=30)  # Load only first 30 seconds
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    # Extract features
    features = {
        'song_duration_ms': librosa.get_duration(y=y, sr=sr) * 1000,
        'acousticness': np.mean(librosa.feature.spectral_flatness(y=y)),  # Approximation
        'danceability': calculate_danceability(y, sr),
        'energy': np.mean(librosa.feature.rms(y=y)),
        'instrumentalness': np.mean(librosa.effects.harmonic(y)),  # Approximation
        'key': guess_key_from_chroma(chroma),  # Using chroma features for key
        'liveness': np.mean(librosa.feature.spectral_flatness(y=y)),
        'loudness': np.mean(librosa.feature.rms(y=y)),
        'audio_mode': binarize_audio_mode(librosa.feature.zero_crossing_rate(y=y)),
        'speechiness': min(max(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)) / 100, 0), 1),  # Correct calculation for speechiness
        'tempo': tempo[0],  # Ensure tempo is a float
        'time_signature': 4 if np.mean(librosa.frames_to_time(beat_frames, sr=sr)) < 1.0 else 3,  # Estimation
        'audio_valence': np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr))  # Extract tonal centroid features
    }

    # Normalize audio_valence to range [0, 1]
    features['audio_valence'] = min(max(features['audio_valence'] / 1.0, 0), 1)

    return features

def binarize_audio_mode(zcr_values):
    # Convert mean of zero-crossing rate into binary mode
    mean_zcr = np.mean(zcr_values)
    # Binarize: if the mean is above 0.5, set to 1; otherwise, set to 0
    return 1 if mean_zcr > 0.5 else 0

# Load the trained model and scaler
model = joblib.load('best_random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

# Function to predict popularity
def predict_popularity(features):
    # Convert features to DataFrame
    df = pd.DataFrame([features])

    # Scale features
    scaled_features = scaler.transform(df)

    # Apply PCA
    pca_features = pca.transform(scaled_features)

    # Predict using the model
    popularity_score = model.predict(pca_features)[0]

    return popularity_score

# Function to interact with OpenAI ChatGPT API (using new interface)
def get_chatgpt_summary(predicted_score, features):
    # Combine popularity score and features into a single message for ChatGPT
    prompt = f"Given the music features: {features} and predicted popularity score: {predicted_score}, if I were you, I’d consider mixing things up a bit. Maybe add some unexpected twists—like incorporating unique sounds, experimenting with tempo changes, or throwing in a catchy, unconventional hook. Given the current song's characteristics and its predicted popularity score, what are some wild yet practical tweaks you could try to make this track stand out and potentially boost its appeal?"
    openai.api_key = os.getenv('OPENAI_API_KEY')
    # Query OpenAI's new API for a response
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use the appropriate model version
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=220  # Limit the response length
    )

    # Extract summary from response
    summary = response.choices[0].message['content'].strip()
    return summary

# Streamlit app
def main():
    st.title("Webs's Music Popularity Predictor")

    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3"])

    if uploaded_file is not None:
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success("File uploaded successfully!")

        # Display uploaded audio
        st.audio(file_path, format='audio/mp3', start_time=0)

        # Feature extraction process
        st.write("Extracting features...")
        features = extract_audio_features(file_path)

        # Display extracted features with progress
        st.write("Extracted Features:", features)

        # Simulate feature extraction with a progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
        st.write("Feature extraction complete!")

        # Predict popularity with a loading animation
        st.write("Predicting popularity...")
        predicted_score = predict_popularity(features)  # Calculate the predicted popularity score

        st.write(f"Predicted Popularity Score: **{predicted_score}**")

        # Get ChatGPT summary based on features and predicted score
        st.write("Let us help you cook...")
        chatgpt_summary = get_chatgpt_summary(predicted_score, features)
        st.write(f"Cook with these: {chatgpt_summary}")

if __name__ == "__main__":
    main()
