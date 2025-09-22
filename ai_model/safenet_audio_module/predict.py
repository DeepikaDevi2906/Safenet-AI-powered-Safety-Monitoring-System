import sounddevice as sd
import numpy as np
import librosa
import torch
import requests
from train import AudioClassifier

def record_audio(duration=10, fs=22050):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Recording done.")
    return audio.flatten()

def sliding_window_predict(model, audio, sr=22050, window_sec=3, step_sec=1):
    window_len = int(window_sec * sr)
    step_len = int(step_sec * sr)
    preds = []

    for start in range(0, len(audio) - window_len + 1, step_len):
        chunk = audio[start:start+window_len]
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        x = torch.tensor(mfcc_mean, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(x)
            pred = torch.argmax(output, dim=1).item()
            preds.append(pred)

    final_pred = max(set(preds), key=preds.count)
    return final_pred, preds

def send_scream_alert(location="Unknown", source="audio_detector"):
    url = "http://localhost:5000/send-alert"  # Change to your backend URL
    data = {
        "type": "Screaming",
        "location": location,
        "source": source
    }
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("Alert sent successfully")
        else:
            print("Failed to send alert:", response.text)
    except Exception as e:
        print("Error sending alert:", e)

if __name__ == "__main__":
    model = AudioClassifier(input_dim=13)
    model.load_state_dict(torch.load("audio_classifier.pth"))
    model.eval()

    audio_data = record_audio(duration=10)
    final_pred, preds = sliding_window_predict(model, audio_data)

    labels = {0: "Not Screaming", 1: "Screaming"}
    print(f"Predictions per window: {[labels[p] for p in preds]}")
    print(f"Final Prediction (majority vote): {labels[final_pred]}")

    if final_pred == 1:
        send_scream_alert(location="Office Lobby")
