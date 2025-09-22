import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Paths
VIOLENCE_DIR = r"datasets/violence/Violence"
NON_VIOLENCE_DIR = r"datasets/violence/NonViolence"

# Parameters
IMG_SIZE = (112, 112)        # frame resize size
FRAMES_PER_CLIP = 16         # number of frames per video sample
TEST_SIZE = 0.2              # train/test split ratio

def extract_features(video_path, num_frames=FRAMES_PER_CLIP, size=IMG_SIZE):
    """Extract fixed number of frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frames.append(frame)
        if len(frames) == num_frames:
            break

    cap.release()

    # Pad with black frames if not enough
    while len(frames) < num_frames:
        frames.append(np.zeros((size[1], size[0], 3), dtype=np.uint8))

    return np.array(frames, dtype=np.float32) / 255.0

def load_dataset():
    """Load violence and non-violence video dataset into numpy arrays."""
    X, y = [], []

    print("[INFO] Processing VIOLENCE videos...")
    for file in tqdm(os.listdir(VIOLENCE_DIR)):
        if file.endswith(".mp4"):
            frames = extract_features(os.path.join(VIOLENCE_DIR, file))
            X.append(frames)
            y.append(1)  # Violence label

    print("[INFO] Processing NON_VIOLENCE videos...")
    for file in tqdm(os.listdir(NON_VIOLENCE_DIR)):
        if file.endswith(".mp4"):
            frames = extract_features(os.path.join(NON_VIOLENCE_DIR, file))
            X.append(frames)
            y.append(0)  # Non-violence label

    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = load_dataset()
    print(f"[INFO] Full dataset shape: {X.shape}, Labels: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )

    print(f"[INFO] Train set: {X_train.shape}, Test set: {X_test.shape}")

    # 🔹 Save datasets
    np.save("X.npy", X)
    np.save("y.npy", y)
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)
    print("[INFO] Saved dataset and train/test split as .npy files")
                                         