import os
import librosa
import numpy as np

DATASET_PATH = r"C:\Users\devid\safenet\datasets\screaming"
LABEL_MAP = {
    "NotScreaming": 0,
    "Screaming": 1
}

N_MFCC = 13

def extract_features(file_path, n_mfcc=N_MFCC):
    print(f"Loading file: {file_path}")
    y, sr = librosa.load(file_path, sr=None)
    print(f"Extracting MFCC from: {file_path}")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    print(f"Extracted features shape: {mfcc_mean.shape}")
    return mfcc_mean

def main():
    features = []
    labels = []

    for label_name, label_num in LABEL_MAP.items():
        folder_path = os.path.join(DATASET_PATH, label_name)
        if not os.path.isdir(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        print(f"Processing folder: {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav") or filename.endswith(".mp3"):
                file_path = os.path.join(folder_path, filename)
                print(f"Processing file: {file_path}")
                feat = extract_features(file_path)
                features.append(feat)
                labels.append(label_num)

    X = np.array(features)
    y = np.array(labels)

    print(f"Extracted {len(X)} samples with feature shape {X[0].shape}")
    counts = np.bincount(y)
    print(f"Labels Distribution before balancing: {counts}")

    # If imbalance detected, downsample majority class
    if counts[0] > counts[1]:
        not_screaming_indices = np.where(y == 0)[0]
        screaming_indices = np.where(y == 1)[0]

        np.random.seed(42)
        downsampled_not_screaming_indices = np.random.choice(
            not_screaming_indices, size=len(screaming_indices), replace=False
        )

        balanced_indices = np.concatenate([downsampled_not_screaming_indices, screaming_indices])
        np.random.shuffle(balanced_indices)

        X = X[balanced_indices]
        y = y[balanced_indices]
        print(f"Labels Distribution after balancing: {np.bincount(y)}")
    else:
        print("Dataset already balanced or screaming class is majority.")

    np.save("features.npy", X)
    np.save("labels.npy", y)
    print("Saved features.npy and labels.npy")

if __name__ == "__main__":
    main()
