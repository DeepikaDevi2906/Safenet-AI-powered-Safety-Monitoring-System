import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

class AudioClassifier(nn.Module):
    def __init__(self, input_dim):
        super(AudioClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 2 classes

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    # Load features and labels
    X = np.load("features.npy")
    y = np.load("labels.npy")

    # Train-test split (same params as train.py)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Convert to torch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Load model
    input_dim = X.shape[1]
    model = AudioClassifier(input_dim)
    model.load_state_dict(torch.load("audio_classifier.pth"))
    model.eval()

    # Evaluate
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_test_tensor).sum().item()
        total = y_test_tensor.size(0)

    print(f"Test Accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    main()
