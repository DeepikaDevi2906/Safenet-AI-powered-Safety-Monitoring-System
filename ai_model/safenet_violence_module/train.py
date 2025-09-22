import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

# -----------------------------
# Parameters
# -----------------------------
BATCH_SIZE = 4          # small batch to save memory
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2         # Violence / NonViolence

BASE_DIR = os.path.dirname(__file__)
SAFE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))  # main SafeNet folder

# -----------------------------
# Memory-efficient Dataset
# -----------------------------
class ViolenceDataset(Dataset):
    def __init__(self, X_path, y_path):
        if not os.path.exists(X_path) or not os.path.exists(y_path):
            raise FileNotFoundError(f"Dataset files not found in {SAFE_DIR}")
        self.X = np.load(X_path, mmap_mode='r')  # memory-mapped
        self.y = np.load(y_path)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]  # shape: (T, H, W, C)
        x = torch.tensor(x, dtype=torch.float32).permute(3, 0, 1, 2)  # (C, T, H, W)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

# -----------------------------
# Simple 3D CNN Model
# -----------------------------
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*4*28*28, 128)  # adjust according to input size
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------
# Load Data
# -----------------------------
train_dataset = ViolenceDataset(
    os.path.join(SAFE_DIR, "X_train.npy"),
    os.path.join(SAFE_DIR, "y_train.npy")
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = ViolenceDataset(
    os.path.join(SAFE_DIR, "X_test.npy"),
    os.path.join(SAFE_DIR, "y_test.npy")
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Initialize Model, Loss, Optimizer
# -----------------------------
model = Simple3DCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, "
          f"Accuracy={100*correct/total:.2f}%")

# -----------------------------
# Save Model
# -----------------------------
torch.save(model.state_dict(), os.path.join(BASE_DIR, "violence_model.pth"))
print("[INFO] Model saved as violence_model.pth")
