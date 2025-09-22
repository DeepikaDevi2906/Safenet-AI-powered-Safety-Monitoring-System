import torch
import torch.nn as nn
import cv2
import numpy as np
import os

# -----------------------------
# Parameters
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2
IMG_SIZE = (112, 112)
FRAMES_PER_CLIP = 16
BASE_DIR = os.path.dirname(__file__)

# -----------------------------
# Model (same as training)
# -----------------------------
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*4*28*28, 128)
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
# Load trained model
# -----------------------------
model = Simple3DCNN().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "violence_model.pth"), map_location=DEVICE))
model.eval()

# -----------------------------
# Helper: Extract 16 frames clip
# -----------------------------
def process_clip(frames):
    """
    Input: list of BGR frames (length <= FRAMES_PER_CLIP)
    Output: tensor (1, C, T, H, W)
    """
    while len(frames) < FRAMES_PER_CLIP:
        frames.append(np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8))
    
    clip = [cv2.resize(f, IMG_SIZE) for f in frames]
    clip = np.array(clip, dtype=np.float32) / 255.0
    clip = torch.tensor(clip).permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)
    return clip.to(DEVICE)

# -----------------------------
# Real-time Webcam Prediction
# -----------------------------
cap = cv2.VideoCapture(0)
clip_frames = []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        clip_frames.append(frame)
        if len(clip_frames) == FRAMES_PER_CLIP:
            input_clip = process_clip(clip_frames)
            with torch.no_grad():
                outputs = model(input_clip)
                _, pred = outputs.max(1)
                label = "Violence" if pred.item() == 1 else "Non-Violence"
            
            # Display prediction
            cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Violence Detection", frame)
            
            # Slide window: remove first frame
            clip_frames.pop(0)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
