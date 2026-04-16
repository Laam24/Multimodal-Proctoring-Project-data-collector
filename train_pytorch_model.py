import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import joblib  # <-- NEW IMPORT

# --- CONFIGURATION ---
INPUT_FILE = "FINAL_TRAINING_DATASET.csv"
MODEL_SAVE_PATH = "best_proctoring_model.pth"
SCALER_SAVE_PATH = "scaler.pkl"            # <-- NEW
ENCODER_SAVE_PATH = "label_encoder.pkl"    # <-- NEW
SEQUENCE_LENGTH = 60
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

FEATURES =['Face_Detected', 'Pitch', 'Yaw', 'Roll', 'Gaze_X', 'Gaze_Y', 'Unauthorized_Window']

print("=== PYTORCH MULTIMODAL LSTM TRAINER ===")

print(f"Loading data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
df[FEATURES] = df[FEATURES].fillna(0.0)

label_encoder = LabelEncoder()
df['Target_Label_Encoded'] = label_encoder.fit_transform(df['Target_Label'])
NUM_CLASSES = len(label_encoder.classes_)
print(f"Classes found: {label_encoder.classes_}")

# --- NEW: Save the Label Encoder ---
joblib.dump(label_encoder, ENCODER_SAVE_PATH)
print(f"Saved Label Encoder to {ENCODER_SAVE_PATH}")

scaler = StandardScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])

# --- NEW: Save the Scaler ---
joblib.dump(scaler, SCALER_SAVE_PATH)
print(f"Saved Scaler to {SCALER_SAVE_PATH}")

print(f"Creating {SEQUENCE_LENGTH}-frame (3-second) sequences...")
X, y = [],[]

for session_id, session_data in df.groupby('Session_ID'):
    features_array = session_data[FEATURES].values
    labels_array = session_data['Target_Label_Encoded'].values
    for i in range(len(features_array) - SEQUENCE_LENGTH):
        window_x = features_array[i : i + SEQUENCE_LENGTH]
        window_y = labels_array[i + SEQUENCE_LENGTH - 1] 
        X.append(window_x)
        y.append(window_y)

X = np.array(X); y = np.array(y)
print(f"Total Sequences Created: {X.shape[0]}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

class ProctoringDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(ProctoringDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ProctoringDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

class ProctoringLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ProctoringLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.dropout(self.relu(out))
        return self.fc2(out)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

model = ProctoringLSTM(input_size=len(FEATURES), hidden_size=64, num_layers=2, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nStarting Training...")
train_losses, val_losses = [],[]
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train(); running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward(); optimizer.step()
        running_loss += loss.item()
        
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    model.eval(); val_loss = 0.0; correct = 0; total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    val_losses.append(avg_val_loss)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"\n✅ Training Complete! Best model saved to: {MODEL_SAVE_PATH}")