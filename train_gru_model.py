import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight # New Import
import matplotlib.pyplot as plt
import os
import joblib

# --- CONFIGURATION (Unchanged) ---
INPUT_FILE = "FINAL_TRAINING_DATASET.csv"
MODEL_SAVE_PATH = "best_gru_model_weighted.pth" # New model name
SCALER_SAVE_PATH = "scaler.pkl"
ENCODER_SAVE_PATH = "label_encoder.pkl"
SEQUENCE_LENGTH = 60
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
FEATURES =['Face_Detected', 'Pitch', 'Yaw', 'Roll', 'Gaze_X', 'Gaze_Y', 'Unauthorized_Window']

print("=== PYTORCH GRU TRAINER (WITH CLASS WEIGHTING) ===")

# --- 1. DATA PREPARATION ---
df = pd.read_csv(INPUT_FILE)
df[FEATURES] = df[FEATURES].fillna(0.0)
label_encoder = LabelEncoder()
df['Target_Label_Encoded'] = label_encoder.fit_transform(df['Target_Label'])
NUM_CLASSES = len(label_encoder.classes_)
print(f"Classes found: {label_encoder.classes_}")
joblib.dump(label_encoder, ENCODER_SAVE_PATH)
scaler = StandardScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])
joblib.dump(scaler, SCALER_SAVE_PATH)

# --- NEW: CALCULATE CLASS WEIGHTS ---
class_labels = np.unique(df['Target_Label_Encoded'])
class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=df['Target_Label_Encoded'])
weights = torch.tensor(class_weights, dtype=torch.float32)
print(f"Calculated Class Weights: {weights}")

# --- 2. SLIDING WINDOW (Unchanged) ---
X, y = [], []
for session_id, session_data in df.groupby('Session_ID'):
    features_array = session_data[FEATURES].values
    labels_array = session_data['Target_Label_Encoded'].values
    for i in range(len(features_array) - SEQUENCE_LENGTH):
        X.append(features_array[i : i + SEQUENCE_LENGTH])
        y.append(labels_array[i + SEQUENCE_LENGTH - 1])
X = np.array(X); y = np.array(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. PYTORCH DATASETS (Unchanged) ---
class ProctoringDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]
train_loader = DataLoader(ProctoringDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ProctoringDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

# --- 4. GRU ARCHITECTURE (Unchanged) ---
class ProctoringGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ProctoringGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc1(out[:, -1, :])
        out = self.dropout(self.relu(out))
        return self.fc2(out)

# --- 5. TRAINING LOOP ---
device = torch.device('cpu') # Force CPU for now for stability
weights = weights.to(device) # Move weights to the same device as model
print(f"Training on device: {device}")
model = ProctoringGRU(input_size=len(FEATURES), hidden_size=64, num_layers=2, num_classes=NUM_CLASSES).to(device)

# --- NEW: APPLY WEIGHTS TO THE LOSS FUNCTION ---
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nStarting GRU Model Training...")
best_val_accuracy = 0.0

for epoch in range(EPOCHS):
    model.train(); running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward(); optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    
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
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"\n✅ Training Complete! Best Weighted GRU model saved to: {MODEL_SAVE_PATH}")