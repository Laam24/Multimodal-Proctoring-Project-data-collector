import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- 1. LOAD RELATIVE DATA ---
print("Loading RELATIVE_TRAINING_DATASET.csv...")
df = pd.read_csv("RELATIVE_TRAINING_DATASET.csv")

# Exclude metadata (Ensure 'Frame' is excluded so the AI doesn't get poisoned)
exclude = ['Session_ID', 'Timestamp', 'Target_Label', 'Frame']
feature_cols = [c for c in df.columns if c in df.columns and c not in exclude]
X_raw = df[feature_cols].values
y_raw = df['Target_Label'].map({'NORMAL': 0, 'CHEAT_PHYSICAL': 1, 'CHEAT_DIGITAL': 2}).values

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
joblib.dump(scaler, "behavioral_scaler.pkl")

# --- 2. SEQUENCE GENERATION ---
def create_sequences(data, labels, seq_length=60):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(labels[i+seq_length-1])
    return np.array(X_seq), np.array(y_seq)

print("Creating 60-frame time-series blocks...")
X_s, y_s = create_sequences(X_scaled, y_raw)
X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.2, stratify=y_s, random_state=42)

train_data = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_data = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# --- 3. THE 1D-CNN + BiLSTM ARCHITECTURE ---
class BehavioralEngine(nn.Module):
    def __init__(self, input_size=len(feature_cols), hidden_size=64, num_layers=2):
        super(BehavioralEngine, self).__init__()
        
        # 1D-CNN (Feature Extractor)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # BiLSTM (Temporal Memory)
        # Note: bidirectional=True doubles the hidden output size
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.3)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64), # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        # PyTorch CNN expects shape: (batch, channels, seq_length)
        # Our data is (batch, seq_length, channels), so we permute it
        x = x.permute(0, 2, 1) 
        
        cnn_out = self.cnn(x)
        
        # Permute back for the LSTM: (batch, new_seq_length, channels)
        cnn_out = cnn_out.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(cnn_out)
        
        # Take the output from the final time step
        return self.fc(lstm_out[:, -1, :])

# --- 4. TRAINING WITH PENALTY WEIGHTS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BehavioralEngine().to(device)

# Penalty weights to stop "Lazy AI" guessing
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

# We use the lower learning rate that prevented NaN errors previously
optimizer = optim.Adam(model.parameters(), lr=0.0005)

print(f"\nStarting 1D-CNN + BiLSTM Training on {device}...")
for epoch in range(30):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Safety Valve: Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    # Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            preds = model(bx).argmax(1)
            correct += (preds == by).sum().item()
    
    acc = (correct / len(test_data)) * 100
    print(f"Epoch {epoch+1}/30 | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

torch.save(model.state_dict(), "behavioral_engine_v1.pth")
print("\n✅ Behavioral Engine Saved as 'behavioral_engine_v1.pth'")