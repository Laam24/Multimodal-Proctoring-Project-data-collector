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

# --- 1. LOAD CLEAN DATA ---
print("Loading cleaned dataset...")
df = pd.read_csv("FINAL_TRAINING_DATASET_CLEAN.csv")

# FIX 1: Add 'Frame' to the exclude list
exclude = ['Session_ID', 'Timestamp', 'Target_Label', 'Frame'] 
feature_cols = [c for c in df.columns if c in df.columns and c not in exclude]
X_raw = df[feature_cols].values
y_raw = df['Target_Label'].map({'NORMAL': 0, 'CHEAT_PHYSICAL': 1, 'CHEAT_DIGITAL': 2}).values

exclude = ['Session_ID', 'Timestamp', 'Target_Label']
feature_cols = [c for c in df.columns if c in df.columns and c not in exclude]
X_raw = df[feature_cols].values
y_raw = df['Target_Label'].map({'NORMAL': 0, 'CHEAT_PHYSICAL': 1, 'CHEAT_DIGITAL': 2}).values

# Scaling is mandatory for GRUs to prevent math issues
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
joblib.dump(scaler, "super_gru_scaler.pkl")

# --- 2. SEQUENCE GENERATION ---
def create_sequences(data, labels, seq_length=60):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(labels[i+seq_length-1])
    return np.array(X_seq), np.array(y_seq)

print("Creating sequences...")
X_s, y_s = create_sequences(X_scaled, y_raw)
X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.2, stratify=y_s, random_state=42)

train_data = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_data = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# --- 3. STABILIZED ARCHITECTURE ---
class SuperProctorGRU(nn.Module):
    def __init__(self, input_size=17, hidden_size=128, num_layers=2):
        super(SuperProctorGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# --- 4. TRAINING WITH GRADIENT CLIPPING ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SuperProctorGRU(input_size=len(feature_cols)).to(device)

# Penalty weights to stop the AI from just guessing "Normal"
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

# Lower learning rate for stability
optimizer = optim.Adam(model.parameters(), lr=0.0005)

print(f"Starting Training on {device}...")
for epoch in range(30):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # SAFETY VALVE: Cap the gradients to prevent NaN
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
    
    acc = (correct/len(test_data)) * 100
    print(f"Epoch {epoch+1}/20 | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

torch.save(model.state_dict(), "super_gru_v1.pth")
print("\n✅ Stabilized Super-GRU Saved as 'super_gru_v1.pth'")