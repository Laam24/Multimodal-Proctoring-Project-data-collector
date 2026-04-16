import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

# --- CONFIGURATION ---
DATA_FILE = "FINAL_TRAINING_DATASET.csv"
SEQUENCE_LENGTH = 30  # 3 seconds at 20 FPS
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.0001

# Map text labels to numbers for PyTorch
LABEL_MAP = {
    "NORMAL": 0,
    "CHEAT_PHYSICAL": 1,
    "CHEAT_DIGITAL": 2
}

# --- 1. DATA PREPARATION ---
class ProctorDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def create_sequences(df, feature_cols, seq_length):
    """Converts a flat dataframe into overlapping 3-second time windows."""
    sequences = []
    labels = []
    
    # We group by Session_ID so we don't accidentally create a sequence 
    # that starts at the end of one exam and finishes at the beginning of another!
    for session_id, group in df.groupby('Session_ID'):
        features = group[feature_cols].values
        target_labels = group['Target_Label'].map(LABEL_MAP).values
        
        for i in range(len(features) - seq_length):
            seq = features[i:(i + seq_length)]
            # We label the entire sequence based on what is happening at the very end of it
            label = target_labels[i + seq_length - 1] 
            
            sequences.append(seq)
            labels.append(label)
            
    return np.array(sequences), np.array(labels)

# --- 2. THE NEURAL NETWORK ---
class ProctorGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ProctorGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The GRU memory layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # The final decision layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Push data through GRU
        out, _ = self.gru(x, h0)
        
        # We only care about the GRU's output at the very last time step of the sequence
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# --- 3. MAIN TRAINING LOOP ---
def main():
    print("Loading Dataset...")
    df = pd.read_csv(DATA_FILE)
    
    # Identify our features (Exclude Session_ID, Timestamp, and Labels)
    exclude_cols = ['Session_ID', 'Timestamp', 'Target_Label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Features detected ({len(feature_cols)}): {feature_cols}")
    
    # Normalize the data (Neural Networks learn much faster when all numbers are between -1 and 1)
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # --- NEW LINE: SAVE THE SCALER TO DISK ---
    joblib.dump(scaler, 'proctor_scaler.pkl')
    print("✅ Scaler saved to 'proctor_scaler.pkl'")
    
    print("Slicing data into 3-second sequences... (This may take a minute)")
    X, y = create_sequences(df, feature_cols, SEQUENCE_LENGTH)
    
    # Split into Training and Testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_dataset = ProctorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- ASYMMETRIC CLASS WEIGHTING (DAMPENED) ---
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    # Using square root to prevent the weights from becoming mathematically unstable
    weights = [np.sqrt(total_samples / class_counts[i]) for i in range(len(LABEL_MAP))]
    weights = torch.tensor(weights, dtype=torch.float32)
    print(f"Smoothed Class Weights: {weights.numpy()}")
    
    # Initialize Model, Loss function, and Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    model = ProctorGRU(input_size=len(feature_cols), hidden_size=64, num_layers=2, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the Model
    print("\n--- STARTING TRAINING ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        
    # Save the trained brain!
    torch.save(model.state_dict(), 'proctor_gru_v1.pth')
    print("\n✅ Model successfully saved as 'proctor_gru_v1.pth'")

if __name__ == "__main__":
    main()