import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fpl_app import config

# Define Model
class FPLModel(nn.Module):
    def __init__(self, input_dim):
        super(FPLModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.net(x)

class FPLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_position_model(pos_name):
    print(f"\n--- Training {pos_name} Model ---")
    data_path = config.DATA_DIR / f"{pos_name.lower()}_df.csv"
    if not data_path.exists():
        print(f"Data not found for {pos_name}")
        return
        
    df = pd.read_csv(data_path)
    
    # Load feature list to ensure order
    with open(config.MODELS_DIR / f"{pos_name.lower()}_features.json", 'r') as f:
        feature_cols = json.load(f)
        
    # Split by Time
    rounds = sorted(df['round'].unique())
    split_idx = int(len(rounds) * 0.7)
    train_rounds = rounds[:split_idx]
    val_rounds = rounds[split_idx:]
    
    print(f"Training rounds: {train_rounds}")
    print(f"Validation rounds: {val_rounds}")
    
    train_df = df[df['round'].isin(train_rounds)]
    val_df = df[df['round'].isin(val_rounds)]
    
    if len(train_df) == 0 or len(val_df) == 0:
        print("Insufficient data for split.")
        return

    X_train = train_df[feature_cols].values
    y_train = train_df['target_points'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['target_points'].values
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save Scaler
    joblib.dump(scaler, config.MODELS_DIR / f"{pos_name.lower()}_scaler.joblib")
    
    # Datasets
    train_dataset = FPLDataset(X_train_scaled, y_train)
    val_dataset = FPLDataset(X_val_scaled, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Init Model
    model = FPLModel(len(feature_cols))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                outputs = model(X_b)
                loss = criterion(outputs, y_b)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODELS_DIR / f"{pos_name.lower()}_model.pt")
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
            
    print(f"Best Val MSE: {best_val_loss:.4f}")
    
    # Evaluation Metrics
    model.load_state_dict(torch.load(config.MODELS_DIR / f"{pos_name.lower()}_model.pt"))
    model.eval()
    
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_val_scaled)).numpy().flatten()
        
    mae = np.mean(np.abs(y_pred - y_val))
    rmse = np.sqrt(np.mean((y_pred - y_val)**2))
    
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")

def run_training():
    positions = ['GK', 'DEF', 'MID', 'FWD']
    for pos in positions:
        train_position_model(pos)

if __name__ == "__main__":
    run_training()
