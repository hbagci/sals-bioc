# ============================================================================
# STAGE 0 — Imports & Global Setup
# ============================================================================
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cpu")

def set_device(device_choice):
    global device
    if device_choice.lower() == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

# ============================================================================
# STAGE 1 — LSTM Model Architecture
# ============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim):
        super(LSTMModel, self).__init__()
        self.lstm1   = nn.LSTM(input_dim, 128, batch_first=True)
        self.lstm2   = nn.LSTM(128, 64, batch_first=True)
        self.fc1     = nn.Linear(64, 32)
        self.fc2     = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.4)
        self.relu    = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out    = self.dropout(out)
        out, _ = self.lstm2(out)
        out    = self.dropout(out)
        out    = self.relu(self.fc1(out[:, -1, :]))
        out    = self.dropout(out)
        return self.fc2(out).view(-1)

# ============================================================================
# STAGE 2 — Data Preprocessing
# ============================================================================
def preprocess_dataframe(df, target_column, contamination, scaler=None, apply_outlier_removal=True):
    df = df.drop(columns=['Unnamed: 0'], errors='ignore').copy()
    df = df.select_dtypes(include=[np.number])
    features = [c for c in df.columns if c != target_column]
    df = df[features + [target_column]].dropna()

    if apply_outlier_removal:
        iso = IsolationForest(contamination=contamination, random_state=42)
        mask = iso.fit_predict(df) != -1
        df = df[mask].reset_index(drop=True)

    data_tensor = torch.tensor(df.values, dtype=torch.float32)
    if scaler is None:
        mean = data_tensor.mean(0, keepdim=True)
        std  = data_tensor.std(0, keepdim=True)
        std  = torch.clamp(std, min=1e-8)
        scaler = (mean, std)
    else:
        mean, std = scaler

    normalised = ((data_tensor - mean) / std).cpu().numpy()
    return normalised, df.columns.tolist(), scaler, len(features)

def inverse_transform(scaler, data_scaled):
    mean, std = scaler
    mean = mean.cpu().numpy() if isinstance(mean, torch.Tensor) else mean
    std  = std.cpu().numpy()  if isinstance(std,  torch.Tensor) else std
    data_scaled = data_scaled.cpu().numpy() if isinstance(data_scaled, torch.Tensor) else data_scaled
    return data_scaled * std + mean

# ============================================================================
# STAGE 3 — Model Training
# ============================================================================
def train_model(model, X_train, y_train, epochs, batch_size, learning_rate, early_stopping_patience):
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)
    criterion = nn.MSELoss()

    input_dim = X_train.shape[1]
    X_tensor  = torch.FloatTensor(X_train.reshape(-1, 1, input_dim)).to(device)
    y_tensor  = torch.FloatTensor(y_train).to(device)
    loader    = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    model.train()
    best_loss        = float('inf')
    patience_counter = 0

    for _ in range(epochs):
        epoch_loss = 0.0
        for bX, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bX).view(-1), by.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        if epoch_loss < best_loss:
            best_loss        = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break
    return model

def predict(model, X):
    model.eval()
    input_dim = X.shape[1]
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X.reshape(-1, 1, input_dim)).to(device)
        return model(X_tensor).cpu().numpy().ravel()

# ============================================================================
# STAGE 4 — Model Persistence
# ============================================================================
def save_model(model, scaler, columns, input_dim, path):
    mean, std = scaler
    torch.save({
        'state_dict': model.state_dict(),
        'input_dim':  input_dim,
        'mean':       mean,
        'std':        std,
        'columns':    columns,
    }, path)

def load_model(path):
    ckpt    = torch.load(path, map_location='cpu', weights_only=False)
    model   = LSTMModel(ckpt['input_dim']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    scaler  = (ckpt['mean'], ckpt['std'])
    columns = ckpt['columns']
    return model, scaler, columns, ckpt['input_dim']
