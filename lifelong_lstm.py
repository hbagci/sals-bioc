# ============================================================================
# STAGE 0 — Imports & Global Setup
# ============================================================================
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Lasso
from numpy.linalg import inv
from scipy.linalg import sqrtm
from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cpu")

DEFAULTS = {
    'mu':                       1e-06,
    'lam':                      1e-06,
    'beta':                     1e-05,
    'k':                        5,
    'batch_size':               32,
    'learning_rate':            1e-3,
    'epochs':                   50,
    'early_stopping_patience':  10,
    'contamination':            0.05,
    'train_ratio':              0.8,
    'adaptation_batch_size':    30,
    'bayes_init_points':        1,
    'bayes_n_iter':             1,
    'mu_min':  1e-7,  'mu_max':   1e-3,
    'lam_min': 1e-7,  'lam_max':  1e-3,
    'beta_min': 1e-9, 'beta_max': 1e-3,
}


def set_device(choice):
    global device
    device = torch.device(
        "cuda" if choice.lower() == "gpu" and torch.cuda.is_available() else "cpu"
    )


# ============================================================================
# STAGE 1 — LSTM Model Architecture
# ============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm1      = nn.LSTM(input_dim, 128, batch_first=True)
        self.lstm2      = nn.LSTM(128, 64, batch_first=True)
        self.fc1        = nn.Linear(64, 32)
        self.fc2        = nn.Linear(32, 1)
        self.theta_proj = nn.Linear(32, input_dim)
        self.dropout    = nn.Dropout(0.4)
        self.relu       = nn.ReLU()

    def features(self, x):
        out, _ = self.lstm1(x)
        out    = self.dropout(out)
        out, _ = self.lstm2(out)
        out    = self.dropout(out)
        out    = self.relu(self.fc1(out[:, -1, :]))
        return self.dropout(out)

    def forward(self, x, theta):
        return self.fc2(self.features(x)).view(-1) + (x[:, -1, :] * theta).sum(dim=1)

    def theta(self, x):
        return self.theta_proj(self.features(x)).mean(dim=0)


# ============================================================================
# STAGE 2 — Data Preprocessing
# ============================================================================
def preprocess_data(df, target_column, contamination, scaler=None):
    features = [c for c in df.columns if c != target_column]
    df = df[features + [target_column]].copy().dropna()

    iso  = IsolationForest(contamination=contamination, random_state=42)
    mask = iso.fit_predict(df) != -1
    df   = df[mask].reset_index(drop=True)

    data_tensor = torch.tensor(df.values, dtype=torch.float32)
    if scaler is None:
        mean = data_tensor.mean(0, keepdim=True)
        std  = data_tensor.std(0, keepdim=True)
        std  = torch.where(std == 0, torch.ones_like(std), std)
    else:
        mean, std = scaler

    normalised = ((data_tensor - mean) / std).cpu().numpy()
    return normalised, df.columns.tolist(), (mean, std), len(features)


def inverse_transform(scaler, data_scaled):
    mean, std = scaler
    mean = mean.cpu().numpy() if isinstance(mean, torch.Tensor) else np.array(mean)
    std  = std.cpu().numpy()  if isinstance(std,  torch.Tensor) else np.array(std)
    if isinstance(data_scaled, torch.Tensor):
        data_scaled = data_scaled.cpu().numpy()
    return data_scaled * std + mean


# ============================================================================
# STAGE 3 — Lifelong Learning Core (Streaming / ELLA)
# ============================================================================
class Streaming:
    def __init__(self, d, k, mu, lam, beta, epochs, early_stopping_patience, batch_size, learning_rate):
        self.d    = d
        self.k    = k
        self.L    = np.random.randn(d, k)
        self.A    = np.zeros((d * k, d * k))
        self.b    = np.zeros((d * k, 1))
        self.S    = np.zeros((k, 0))
        self.T    = 0
        self.mu   = mu
        self.lam  = lam
        self.beta = beta
        self.epochs                  = epochs
        self.early_stopping_patience = early_stopping_patience
        self.batch_size              = batch_size
        self.single_task_model = LSTMModel(d).to(device)
        self.optimizer = torch.optim.RMSprop(
            self.single_task_model.parameters(), lr=learning_rate, alpha=0.9
        )
        self.criterion = nn.MSELoss()

    def fit(self, X, y):
        self.T += 1
        y = y.ravel()

        X_tensor = torch.FloatTensor(X.reshape(-1, 1, self.d)).to(device)
        y_tensor = torch.FloatTensor(y).to(device)
        loader   = DataLoader(
            TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=True
        )

        self.single_task_model.train()
        best_loss        = float('inf')
        patience_counter = 0

        for _ in range(self.epochs):
            epoch_loss = 0.0
            for bX, by in loader:
                self.optimizer.zero_grad()
                f       = self.single_task_model.features(bX)
                base    = self.single_task_model.fc2(f).view(-1)
                theta_b = self.single_task_model.theta_proj(f.detach()).mean(dim=0)
                lin     = (bX[:, -1, :] * theta_b).sum(dim=1)
                loss    = (self.criterion(base, by.view(-1))
                           + self.criterion(base.detach() + lin, by.view(-1)))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)
            if epoch_loss < best_loss:
                best_loss        = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break

        self.single_task_model.eval()
        with torch.no_grad():
            theta_t = self.single_task_model.theta(X_tensor).cpu().numpy()

        D_t      = self.get_hessian(X)
        D_t_sqrt = np.real(sqrtm(D_t))

        s_t = Lasso(alpha=self.mu, max_iter=5000, fit_intercept=False).fit(
            D_t_sqrt @ self.L, D_t_sqrt @ theta_t
        ).coef_.reshape(self.k, 1)

        self.S   = np.hstack((self.S, s_t))
        self.A  += np.kron(s_t @ s_t.T, D_t) + self.mu * np.eye(self.d * self.k)
        self.b  += np.kron(s_t.T, theta_t @ D_t).T

        L_vec  = np.real(inv(self.A / self.T + self.lam * np.eye(self.d * self.k))) @ self.b / self.T
        self.L = L_vec.reshape((self.k, self.d)).T
        self.revive_dead_components()
        self.L /= np.maximum(np.linalg.norm(self.L, axis=0, keepdims=True), 1e-12)

    def predict(self, X):
        self.single_task_model.eval()
        with torch.no_grad():
            theta    = torch.FloatTensor((self.L @ self.S[:, -1:]).ravel()).to(device)
            X_tensor = torch.FloatTensor(X.reshape(-1, 1, self.d)).to(device)
            return self.single_task_model(X_tensor, theta).cpu().numpy().ravel()

    def get_hessian(self, X):
        XTX = X.T @ X
        return (XTX + self.beta * np.eye(XTX.shape[0])) / (2.0 * X.shape[0])

    def revive_dead_components(self):
        for i, val in enumerate(np.sum(self.L, axis=0)):
            if abs(val) < 1e-8:
                self.L[:, i] = np.random.randn(self.d)


# ============================================================================
# STAGE 4 — Metrics
# ============================================================================
def _safe(v):
    v = float(v)
    return None if not np.isfinite(v) else v


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        'R2':   _safe(r2_score(y_true, y_pred)),
        'RMSE': _safe(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE':  _safe(mean_absolute_error(y_true, y_pred)),
        'MSE':  _safe(mean_squared_error(y_true, y_pred)),
    }


def calculate_weighted_metrics(metrics_list):
    n       = len(metrics_list)
    weights = np.linspace(1, n, n) / np.linspace(1, n, n).sum()
    return {
        key: _safe(np.sum([
            (m[key] if m[key] is not None else 0.0) * w
            for m, w in zip(metrics_list, weights)
        ]))
        for key in metrics_list[0]
    }


def calculate_last_batch_metrics(batch_metrics):
    last = batch_metrics[-1]
    return calculate_metrics(last['y_true'], last['y_pred'])


def calculate_first_batch_metrics(batch_metrics):
    first = batch_metrics[0]
    return calculate_metrics(first['y_true'], first['y_pred'])


# ============================================================================
# STAGE 5 — Bayesian Optimisation
# ============================================================================
def bayes_optimize(X_train, y_train, X_test, y_test, input_dim, params, init_points, n_iter):
    epochs   = int(params.get('epochs',                  DEFAULTS['epochs']))
    patience = int(params.get('early_stopping_patience', DEFAULTS['early_stopping_patience']))
    bs       = int(params.get('batch_size',              DEFAULTS['batch_size']))
    lr       = float(params.get('learning_rate',         DEFAULTS['learning_rate']))
    k        = int(params.get('k',                       DEFAULTS['k']))
    adapt_bs = int(params.get('adaptation_batch_size', DEFAULTS['adaptation_batch_size']))

    mu_bounds   = (float(params.get('mu_min',   DEFAULTS['mu_min'])),
                   float(params.get('mu_max',   DEFAULTS['mu_max'])))
    lam_bounds  = (float(params.get('lam_min',  DEFAULTS['lam_min'])),
                   float(params.get('lam_max',  DEFAULTS['lam_max'])))
    beta_bounds = (float(params.get('beta_min', DEFAULTS['beta_min'])),
                   float(params.get('beta_max', DEFAULTS['beta_max'])))

    def objective(mu, lamda, beta):
        s = Streaming(input_dim, k, mu, lamda, beta, epochs, patience, bs, lr)
        s.fit(X_train, y_train)
        preds, trues = [], []
        for i in range(0, len(X_test), adapt_bs):
            Xb, yb = X_test[i:i + adapt_bs], y_test[i:i + adapt_bs]
            preds.append(s.predict(Xb))
            trues.append(yb)
            s.fit(Xb, yb)
        return -float(np.sqrt(mean_squared_error(
            np.concatenate(trues), np.concatenate(preds))))

    opt = BayesianOptimization(
        f=objective,
        pbounds={'mu': mu_bounds, 'lamda': lam_bounds, 'beta': beta_bounds},
        random_state=42,
        verbose=0,
    )
    opt.maximize(init_points=init_points, n_iter=n_iter)

    best = opt.max['params']
    return {
        'mu':    float(best['mu']),
        'lam':   float(best['lamda']),
        'beta':  float(best['beta']),
        'score': float(opt.max['target']),
    }


# ============================================================================
# STAGE 6 — Train phase (source CSV only)
# ============================================================================
def run_train(data1, input_dim, scaler, params):
    mu       = float(params.get('mu',                      DEFAULTS['mu']))
    lam      = float(params.get('lam',                     DEFAULTS['lam']))
    beta     = float(params.get('beta',                    DEFAULTS['beta']))
    bs       = int(params.get('batch_size',                DEFAULTS['batch_size']))
    lr       = float(params.get('learning_rate',           DEFAULTS['learning_rate']))
    epochs   = int(params.get('epochs',                    DEFAULTS['epochs']))
    patience = int(params.get('early_stopping_patience',   DEFAULTS['early_stopping_patience']))
    ratio    = float(params.get('train_ratio',             DEFAULTS['train_ratio']))
    k        = int(params.get('k',                         DEFAULTS['k']))

    X1 = data1[:, :input_dim]
    y1 = data1[:, input_dim]

    split    = int(ratio * len(X1))
    X_train, y_train = X1[:split], y1[:split]
    X_test,  y_test  = X1[split:], y1[split:]

    streaming = Streaming(input_dim, k, mu, lam, beta, epochs, patience, bs, lr)
    streaming.fit(X_train, y_train)

    def rescale(X, y):
        arr = np.column_stack((X, y.ravel()))
        return inverse_transform(scaler, arr)[:, -1]

    train_true_r = rescale(X_train, y_train).tolist()
    train_pred_r = rescale(X_train, streaming.predict(X_train)).tolist()
    test_true_r  = rescale(X_test,  y_test).tolist()
    test_pred_r  = rescale(X_test,  streaming.predict(X_test)).tolist()

    return {
        'train':      {'y_true': train_true_r, 'y_pred': train_pred_r},
        'test':       {'y_true': test_true_r,  'y_pred': test_pred_r},
        'metrics':    {
            'train': calculate_metrics(train_true_r, train_pred_r),
            'test':  calculate_metrics(test_true_r,  test_pred_r),
        },
        '_streaming': streaming,
        '_scaler':    scaler,
        '_input_dim': input_dim,
    }


# ============================================================================
# STAGE 7 — Adapt phase (target CSV, prequential predict-then-fit)
# ============================================================================
def run_adapt(streaming, data2, input_dim, scaler, params):
    adapt_bs = int(params.get('adaptation_batch_size', DEFAULTS['adaptation_batch_size']))

    X2 = data2[:, :input_dim]
    y2 = data2[:, input_dim]

    def rescale(X, y):
        arr = np.column_stack((X, y.ravel()))
        return inverse_transform(scaler, arr)[:, -1]

    n_batches     = max(1, (len(X2) + adapt_bs - 1) // adapt_bs)
    batch_metrics = []

    for i in range(n_batches):
        Xb   = X2[i * adapt_bs:(i + 1) * adapt_bs]
        yb   = y2[i * adapt_bs:(i + 1) * adapt_bs]
        pred = streaming.predict(Xb)
        tr   = rescale(Xb, yb)
        pr   = rescale(Xb, pred)
        m    = calculate_metrics(tr, pr)
        batch_metrics.append({
            'y_true': tr.tolist(),
            'y_pred': pr.tolist(),
            'rmse':   m['RMSE'],
            'r2':     m['R2'],
        })
        streaming.fit(Xb, yb)

    adap_true_all = [v for b in batch_metrics for v in b['y_true']]
    adap_pred_all = [v for b in batch_metrics for v in b['y_pred']]

    return {
        'adapt':         {'y_true': adap_true_all, 'y_pred': adap_pred_all},
        'metrics':       {
            'adapt':             calculate_metrics(adap_true_all, adap_pred_all),
            'first_batch_adapt': calculate_first_batch_metrics(batch_metrics),
            'last_batch_adapt':  calculate_last_batch_metrics(batch_metrics),
        },
        'batch_metrics': batch_metrics,
    }
