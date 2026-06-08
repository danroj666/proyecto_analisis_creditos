"""
model.py
--------
Define la arquitectura de la red neuronal y el preprocesador de datos.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pickle


# ─────────────────────────────────────────────────────────────────────────────
# ARQUITECTURA
# ─────────────────────────────────────────────────────────────────────────────
class CreditNet(nn.Module):
    """
    Red neuronal feedforward para clasificación binaria.

    Capas:
        Input(9) → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
                 → Linear(32) → BatchNorm → ReLU → Dropout(0.2)
                 → Linear(16) → ReLU
                 → Linear(1)  → Sigmoid
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze()


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESADOR
# ─────────────────────────────────────────────────────────────────────────────
CAT_COLS = ['savings', 'employment', 'credit_history', 'purpose', 'housing']
FEATURE_ORDER = ['age', 'duration', 'amount', 'savings', 'employment',
                 'credit_history', 'purpose', 'housing', 'existing_credits']


class Preprocessor:
    """Codifica variables categóricas y escala numéricas."""

    def __init__(self):
        self.encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()

    def fit_transform(self, df):
        import pandas as pd
        df_enc = df.copy()
        for col in CAT_COLS:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df[col])
            self.encoders[col] = le

        X = df_enc[FEATURE_ORDER].values.astype(np.float32)
        y = df_enc['target'].values.astype(np.float32)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr, y_tr, test_size=0.15, random_state=42)

        X_tr  = self.scaler.fit_transform(X_tr)
        X_val = self.scaler.transform(X_val)
        X_te  = self.scaler.transform(X_te)

        return (
            self._loader(X_tr,  y_tr,  shuffle=True),
            self._loader(X_val, y_val),
            self._loader(X_te,  y_te),
            X_tr.shape[1]
        )

    def transform_single(self, sample: dict) -> torch.Tensor:
        """Transforma un dict con datos de un cliente a tensor."""
        row = sample.copy()
        for col in CAT_COLS:
            row[col] = self.encoders[col].transform([row[col]])[0]
        X = np.array([[row[f] for f in FEATURE_ORDER]], dtype=np.float32)
        X = self.scaler.transform(X)
        return torch.FloatTensor(X)

    def save(self, path: str = 'preprocessor.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Preprocesador guardado en {path}")

    @staticmethod
    def load(path: str = 'preprocessor.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _loader(X, y, shuffle=False):
        ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        return DataLoader(ds, batch_size=32, shuffle=shuffle)