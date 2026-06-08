"""
train.py
--------
Entrena la red neuronal y guarda el modelo entrenado.

Uso:
    python train.py
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data.generate_data import generate_credit_data
from model import CreditNet, Preprocessor


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS    = 60
LR        = 1e-3
SAVE_PATH = 'credit_model.pt'


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    history   = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print(f"\nEntrenando por {epochs} épocas...\n{'─'*55}")

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validar ──
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred    = model(xb)
                val_loss += criterion(pred, yb).item()
                correct  += ((pred > 0.5).float() == yb).sum().item()
                total    += len(yb)

        scheduler.step()

        tl = train_loss / len(train_loader)
        vl = val_loss   / len(val_loader)
        va = correct    / total
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['val_acc'].append(va)

        if (epoch + 1) % 10 == 0:
            print(f"  Época {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {tl:.4f} | Val Loss: {vl:.4f} | Val Acc: {va:.3f}")

    return history


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss', color='#2563eb')
    ax1.plot(history['val_loss'],   label='Val Loss',   color='#dc2626')
    ax1.set_title('Pérdida durante entrenamiento')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss (BCE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['val_acc'], color='#16a34a')
    ax2.set_title('Accuracy en validación')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("✓ Curvas guardadas en training_curves.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("1. Generando datos...")
    df = generate_credit_data(n=1000)
    df.to_csv('data/credit_data.csv', index=False)
    print(f"   {len(df)} registros | {df['target'].mean():.1%} buenos pagadores")

    print("\n2. Preprocesando datos...")
    prep = Preprocessor()
    train_loader, val_loader, test_loader, input_dim = prep.fit_transform(df)
    prep.save('preprocessor.pkl')
    print(f"   input_dim = {input_dim}")

    print("\n3. Inicializando modelo...")
    model = CreditNet(input_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parámetros totales: {total_params:,}")

    history = train_model(model, train_loader, val_loader)

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n✓ Modelo guardado en {SAVE_PATH}")

    plot_history(history)
    print("\n¡Entrenamiento completado! Ahora corre: python evaluate.py")