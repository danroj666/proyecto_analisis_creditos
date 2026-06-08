"""
evaluate.py
-----------
Carga el modelo entrenado y evalúa métricas completas en el test set.

Uso:
    python evaluate.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, roc_curve
)

from data.generate_data import generate_credit_data
from model import CreditNet, Preprocessor


def evaluate_model():
    # ── Cargar datos y preprocesador ──
    print("Cargando datos y modelo...\n")
    df   = generate_credit_data(n=1000)
    prep = Preprocessor()
    _, _, test_loader, input_dim = prep.fit_transform(df)

    # ── Cargar modelo ──
    model = CreditNet(input_dim)
    model.load_state_dict(torch.load('credit_model.pt', weights_only=True))
    model.eval()

    # ── Predicciones ──
    probs, preds, labels = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            p = model(xb)
            probs.extend(p.numpy())
            preds.extend((p > 0.5).float().numpy())
            labels.extend(yb.numpy())

    probs  = np.array(probs)
    preds  = np.array(preds)
    labels = np.array(labels)

    # ── Métricas ──
    print("=" * 55)
    print("EVALUACIÓN FINAL — TEST SET")
    print("=" * 55)
    print(classification_report(labels, preds,
          target_names=['Mal crédito', 'Buen crédito']))

    auc = roc_auc_score(labels, probs)
    print(f"AUC-ROC: {auc:.4f}  (> 0.70 es aceptable en industria)")

    # ── Gráficas ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0],
                xticklabels=['Malo', 'Bueno'],
                yticklabels=['Malo', 'Bueno'],
                cmap='Blues')
    axes[0].set_title('Matriz de Confusión')
    axes[0].set_ylabel('Real')
    axes[0].set_xlabel('Predicho')

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    axes[1].plot(fpr, tpr, color='#2563eb', lw=2, label=f'AUC = {auc:.3f}')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.4)
    axes[1].set_title('Curva ROC')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150)
    print("\n✓ Gráficas guardadas en evaluation_results.png")
    plt.show()


if __name__ == '__main__':
    evaluate_model()