import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_model(model, val_loader, device):
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                # BERT-style batch: (input_ids, attention_mask, labels)
                input_ids, attention_mask, labels = [x.to(device) for x in batch]
                outputs = model(input_ids, attention_mask)
            elif len(batch) == 2:
                # LSTM-style batch: (inputs, labels)
                inputs, labels = [x.to(device) for x in batch]
                outputs = model(inputs)
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} items")

            # Get probabilities & predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            # For binary classification, store class-1 probabilities
            if probs.shape[1] > 1:
                y_probs.extend(probs[:, 1].cpu().numpy())
            else:
                y_probs.extend(probs[:, 0].cpu().numpy())

    # Metrics
    print("\n=== Evaluation Metrics ===")
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    try:
        auc = roc_auc_score(y_true, y_probs)
        print(f"ROC-AUC: {auc:.4f}")
    except ValueError:
        print("ROC-AUC: Cannot be computed (check number of classes)")

    mse = mean_squared_error(y_true, y_probs)
    mae = mean_absolute_error(y_true, y_probs)
    rmse = np.sqrt(mse)

    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Confusion Matrix Plot
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ✅ Make sure to return accuracy
    return acc
