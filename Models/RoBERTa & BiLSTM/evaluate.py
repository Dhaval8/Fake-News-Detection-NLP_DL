import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, val_loader, device, plot_cm=True):
    """
    Evaluate the model on val_loader.

    Returns:
      dict of metrics and also shows confusion matrix plot if plot_cm=True.
    """
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            # if binary, use probs[:,1], otherwise pad with predicted prob for class index
            if probs.shape[1] >= 2:
                y_probs.extend(probs[:, 1].cpu().numpy())
            else:
                y_probs.extend(probs[:, 0].cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['classification_report'] = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm

    # ROC AUC only when there are at least two classes present in y_true
    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = None
    metrics['roc_auc'] = auc

    mse = mean_squared_error(y_true, y_probs)
    mae = mean_absolute_error(y_true, y_probs)
    rmse = np.sqrt(mse)

    metrics['mse'] = mse
    metrics['mae'] = mae
    metrics['rmse'] = rmse

    # Print the metrics (keeps same behavior as your old evaluate.py)
    print("\n=== Evaluation Metrics ===")
    print("Accuracy:", metrics['accuracy'])
    print("Classification Report:\n", metrics['classification_report'])
    print("Confusion Matrix:\n", cm)
    if auc is not None:
        print(f"ROC-AUC: {auc:.4f}")
    else:
        print("ROC-AUC: could not compute (maybe single-class in y_true)")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    if plot_cm:
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    return metrics
