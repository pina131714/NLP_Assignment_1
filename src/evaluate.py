import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)
import os

def get_metrics(y_true, y_pred, labels=None):
    """
    Calculates global and per-class metrics.
    """
    metrics = {
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    p_class = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    r_class = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    
    per_class_df = pd.DataFrame({
        "Label": labels if labels is not None else np.unique(y_true),
        "Precision": p_class,
        "Recall": r_class,
        "F1-Score": f1_class
    })
    
    return metrics, per_class_df

def run_full_evaluation(clf_clarity, clf_evasion, X, y_clarity, y_evasion, 
                       label_names_clarity=None, label_names_evasion=None, 
                       title="Model Results", save_dir="figures"):

    os.makedirs(save_dir, exist_ok=True)
    
    y_pred_clarity = clf_clarity.predict(X)
    y_pred_evasion = clf_evasion.predict(X)
    
    unique_labels_clarity = np.unique(y_clarity)
    unique_labels_evasion = np.unique(y_evasion)
    
    global_metrics_clarity, per_class_metrics_clarity = get_metrics(
        y_clarity, y_pred_clarity, labels=unique_labels_clarity
    )
    
    global_metrics_evasion, per_class_metrics_evasion = get_metrics(
        y_evasion, y_pred_evasion, labels=unique_labels_evasion
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    cm_clarity = confusion_matrix(y_clarity, y_pred_clarity, labels=unique_labels_clarity)
    disp_clarity = ConfusionMatrixDisplay(
        confusion_matrix=cm_clarity, 
        display_labels=label_names_clarity if label_names_clarity else unique_labels_clarity
    )
    disp_clarity.plot(cmap='Blues', ax=axes[0], values_format='d')
    axes[0].set_title(f"Confusion Matrix: {title} - Clarity")
    
    cm_evasion = confusion_matrix(y_evasion, y_pred_evasion, labels=unique_labels_evasion)
    disp_evasion = ConfusionMatrixDisplay(
        confusion_matrix=cm_evasion, 
        display_labels=label_names_evasion if label_names_evasion else unique_labels_evasion
    )
    disp_evasion.plot(cmap='Greens', ax=axes[1], values_format='d')
    axes[1].set_title(f"Confusion Matrix: {title} - Evasion")
    
    plt.tight_layout()
    
    safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_")
    save_path = os.path.join(save_dir, f"confusion_matrices_{safe_title}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved to: {save_path}")
    
    plt.show()
    
    results = {
        "clarity": {
            "global_metrics": global_metrics_clarity,
            "per_class_metrics": per_class_metrics_clarity,
            "predictions": y_pred_clarity
        },
        "evasion": {
            "global_metrics": global_metrics_evasion,
            "per_class_metrics": per_class_metrics_evasion,
            "predictions": y_pred_evasion
        }
    }
    
    return results
