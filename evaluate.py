import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)

def get_metrics(y_true, y_pred, labels=None):
    """
    Calculates global and per-class metrics.
    """
    # Global metrics
    metrics = {
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    # Per-class metrics
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

def run_full_evaluation(clf, X, y, label_names=None, title="Model Results"):
    """
    Runs prediction and returns all requested metrics and a plot.
    """
    y_pred = clf.predict(X)
    
    # Get unique labels in the data to ensure alignment
    unique_labels = np.unique(y)
    
    # Calculate Metrics
    global_metrics, per_class_metrics = get_metrics(y, y_pred, labels=unique_labels)
    
    # Display Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y, y_pred, labels=unique_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title(f"Confusion Matrix: {title}")
    plt.show()
    
    return global_metrics, per_class_metrics
