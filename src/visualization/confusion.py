"""Confusion matrix + classification report.

- plot_confusion_sklearn : ConfusionMatrixDisplay (predict 노트북 스타일)
- plot_confusion_seaborn : sns.heatmap            (train 노트북 스타일)
"""
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

from src.config import STAGE_NAMES, N_CLASSES


def _align(y_true, y_pred):
    min_length = min(len(y_true), len(y_pred))
    return y_true[:min_length], y_pred[:min_length]


def plot_confusion_sklearn(y_true, y_pred, title='Confusion Matrix', save_path=None):
    y_true_cm, y_pred_cm = _align(y_true, y_pred)
    cm = confusion_matrix(y_true_cm, y_pred_cm, labels=range(N_CLASSES))

    fig, ax = plt.subplots(figsize=(10, 10))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=STAGE_NAMES)
    cm_display.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)

    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
    return cm


def plot_confusion_seaborn(y_true, y_pred, title='Confusion Matrix', save_path=None):
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return cm


def print_report(y_true, y_pred, header="--- 분류 성능 보고서 ---"):
    y_true_cm, y_pred_cm = _align(y_true, y_pred)
    report = classification_report(y_true_cm, y_pred_cm, target_names=STAGE_NAMES,
                                   zero_division=0)
    print(header)
    print(report)
    return report
