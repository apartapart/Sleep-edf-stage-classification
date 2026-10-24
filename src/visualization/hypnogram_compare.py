"""Ground truth vs predicted hypnogram 2단 비교."""
import numpy as np
import matplotlib.pyplot as plt

from src.config import STAGE_NAMES, EPOCH_DURATION


def plot_hypnogram_comparison(y_true, y_pred, epoch_duration=EPOCH_DURATION,
                              save_path=None):
    min_length = min(len(y_true), len(y_pred))
    y_true_plot = y_true[:min_length]
    y_pred_plot = y_pred[:min_length]

    num_epochs = len(y_true_plot)
    times = np.arange(num_epochs) * epoch_duration / 60

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True, sharey=True)

    # 실제 하이프노그램 플롯
    ax1.step(times, y_true_plot, where='post', color='darkblue')
    ax1.set_title('Actual Hypnogram (Ground Truth)')
    ax1.set_ylabel('Sleep Stage')
    ax1.set_yticks(list(range(len(STAGE_NAMES))))
    ax1.set_yticklabels(STAGE_NAMES)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 예측 하이프노그램 플롯
    ax2.step(times, y_pred_plot, where='post', color='darkred')
    ax2.set_title('Predicted Hypnogram')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Sleep Stage')
    ax2.set_yticks(list(range(len(STAGE_NAMES))))
    ax2.set_yticklabels(STAGE_NAMES)
    ax2.grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
