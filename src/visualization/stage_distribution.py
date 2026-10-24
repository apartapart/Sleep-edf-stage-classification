"""수면 단계 분포 bar plot (클래스 불균형 확인)."""
import numpy as np
import matplotlib.pyplot as plt


def plot_stage_distribution(labels, save_path=None):
    unique_labels, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(unique_labels, counts, tick_label=[f"Stage {l}" for l in unique_labels])
    ax.set_xlabel("Sleep Stage")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Distribution of Sleep Stages")
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
