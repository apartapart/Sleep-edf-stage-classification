"""EEG raw 신호 + hypnogram 오버레이 (단일 밤)."""
import matplotlib.pyplot as plt

from src.config import STAGE_MAP


def plot_eeg_with_hypnogram(times, data, labels, file_id, stage_map=STAGE_MAP,
                            save_path=None):
    """times: (n,), data: (n, C) -> 채널 0 만 표시, labels: (n,) 샘플 단위 라벨."""
    fig, ax1 = plt.subplots(figsize=(20, 6))

    ax1.plot(times, data[:, 0], color="steelblue", alpha=0.7, label="EEG")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("EEG (uV)", color="steelblue")
    ax1.tick_params(axis='y', labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.step(times, labels, where="post", color="darkred", linewidth=0.8, label="Sleep Stage")
    ax2.set_ylabel("Stage (0=W, 1=N1, 2=N2, 3=N3, 4=N4, 5=R)", color="darkred")
    ax2.tick_params(axis='y', labelcolor="darkred")
    ax2.set_yticks(list(stage_map.values()))
    ax2.set_yticklabels(list(stage_map.keys()))
    ax2.set_ylim(-1.5, 5.5)

    plt.title(f"{file_id} EEG + Hypnogram")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
