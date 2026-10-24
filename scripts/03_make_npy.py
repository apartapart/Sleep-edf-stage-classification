"""Step 3. fif + hypnogram -> 샘플 라벨 -> 30s epoch -> X/Y npy.

python scripts/03_make_npy.py --fif ./data/fif/ST7242J0-PSG-raw.fif --hyp ...-Hypnogram.edf --out ./data/npy --plot
"""
import argparse
import os
import _path  # noqa: F401

from src.config import RAW_EDF_DIR, EPOCH_DURATION
from src.preprocessing.labeling import (load_fif_with_annotations,
                                        build_sample_labels, print_label_counts)
from src.preprocessing.epoching import make_epochs, save_npy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fif", default="/content/ST7242J0-PSG-raw.fif")
    p.add_argument("--hyp", default=f"{RAW_EDF_DIR}/ST7242JO-Hypnogram.edf")
    p.add_argument("--out", default="/content/")
    p.add_argument("--epoch", type=int, default=EPOCH_DURATION)
    p.add_argument("--plot", action="store_true", help="EEG+hypnogram, 단계 분포 시각화")
    p.add_argument("--train-style", action="store_true",
                   help="PDF 4-6 학습용 방식: undefined->W(0), epoch 스킵 없음 (model_4 학습 데이터와 동일)")
    p.add_argument("--fig-dir", default="results/figures")
    args = p.parse_args()

    try:
        raw, annotations = load_fif_with_annotations(args.fif, args.hyp)
    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {e}")
        return

    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    data = raw.get_data().T  # (n_samples, n_channels)

    labels = build_sample_labels(annotations, sfreq, n_samples,
                                 undefined_value=0 if args.train_style else -1)
    print_label_counts(labels)

    if args.plot:
        from src.visualization.eeg_hypnogram import plot_eeg_with_hypnogram
        from src.visualization.stage_distribution import plot_stage_distribution
        os.makedirs(args.fig_dir, exist_ok=True)
        file_id = os.path.basename(args.fif).replace("-raw.fif", "")
        plot_eeg_with_hypnogram(raw.times, data, labels, file_id,
                                save_path=os.path.join(args.fig_dir, f"{file_id}_eeg_hypnogram.png"))
        plot_stage_distribution(labels,
                                save_path=os.path.join(args.fig_dir, f"{file_id}_stage_distribution.png"))

    X, Y = make_epochs(data, labels, sfreq, epoch_duration=args.epoch,
                       skip_undefined=not args.train_style)
    print("X shape:", X.shape, "Y shape:", Y.shape)

    os.makedirs(args.out, exist_ok=True)
    save_npy(X, Y, args.fif, args.out)


if __name__ == "__main__":
    main()
