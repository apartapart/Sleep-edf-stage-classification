"""Step 8. 배포용 추론 (PDF 10장 VSCode 버전): edf -> 세그먼트 병렬 추론 -> 후처리 -> JSON.

python scripts/08_stream_predict.py --edf ST7242J0-PSG.edf --model artifacts/model_4.keras \
    --mean artifacts/mean.npy --std artifacts/std.npy --out results/deploy
옵션: --no-filter (band-pass 끔, 학습 분포와 일치) / --no-postprocess / --segment-min 15 / --cores 2
"""
import argparse
import os
import _path  # noqa: F401

from src.config import DEPLOY_SEGMENT_SEC, DEPLOY_NUM_CORES
from src.deploy.stream_predict import stream_predict, save_prediction_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edf", required=True)
    p.add_argument("--model", default="artifacts/model_4.keras")
    p.add_argument("--mean", default="artifacts/mean.npy")
    p.add_argument("--std", default="artifacts/std.npy")
    p.add_argument("--out", default="results/deploy")
    p.add_argument("--segment-min", type=float, default=DEPLOY_SEGMENT_SEC / 60)
    p.add_argument("--cores", type=int, default=DEPLOY_NUM_CORES)
    p.add_argument("--no-filter", action="store_true")
    p.add_argument("--no-postprocess", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pred, sfreq = stream_predict(args.edf, args.model, args.mean, args.std,
                                 segment_sec=args.segment_min * 60, num_cores=args.cores,
                                 apply_filter=not args.no_filter,
                                 postprocess=not args.no_postprocess)
    print(f"predicted epochs: {len(pred)}")
    save_prediction_json(args.edf, sfreq, pred, args.out)


if __name__ == "__main__":
    main()
