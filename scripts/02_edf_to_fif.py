"""Step 2. EDF -> 채널 2개 선택 -> fif 저장.

python scripts/02_edf_to_fif.py --psg ...-PSG.edf --hyp ...-Hypnogram.edf --out ./data/fif
"""
import argparse
import os
import _path  # noqa: F401

from src.config import RAW_EDF_DIR, CHANNELS
from src.preprocessing.edf_to_fif import convert_edf_to_fif


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--psg", default=f"{RAW_EDF_DIR}/ST7242J0-PSG.edf")
    p.add_argument("--hyp", default=f"{RAW_EDF_DIR}/ST7242JO-Hypnogram.edf")
    p.add_argument("--out", default="/content/")
    p.add_argument("--channels", nargs="+", default=list(CHANNELS))
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    convert_edf_to_fif(args.psg, args.hyp, args.out, channels=args.channels)


if __name__ == "__main__":
    main()
