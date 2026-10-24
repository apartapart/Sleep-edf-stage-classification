"""Step 1. 원본 PSG EDF 메타 확인.

python scripts/01_inspect_edf.py --psg /path/ST7242J0-PSG.edf
"""
import argparse
import _path  # noqa: F401

from src.config import RAW_EDF_DIR
from src.preprocessing.edf_inspect import inspect_edf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--psg", default=f"{RAW_EDF_DIR}/ST7242J0-PSG.edf")
    args = p.parse_args()
    inspect_edf(args.psg)


if __name__ == "__main__":
    main()
