"""PSG .edf -> 채널 선택 후 .fif 저장 (mne).

fif 를 거치는 이유: mne 에서 annotation 과 함께 빠르게 재로딩 가능.
"""
import os
import mne

from src.config import CHANNELS


def convert_edf_to_fif(psg_file: str, hyp_file: str, out_dir: str,
                       channels=CHANNELS) -> str:
    """EDF 로드 -> hypnogram annotation 부착 -> 채널 선택 -> fif 저장. 저장 경로 반환."""
    raw = mne.io.read_raw_edf(psg_file, preload=True)

    ann = mne.read_annotations(hyp_file)
    raw.set_annotations(ann)

    # 노트북: raw.pick_channels([...]) (mne>=1.5 deprecated) -> 동일 동작
    raw.pick(list(channels))

    out_filename = os.path.basename(psg_file).replace(".edf", "-raw.fif")
    out_file_path = os.path.join(out_dir, out_filename)

    raw.save(out_file_path, overwrite=True)

    print(f"파일이 성공적으로 저장되었습니다: {out_file_path}")
    return out_file_path
