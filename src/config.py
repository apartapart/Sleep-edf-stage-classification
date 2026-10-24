"""전역 상수 및 기본 경로.

노트북(kakao_final_sleep_edf_predict, kakao_npy_lstm_or_cnn_predict)에서
하드코딩되어 있던 값들을 한 곳에 모음. 경로는 CLI 인자로 덮어쓸 수 있음.
"""
import os

# ---------------------------------------------------------------- stages ----
# Sleep-EDF(telemetry) annotation description -> 정수 라벨
STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
}
STAGE_NAMES = ["W", "N1", "N2", "N3", "N4", "R"]
N_CLASSES = len(STAGE_NAMES)
UNDEFINED_LABEL = -1  # Movement / Sleep stage ? 등 매핑 안 되는 구간

# ---------------------------------------------------------------- signal ----
CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]   # 사용 채널 2개
EPOCH_DURATION = 30                       # seconds (AASM 표준 epoch)
SFREQ = 100                               # Hz (Sleep-EDF EEG)
SAMPLES_PER_EPOCH = SFREQ * EPOCH_DURATION  # 3000
INPUT_SHAPE = (SAMPLES_PER_EPOCH, len(CHANNELS))  # (3000, 2)

# ---------------------------------------------------------------- paths -----
# Colab 기준 기본 경로. 로컬에서는 CLI --* 인자 또는 환경변수 SLEEP_ROOT 로 변경.
SLEEP_ROOT = os.environ.get(
    "SLEEP_ROOT", "/content/drive/MyDrive/sleep-telemetry"
)
RAW_EDF_DIR = os.path.join(
    SLEEP_ROOT, "sleep-edf-database-expanded-1.0.0", "sleep-telemetry"
)
MERGE_DIR = os.path.join(SLEEP_ROOT, "merge_dataset_tel")       # X_/Y_*.npy
STATS_DIR = os.path.join(SLEEP_ROOT, "exam_file_dataset_tel")   # mean.npy / std.npy
MODEL_DIR = os.path.join(SLEEP_ROOT, "sleep_small_model")       # *.keras

# ---------------------------------------------------------------- training --
RANDOM_STATE = 42
TEST_SIZE = 0.3          # train : (val+test) = 7 : 3  (PDF 6-1: 0.4 -> 0.3 으로 변경, model_4 기준)
                         # 노트북 kakao_npy_lstm_or_cnn_predict 는 0.4 (model_5 실험)
VAL_TEST_SPLIT = 0.5     # (val+test) 를 다시 절반
LEARNING_RATE = 1e-4
EPOCHS = 50
BATCH_SIZE = 128
ES_PATIENCE = 10

# ---------------------------------------------------------------- deploy ----
# PDF 9~10장 배포 코드. 추론 시에만 band-pass 적용 (학습엔 미적용 -> 불일치, README 참고)
DEPLOY_FILTER = dict(l_freq=0.5, h_freq=30.0, fir_design="firwin")
DEPLOY_SEGMENT_SEC = 10 * 60      # 세그먼트 단위 (VSCode 버전 10 min, Colab 버전 15 min)
DEPLOY_TARGET_SEC = 10 * 60 * 60  # Colab 버전: 10h 로 0-padding / 자르기
DEPLOY_NUM_CORES = 2
