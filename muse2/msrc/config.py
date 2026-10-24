"""Muse 2 실험 상수 (노트북 MUSE2_npy_lstm_or_cnn_predict 기준)."""
FS = 256                 # 모델이 가정한 Hz. 실제 Muse CSV 는 ~250 Hz (README 이슈)
SEQ_SEC = 5
SEQ_LEN = FS * SEQ_SEC   # 1280
STRIDE_SEC = 5           # Sleep-EDF 기초 학습 stride
STRIDE_INFER = FS        # CSV 추론 stride = 1 s
MUSE_CHANNELS = ["AF7", "AF8"]
BANDPASS = dict(low=0.5, high=40.0, order=4)
CLIP = 100.0
WAKE_RATIO = 0.6         # 5 s 중 W 가 60% 이상이면 label 1 (Wake)
BEST_THRESHOLD = 0.779   # Attention_Final 의 val F1 최적 임계값 (노트북 출력)
HYSTERESIS = dict(high=0.55, low=0.45)
ACTIVITY_PCT = (30, 70)  # pseudo-label 하위/상위 백분위
