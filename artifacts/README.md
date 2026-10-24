# Artifacts

| 파일 | 내용 |
|---|---|
| `model_4.keras` | CNN-BiLSTM, Keras 3.10 저장 (2025-09-11). telemetry ST7* 학습. 187,398 params, input (3000, 2) |
| `mean.npy` | z-score 평균, shape (1,1,2), 단위 V. train 셋 통계 |
| `std.npy` | z-score 표준편차, shape (1,1,2), 단위 V |

주의: model_4 학습 split 에 `ST7241J0` 포함 → `ST7242J0` 평가는 피험자 누수. cassette 평가는 무관.
