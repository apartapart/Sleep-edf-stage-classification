# KAKAO_AI.pdf (프로젝트 보고서) 요약 — 노트북과 다른 점

보고서 목차: Download Dataset → Colab → Drive/Install → Preprocessing → Analysis → CNN&LSTM → Predict → Result → Adjust model → VSCode 배포

## 노트북과 다른 설정 (model_4 기준은 PDF 쪽)
| 항목 | PDF (model_4) | 노트북 (model_5 실험) | 레포 반영 |
|---|---|---|---|
| split | `test_size=0.3` → 70/15/15 | 0.4 | `config.TEST_SIZE=0.3` |
| undefined 라벨 | `stage_map.get(desc, 0)` → W | `-1` 후 epoch 스킵 | `03_make_npy --train-style` |
| epoch 루프 | `range(0, n - step, step)`, 스킵 없음 | `-1` 포함 epoch 제외 | `make_epochs(skip_undefined=False)` |
| 불균형 | class_weight 시도 → 저조 → SMOTE 채택 | 동일 | 동일 |
| 모델 저장 | `.keras` (h5 대비 장점 비교표) | 동일 | 동일 |

## 배포 코드 (PDF 9~10장, 노트북에 없음) → `src/deploy/`, `scripts/08_stream_predict.py`
- `raw.filter(l_freq=0.5, h_freq=30.0, fir_design='firwin')` — **추론에만** 적용. 학습 미적용 → 불일치. **실측: SC4001E0 full κ 0.76 → 0.18, W 가 N2 로 붕괴** (README 이슈 10)
- Colab 버전: 10h 로 zero-padding/자르기, 15 min 세그먼트 → 40 세그먼트
- VSCode 버전: 10 min 세그먼트, `ProcessPoolExecutor(max_workers=2)`, 세그먼트마다 모델 재로드
- 후처리: `[R, N2, R] → [R, R, R]`, 연속 W 30 epoch 이상이면 첫 W + 30 까지 자름
- 출력: `{file, sampling_rate, total_epochs, predicted_classes}` JSON → 프론트에서 시간 매핑
- 환경: anaconda, `C:\AI-sleep-service\app\{models,psg_file}`

## 기타
- SC4001E0 (cassette) 는 5장 "Analysis Data" 분포 시각화에만 사용. 학습 미사용
- 샘플 분포 (SC4001E0, 샘플 단위): W 179000 / N1 66000 / N2 1773000 / N3 294000 / N4 123000 / R 510000 / undefined 3000
- 저자 자평: "accuracy 82% 가 논문 최소 목표라 만족. 1달 더 있었으면 더 올렸을 것"
