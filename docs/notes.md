# 개념 정리 노트 (노트북 markdown 이관)

## 진행 요약
1. Sleep-EDF(expanded, telemetry) 8.1GB 데이터 확보
2. 샘플 수 / 타입 파악 → 채널 5개, 샘플 약 400만 개
3. PSG `.edf` → `.fif` 변환 (mne 에서 다루기 편함). npy 로 바로 가지 않고 fif 경유
4. fif 기준 전체 normalization (z-score, train 통계만 사용 → 누수 방지)
5. 데이터 불균형 / 과적합 고려하여 mean, std 를 npy 로 저장해 추론 시 재사용
6. 여러 시도를 거쳐 **82%** 유의미한 결과 도출 (hold-out 피험자 ST7242J0)

## 데이터 merge 관련
- 시계열은 continuous → 파일 단위 concat 은 가능하지만 epoch 경계와 shape 관리 유의
- X/Y 를 merge 해서 학습하는 이유: 시간 패턴 + 채널 간 상관을 동시에 학습, long-term dependency, 메모리 효율

## Imbalanced data 대응
1. **Oversampling (SMOTE)** : noise 증폭 / 존재하지 않는 파형 생성 / overfitting 위험
2. **Class weighting** : 저빈도 클래스(N1) 오분류에 더 큰 loss 패널티. 실제 분포 유지, 효율성, 과적합 방지
- 결론은 class weighting 방향이었으나, **최종 저장 모델은 SMOTE 적용 + class_weight 미사용** 상태 (notebook 기준). 재검증 필요.
- `balanced` : 샘플 수 역수 비례 가중치 / `sqrt softening` : 가중치 효과 완화

## 모델 관련
- **LSTM** : vanishing gradient 로 깊게 쌓으면 학습 어려움 → 얕게. CNN 등 다른 구조도 열어둘 것
- 연속 신호라 LSTM 이 유리할 줄 알았으나 **CNN(front-end) 이 더 효과적**인 결과 관찰 → 추가 정리 필요
- **MaxPooling1D** : downsample, 중요 특징 보존, 계산량/noise 감소
- **GlobalAveragePooling1D / Dropout** : 파라미터 수 감소, 과적합 방지. 소규모 데이터에서 특히 중요
- **ReLU** : x>0 에서 기울기 1 → gradient 손실 없음
- **softmax vs sigmoid** : 다중 클래스 중 하나 선택 → softmax 사용
- **padding** : `same` 출력=입력 크기 / `valid` 출력<입력
- **optimizer** : Adam, lr=1e-4 (lr 튜닝은 보류)
- **EarlyStopping** : val_loss 개선 없으면 patience=10 후 중단, best weight 복원

## 라이브러리
- **mne** : EEG 특화 라이브러리
- **pyedflib** : EDF 헤더/신호 저수준 접근
- **glob** : 패턴 기반 파일 탐색
