# 원본 노트북 보고서 (수정 없이 보존)

> 아래는 Colab 노트북 markdown 셀 원문. 수치·해석은 당시 기준이며, 재검증 결과는 `muse2/README.md` 참고.

---

## 수면-각성 탐지 모델 개발 최종 보고서

## 프로젝트 개요
본 프로젝트는 `sleep-edf-database-expanded-1.0.0`에서 제공하는 EDF 형식의 수면다원검사(PSG) 데이터와 Muse 뇌파(EEG) 데이터를 활용하여 수면-각성 상태를 자동으로 탐지하는 딥러닝 모델을 개발하는 것을 목표로 합니다. 데이터 전처리, 딥러닝 모델 구축 및 학습, 활동성 기반 미세 조정, 그리고 최종 성능 평가에 이르는 전 과정을 다룹니다.

## 1. 수면 원격 측정 데이터 전처리 및 `.fif`, `.npy` 변환

### 3.1 EDF 데이터 로딩 및 정보 확인
- **데이터 경로 정의**: `base_dir`을 `/content/drive/MyDrive/sleep-telemetry/sleep-edf-database-expanded-1.0.0/sleep-telemetry`로 설정하고, `first_path_list`와 `second_path_list`에 정의된 고유 식별자를 사용하여 PSG(`-PSG.edf`) 파일과 Hypnogram(`-Hypnogram.edf`) 파일의 전체 경로를 생성했습니다. `paired_paths`를 통해 각 환자별 PSG와 Hypnogram 파일이 올바르게 매칭되었음을 확인했습니다.
- **EDF 파일 정보 확인**: `pyedflib.EdfReader`를 사용하여 각 PSG 파일의 헤더 정보와 채널별 상세 정보(채널명, 샘플 수, 단위)를 출력하여 데이터 구조를 파악했습니다. `EEG Fpz-Cz`, `EEG Pz-Oz`, `EOG horizontal`, `EMG submental`, `Marker` 등의 채널이 포함되어 있으며, 뇌파 분석의 핵심 채널인 `EEG Fpz-Cz`와 `EEG Pz-Oz`에 집중했습니다.

### 3.2 뇌파 채널 선택 및 `.fif` 변환
- `mne` 라이브러리를 사용하여 EDF 파일을 읽고, `EEG Fpz-Cz`와 `EEG Pz-Oz` 두 채널만 선택한 후 MNE의 자체 이진 형식인 `.fif` 파일로 `/content/drive/MyDrive/sleep-telemetry/sleep_fif` 경로에 저장했습니다.
- **근거**: `.fif` 형식은 MNE에 최적화된 파일 형식으로, 빠른 데이터 로딩과 효율적인 저장을 가능하게 합니다. 불필요한 채널을 미리 제거하여 후속 처리 단계를 단순화하고 계산 효율성을 높였습니다.

### 3.3 데이터 병합 및 `.npy` 변환 (수면 단계 분류를 위한 에포크 생성)
- `stage_map`을 정의하여 EDF Hypnogram 파일의 텍스트 기반 수면 단계 주석을 수치형 레이블(0, 1, 2, 3, 4, 5)로 매핑했습니다.
- `time_step_sec = 30`으로 30초를 하나의 수면 에포크(epoch) 단위로 설정했습니다. 이는 수면 단계 분류의 표준 관행입니다.
- 각 `.fif` 파일에서 원시 데이터를 로드하고, Hypnogram 파일에서 읽은 주석을 `sfreq` (샘플링 주파수)를 사용하여 각 샘플에 해당하는 수면 단계 레이블 배열로 변환했습니다.
- 전체 데이터를 30초 간격으로 슬라이딩하여 `X` (데이터 윈도우)와 `Y` (해당 윈도우의 가장 빈번한 수면 단계 레이블)를 생성했습니다. `np.bincount(labels[i:i+time_step]).argmax()`를 사용하여 30초 내에서 가장 많이 나타나는 수면 단계를 해당 에포크의 대표 레이블로 지정했습니다.
- 생성된 `X`와 `Y` 데이터를 `/content/drive/MyDrive/sleep-telemetry/merge_dataset_tel` 경로에 NumPy의 바이너리 형식인 `.npy` 파일로 저장했습니다.
- **근거**: `.npy` 형식은 데이터 누수 방지 및 효율적인 I/O를 제공하여 대용량 시계열 데이터 처리 및 기계 학습 모델 입력에 매우 적합합니다.

## 2. 초기 딥러닝 모델 개발 및 학습

### 2.1 데이터 로딩 및 준비
- Google Drive에서 `/content/drive/MyDrive/sleep-telemetry/muse_merge_dataset_tel` 경로의 `.npy` 파일을 `/content/dataset` 로컬 디렉토리로 복사하여 학습 효율성을 높였습니다. 16명의 피험자 중 11명은 학습, 3명은 검증, 2명은 테스트 데이터로 분할했습니다.
- **하이퍼파라미터**: `BATCH_SIZE = 32`, `SAMPLING_RATE = 256`, `SEQ_SEC = 5`, `STRIDE_SEC = 5`, `SEQ_LEN = 1280`으로 설정되었습니다.

### 2.2 전처리 및 데이터 증강
- **전처리 함수**: `bandpass_filter` (0.5-40Hz 대역 통과 필터) 및 `is_noisy` (노이즈 샘플 제거) 함수가 정의되었습니다. 각 샘플은 표준화(`Z-score normalization`) 과정을 거쳤습니다.
- **데이터 제너레이터 (`data_generator`)**: 데이터 불균형 해소를 위해 Wake(0)과 Sleep(1) 샘플의 비율을 조절하고, 학습 데이터에 대해 임의 노이즈 추가 및 스케일링을 통한 데이터 증강을 적용했습니다.

### 2.3 모델 아키텍처 및 학습
- **모델 아키텍처 (`build_model`)**:
    - 입력 레이어 `(SEQ_LEN, 2)`.
    - `Conv1D` (32, 7), `BatchNormalization`, `MaxPooling1D` (4).
    - `Conv1D` (64, 5), `BatchNormalization`, `MaxPooling1D` (4).
    - `Bidirectional GRU` (32)를 통한 시퀀스 학습.
    - **어텐션 메커니즘**: `Dense(1, activation='tanh')`와 `Softmax`를 활용하여 시퀀스 내 중요한 특징에 가중치를 부여했습니다.
    - `GlobalAveragePooling1D`를 사용하여 `Lambda` 레이어 사용 시 발생할 수 있는 모델 저장/로드 에러를 방지했습니다.
    - `Dropout` (0.4), `Dense` (16, relu), `Dense` (1, sigmoid)를 통한 최종 분류.
- **컴파일**: `Adam` (1e-4), `binary_crossentropy`, `accuracy`.
- **학습 진행**: `train_ds`와 `val_ds`는 `tf.data.Dataset.from_generator`를 통해 생성되었으며, `EarlyStopping` (patience=7) 콜백을 사용하여 과적합을 방지했습니다.

### 2.4 초기 학습 결과 및 모델 저장
- **평가**: `find_best_threshold` 함수를 통해 최적의 분류 임계값(F1-score 기반)을 탐색하고, `evaluate` 함수로 최종 성능(precision, recall, f1-score)을 평가했습니다.
- **초기 학습 결과**: 15 epoch까지 학습을 진행한 결과, 검증 세트에서 안정적인 성능을 보였습니다. 최종적으로 테스트 세트에서 최적 임계값 0.779를 사용했을 때, 전체 정확도 95%의 결과를 얻었습니다. Wake 클래스(0)에 대한 Precision 0.98, Recall 0.95, Sleep 클래스(1)에 대한 Precision 0.85, Recall 0.94를 기록하여 높은 분류 성능을 확인했습니다.
- **모델 저장**: 학습된 모델 (`MUSE2_Attention_Final.keras`)과 가중치 (`MUSE2_Attention_Weights.weights.h5`)는 `/content/drive/MyDrive/sleep-telemetry/` 경로에 저장되었습니다.

## 3. 데이터 분포 분석

모델 학습에 앞서, 데이터셋의 클래스 불균형 문제를 확인하기 위해 `analyze_dataset_distribution` 함수를 통해 전체 피험자별 Wake/Sleep 비율을 분석했습니다.
- **분석 대상**: `/content/dataset` 경로에 복사된 `Y_ST7{fid}-PSG.npy` 파일들을 분석했습니다.
- **Wake/Sleep 정의**: Sleep-EDF 원본 기준 0을 Wake로, 그 외 모든 수면 단계(1, 2, 3, 4, R)를 Sleep으로 간주했습니다.
- **결과**: 피험자별로 Sleep 비율이 53.40%에서 92.90%까지 다양하게 나타났으며, 전체 데이터 평균 졸음 비율은 81.86%로 Sleep 클래스가 압도적으로 많아 데이터 불균형이 존재함을 확인했습니다. 이는 모델 학습 시 Wake 클래스에 대한 학습 부족을 야기할 수 있음을 시사했습니다.

## 4. 활동성 기반 미세 조정 (Finetuning) 모델 개발

초기 모델의 전이 학습을 활용하여 활동성(Activity) 기반으로 수면-각성 분류를 보다 정교하게 수행하는 파인튜닝 모델을 개발했습니다.

### 4.1 새로운 데이터셋 및 활동성 기반 라벨링
- **새로운 데이터셋**: `/content/awake_study.csv`, `/content/awake_study2.csv`, `/content/awake_study3.csv`, `/content/bedtime.csv`, `/content/bedtime2.csv`와 같은 CSV 형식의 Muse 데이터를 활용했습니다. 이 데이터는 실제 사용 환경에 가까운 데이터를 모의하기 위해 사용되었습니다.
- **활동성(Activity) 기반 라벨링**: EEG 신호의 활동성을 측정하는 `get_activity` (절대값 미분 평균)와 `get_spike_ratio` (스파이크 비율) 함수를 정의했습니다. `get_activity`를 통해 전체 데이터의 활동성 분포를 파악하고, `low_thr` (30th percentile)과 `high_thr` (70th percentile)을 설정하여 **Sleep-like (0.0)** 및 **Awake-like (1.0)** 라벨을 자동으로 생성했습니다. 이는 EEG 신호 자체의 특성을 활용하여 수면-각성 구간을 구분하는 새로운 접근법입니다.
    - `GLOBAL ACTIVITY THRESHOLD` 결과: LOW: 0.505, HIGH: 0.869로 설정되었습니다.

### 4.2 데이터 증강 및 미세 조정 학습
- **데이터 증강**: `augment` 함수를 통해 학습 데이터에 노이즈와 스케일링을 적용하여 모델의 강건함을 높였습니다.
- **미세 조정 모델 구조**: 이전에 학습된 모델 (`MUSE_pretrained_2.keras`)을 불러와 모든 레이어를 `trainable=True`로 설정하고 미세 조정했습니다. 옵티마이저, 손실 함수, 지표는 초기 모델과 동일하게 사용했습니다.
- **미세 조정 학습**: `train_ds`와 `val_ds`는 `muse_generator`를 통해 활동성 기반으로 라벨링된 데이터를 사용하여 생성되었습니다. `steps_per_epoch=25`, `validation_steps=10`, `epochs=20`으로 설정되었으며, `EarlyStopping` (patience=3) 콜백을 적용했습니다.

### 4.3 미세 조정 학습 결과 및 모델 저장
- **학습 결과**: 미세 조정 학습 과정에서 Loss가 점진적으로 감소하고 Accuracy가 향상되는 것을 확인했습니다. 최종적으로 Epoch 20에서 가장 좋은 성능을 달성하여 가중치가 복원되었습니다.
- **모델 저장**: 미세 조정된 모델 (`MUSE_activity_model.keras`)과 가중치 (`MUSE_activity_weights.weights.h5`)는 `/content/drive/MyDrive/sleep-telemetry` 경로에 저장되었습니다.

## 5. 추론 및 시각화

미세 조정된 모델을 사용하여 새로운 데이터에 대한 수면-각성 추론을 수행하고, 결과를 시각화하여 모델의 작동 방식을 직관적으로 이해했습니다.
- **모델 로드**: `/content/drive/MyDrive/sleep-telemetry/MUSE_activity_model.keras`에 저장된 미세 조정 모델을 로드했습니다. Keras의 `safe_mode=False` 옵션을 사용하여 모든 커스텀 레이어(특히 `Lambda` 레이어 대체 전)가 올바르게 로드되도록 했습니다.
- **전처리**: `bandpass` 필터링, 중앙값 제거, 클리핑, Z-score 정규화 등의 과정을 거쳤습니다.
- **이력 현상(Hysteresis) 적용**: `hysteresis` 함수를 사용하여 모델의 확률 예측값에 기반하여 최종 수면-각성 상태(0: Sleep, 1: Awake)를 결정했습니다. `high=0.55`, `low=0.45` 임계값을 적용하여 전환 구간에서의 불안정성을 줄였습니다.
- **추론 및 시각화 (`run_inference`)**: `awake_study.csv`와 `bedtime.csv` 파일을 대상으로 추론을 실행하고, 시간에 따른 Awake 확률과 이력 현상이 적용된 예측 상태를 그래프로 시각화했습니다. 이를 통해 모델이 실제 EEG 신호 변화에 어떻게 반응하여 수면-각성 상태를 추정하는지 확인했습니다.

## 6. 개선된 추론 및 최종 평가

모델의 예측 결과가 특정 구간에 집중되는 경향을 개선하고, 보다 명확한 수면-각성 분류를 위해 확률값 스케일링을 추가한 추론 프로세스를 도입했습니다.

- **추론 개선 (`run_all_inference_v2`)**: 기존 추론 방식에서 모델 예측 확률이 0.5 근처에 밀집되는 문제를 해결하기 위해, 각 파일별 예측 확률 (`preds`)에 대해 **Min-Max Scaling**을 적용하여 0과 1 사이로 강제로 분포를 확장했습니다. (`preds_scaled = (preds - np.min(preds)) / (np.max(preds) - np.min(preds) + 1e-6)`).
- **확실한 분류**: 스케일링된 `preds_scaled`에 `hysteresis` 함수를 다시 적용하여 0.5를 기준으로 보다 명확하게 수면-각성 상태를 분류했습니다.
- **시각화**: 개선된 추론 결과를 `awake_study.csv`, `awake_study2.csv`, `awake_study3.csv`, `bedtime.csv`, `bedtime2.csv` 파일에 대해 시각화하여, 스케일링을 통해 예측 확률 분포가 0~1 전체 범위로 확대되고 수면-각성 전환이 더욱 뚜렷하게 나타나는 것을 확인했습니다.
- **통합 혼동 행렬 (`plot_integrated_confusion_matrix`)**: `awake_study.csv`와 `bedtime.csv` 두 파일을 대상으로 통합 혼동 행렬 및 분류 보고서를 생성하여 최종 모델의 성능을 정량적으로 평가했습니다.
    - **결과**: Sleep 클래스에 대한 Precision 0.52, Recall 0.79, F1-score 0.63을, Awake 클래스에 대한 Precision 0.56, Recall 0.27, F1-score 0.37을 기록했습니다. 전체 정확도는 0.53으로 나타났습니다. Awake 클래스에 대한 Recall이 상대적으로 낮은 것은 활동성이 낮은 Awake 구간을 Sleep으로 오분류하는 경향이 있음을 시사합니다. 이는 활동성 기반 라벨링의 한계일 수 있으며, 실제 PSG 데이터를 통한 라벨링과의 차이에서 기인할 수 있습니다.
- **성과**: Min-Max Scaling을 통해 모델의 예측 분포를 효과적으로 확장하여 시각적으로 더 명확한 분류 결과를 얻었습니다. 하지만, 활동성 기반의 단순한 0/1 라벨링은 실제 수면 단계의 복잡성을 완전히 반영하지 못하여, 특히 Awake 클래스에서 Recall 값이 낮은 한계점을 보였습니다.

## 7. 결론 및 향후 계획

본 프로젝트를 통해 Muse 뇌파 데이터를 활용한 수면-각성 탐지 모델을 성공적으로 개발하고, 여러 단계의 전처리, 모델 구축, 학습 및 미세 조정을 거쳤습니다. 특히 EDF 데이터를 `.fif`와 `.npy`로 변환하는 과정에서 데이터 처리의 효율성을 크게 높였으며, 어텐션 메커니즘을 포함한 딥러닝 모델을 통해 시계열 뇌파 데이터의 특징을 효과적으로 학습했습니다.

활동성 기반의 라벨링을 통해 실제 환경과 유사한 데이터를 활용한 미세 조정을 시도했으나, 이 라벨링 방식이 실제 수면 단계와 완벽하게 일치하지 않아 Awake 클래스에서 Recall 성능에 한계가 있음을 확인했습니다. 이는 활동성이 낮은 각성 상태(조용한 각성)를 수면으로 오분류할 가능성을 내포합니다.

**향후 계획**:
1.  **다중 수면 단계 분류 확장**: 현재 이진 분류(수면/각성) 모델을 N1, N2, N3, REM 등 더 세분화된 수면 단계 분류 모델로 확장하는 것을 고려할 수 있습니다. 이를 위해서는 보다 정교한 수면 단계 라벨링 데이터셋과 다중 클래스 분류에 적합한 모델 구조 및 손실 함수가 필요합니다.
2.  **전처리 개선**: EEG 신호 외에 EOG(안구 운동) 및 EMG(근전도) 신호를 추가적으로 활용하여 모델의 분류 정확도를 높일 수 있습니다. 또한, 주파수 도메인 특징(예: FFT, 웨이블릿 변환)을 추출하여 모델 입력으로 사용하는 방안도 탐색할 수 있습니다.
3.  **데이터셋 확장 및 균형**: 더 많은 피험자 데이터와 다양한 수면 패턴을 포함하는 데이터셋을 확보하고, 각 수면 단계의 샘플 수를 균형 있게 조절하여 모델의 일반화 성능을 향상시킬 수 있습니다.
4.  **모델 최적화**: 경량화된 모델 아키텍처 (예: MobileNet 계열, EfficientNet)를 적용하여 임베디드 장치나 모바일 환경에서도 효율적으로 동작하는 모델을 개발할 수 있습니다.
5.  **실시간 추론 고려**: 모델의 추론 속도를 최적화하여 실시간 또는 준실시간 수면-각성 모니터링 시스템에 적용 가능성을 탐색할 수 있습니다.

---

# 전처리 노트북 보고서 원문

## 작업 보고서: 수면 원격 측정 데이터 분석 및 전처리

본 보고서는 수면 원격 측정(sleep telemetry) 데이터 분석 및 전처리를 위해 Colab 환경에서 수행된 일련의 작업을 요약합니다. 목표는 원시 EDF 형식의 수면다원검사(PSG) 데이터와 수면 단계 주석(hypnogram) 데이터를 기계 학습 모델 학습에 적합한 형태로 변환하는 것입니다.

### 1. 환경 설정 및 라이브러리 설치

데이터 접근 및 처리를 위한 기본 환경이 설정되었습니다.

*   **Google Drive 마운트**: `/content/drive` 경로에 Google Drive를 마운트하여 데이터 파일에 접근했습니다. 이는 `google.colab import drive` 및 `drive.mount('/content/drive')` 코드를 통해 이루어졌습니다.
*   **필수 라이브러리 설치**: 수면 데이터 처리에 특화된 두 가지 주요 라이브러리가 설치되었습니다.
    *   `pyedflib`: EDF(European Data Format) 파일을 읽고, 쓰고, 조작하는 데 사용되는 라이브러리입니다. EDF 파일은 EEG, ECG 등 생체 신호 데이터를 저장하는 표준 형식입니다.
    *   `mne`: 뇌 자기장측정(MEG) 또는 뇌전도(EEG) 데이터를 분석하고 시각화하는 데 널리 사용되는 강력한 라이브러리입니다. 복잡한 신호 처리 및 수면 단계 주석 처리에 활용되었습니다.

### 2. 초기 Muse 2 데이터 예측 시연 (사전 모델 활용)

이 섹션은 Muse 2 장치에서 얻은 EEG 데이터와 사전 훈련된 모델을 사용하여 졸음 상태를 예측하는 초기 시연을 보여줍니다. 이는 주된 데이터 전처리 파이프라인과는 별개로, 유사한 유형의 EEG 데이터에 대한 모델 적용 가능성을 탐색한 부분입니다.

*   **사용 모델**: `/content/drive/MyDrive/sleep-telemetry/sleep_small_model/model_4.keras` 경로의 Keras 모델이 로드되었습니다. 이 모델은 특정 EEG 신호를 기반으로 수면 단계를 분류하도록 훈련된 것으로 보입니다.
*   **데이터 로딩 및 전처리**: `/content/EEG_recording_2026-03-06-04.22.17.csv` 파일에서 Muse 2 EEG 데이터(`AF7`, `AF8` 채널)를 로드했습니다. 이 데이터는 `mne.io.RawArray` 객체로 변환되었으며, `EEG Fpz-Cz`, `EEG Pz-Oz` 채널로 매핑되었습니다.
    *   **리샘플링**: 모델의 입력 요구 사항에 맞춰 데이터를 256Hz에서 100Hz로 리샘플링했습니다. 이는 30초 길이의 데이터 창을 3000개 샘플로 유지하기 위함입니다.
    *   **필터링**: 0.5Hz ~ 30Hz 대역통과 필터를 적용하여 EEG 신호에서 관심 주파수 대역만 추출하고 노이즈를 제거했습니다. (`fir_design='firwin'` 사용).
*   **졸음 상태 예측**: 30초 (`window_size = 3000` 샘플) 슬라이딩 윈도우를 1초 (`step_size = 100` 샘플) 단위로 이동하며 데이터를 추출했습니다. 각 윈도우에 대해 정규화(`mean_val`, `std_val` 사용) 후 모델 예측을 수행하고, 예측된 수면 단계 확률을 기반으로 '졸음 점수'를 계산하여 실시간 졸음 상태를 시연했습니다.
    *   **졸음 점수 산정**: `prob[1]` (Stage 1), `prob[2]` (Stage 2), `prob[3]` (Stage 3), `prob[4]` (Stage 4), `prob[5]` (Stage R)의 확률을 합산하여 100을 곱한 값으로 졸음 점수를 정의했습니다. 점수가 70을 초과하면 '졸음 위험!'으로 분류했습니다.
*   **근거**: 이 시연은 실제 수면다원검사 데이터 전처리 전에, 특정 장비(Muse 2)로 획득한 EEG 데이터를 활용하여 학습된 모델의 즉각적인 적용 가능성을 보여주기 위함입니다.

### 3. 수면 원격 측정 데이터 전처리 및 `.fif`, `.npy` 변환

`sleep-edf-database-expanded-1.0.0`에서 제공하는 EDF 형식의 수면다원검사(PSG) 데이터와 Hypnogram 데이터를 기계 학습 모델 학습에 사용하기 위해 전처리 및 효율적인 형식으로 변환하는 과정입니다.

*   **데이터 경로 정의**: `base_dir`을 `/content/drive/MyDrive/sleep-telemetry/sleep-edf-database-expanded-1.0.0/sleep-telemetry`로 설정하고, `first_path_list`와 `second_path_list`에 정의된 고유 식별자를 사용하여 PSG(`-PSG.edf`) 파일과 Hypnogram(`-Hypnogram.edf`) 파일의 전체 경로를 생성했습니다. `paired_paths`를 통해 각 환자별 PSG와 Hypnogram 파일이 올바르게 매칭되었음을 확인했습니다.
*   **EDF 파일 정보 확인**: `pyedflib.EdfReader`를 사용하여 각 PSG 파일의 헤더 정보와 채널별 상세 정보(채널명, 샘플 수, 단위)를 출력하여 데이터 구조를 파악했습니다.
    *   **주요 채널 확인**: `EEG Fpz-Cz`, `EEG Pz-Oz`, `EOG horizontal`, `EMG submental`, `Marker` 등의 채널이 포함되어 있음을 확인했습니다. 이 중 `EEG Fpz-Cz`와 `EEG Pz-Oz`는 뇌파 분석의 핵심 채널입니다.
*   **뇌파 채널 선택 및 `.fif` 변환**: `mne` 라이브러리를 사용하여 EDF 파일을 읽고 특정 뇌파 채널만 선택한 후 MNE의 자체 이진 형식인 `.fif` 파일로 저장했습니다.
    *   `raw = mne.io.read_raw_edf(psg_file, preload=True)`: EDF 파일을 로드합니다.
    *   `ann = mne.read_annotations(hyp_file)` 및 `raw.set_annotations(ann)`: Hypnogram 파일을 읽어 수면 단계 주석을 `raw` 데이터에 적용합니다.
    *   `raw.pick_channels(["EEG Fpz-Cz", "EEG Pz-Oz"])`: 5개의 원본 채널 중 뇌파 분석에 필요한 `EEG Fpz-Cz`와 `EEG Pz-Oz` 두 채널만 선택했습니다. 이는 모델 입력 채널 수를 줄여 계산 효율성을 높이고, 수면 단계 분류에 가장 관련성이 높은 신호에 집중하기 위함입니다.
    *   `raw.save(out_file, overwrite=True)`: 전처리된 데이터를 `/content/drive/MyDrive/sleep-telemetry/sleep_fif` 경로에 `.fif` 형식으로 저장했습니다.
    *   **근거**: `.fif` 형식은 MNE에 최적화된 파일 형식으로, 빠른 데이터 로딩과 효율적인 저장을 가능하게 합니다. 또한, 불필요한 채널을 미리 제거하여 후속 처리 단계를 단순화합니다.
*   **데이터 병합 및 `.npy` 변환 (수면 단계 분류를 위한 에포크 생성)**:
    *   `stage_map` 정의: EDF Hypnogram 파일에 포함된 텍스트 기반 수면 단계 주석(`Sleep stage W`, `Sleep stage 1` 등)을 수치형 레이블(0, 1, 2, 3, 4, 5)로 매핑하는 딕셔너리를 정의했습니다.
    *   `time_step_sec = 30`: 30초를 하나의 수면 에포크(epoch) 단위로 설정했습니다. 이는 수면 단계 분류의 표준 관행입니다.
    *   **데이터 및 레이블 추출**: 각 `.fif` 파일에서 원시 데이터를 로드하고(`mne.io.read_raw_fif`), Hypnogram 파일에서 읽은 주석을 `sfreq` (샘플링 주파수)를 사용하여 각 샘플에 해당하는 수면 단계 레이블(`labels`) 배열로 변환했습니다.
    *   **30초 에포크 생성**: 전체 데이터를 30초 간격으로 슬라이딩하여 `X` (데이터 윈도우)와 `Y` (해당 윈도우의 가장 빈번한 수면 단계 레이블)를 생성했습니다. `np.bincount(labels[i:i+time_step]).argmax()`를 사용하여 30초 내에서 가장 많이 나타나는 수면 단계를 해당 에포크의 대표 레이블로 지정했습니다.
    *   **`.npy` 파일 저장**: 생성된 `X`와 `Y` 데이터를 `/content/drive/MyDrive/sleep-telemetry/merge_dataset_tel` 경로에 NumPy의 바이너리 형식인 `.npy` 파일로 저장했습니다.
    *   **근거**: `.npy` 형식은 텍스트 기반 형식(예: CSV)에 비해 **데이터 누수 방지** 및 **효율성** 측면에서 큰 이점을 가집니다. 바이너리 형식으로 데이터를 연속적으로 저장하여 입출력(I/O) 병목 현상을 줄이고, 데이터 로딩 및 저장 속도를 압도적으로 향상시킵니다. 이는 대용량 시계열 데이터를 처리하고 기계 학습 모델의 입력으로 활용하는 데 매우 적합한 방식입니다. 30초 에포크는 수면 생리학 연구 및 수면 단계 분류 모델에서 표준적으로 사용되는 시간 단위입니다.
