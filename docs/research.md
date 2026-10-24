# Research 배경 및 수면 점수 계산식

## 영감을 받은 논문
- **Deep learning-based sleep stage classification with cardiorespiratory and body movement activities in individuals with suspected sleep disorders** (time series data preprocessing)
  https://www.nature.com/articles/s41598-023-45020-7
  - 파라미터: channel 3, unit 128, dropout 0.2 고정, epoch 5000
  - 대규모 데이터 기반. 본 프로젝트는 **소수 데이터로 유의미한 결과**를 1차 목표로 설정. 데이터 불균형 파악이 선행 과제.

## 정상 수면 기준 근거
| 항목 | 근거 |
|---|---|
| 정상 수면 조절기전 기반 건강한 수면 습관 | https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003102504 |
| 총 수면 시간 권장 (성인 7~9h, 학생 8~10h) | National Sleep Foundation, https://pubmed.ncbi.nlm.nih.gov/29073398/ |
| 수면 단계 비율 (N3 15~25%, REM 20~25%) | AASM Manual, https://jcsm.aasm.org/doi/10.5664/jcsm.1928 |
| 무호흡증 판정 (AHI) | https://www.sleepfoundation.org/sleep-apnea/ahi |

## 수면 점수 계산식 (설계안, 미구현)

### 조건 1. 총 수면시간 — 나이대 기반 (30점)
1. 신생아 14~17시간
2. 영아 12~15시간
3. 유아 11~14시간
4. 미취학 아동 10~13시간
5. 학령기 아동 9~11시간
6. 청소년 8~10시간
7. 성인 7~9시간
8. 노년층 7~8시간

### 조건 2. 수면 단계 비율 (35 + 35점)
1. N3+N4 (논문에서는 합쳐서 N3) 비중 15~25%
2. REM 비중 20~25%

### 조건 3. 수면 효율 / 방해 감점 (최대 -15)
- 수면 도중 W(0) 발생 개수만큼 -1, 최대 15개까지 → 최대 -15

### 총점
```
수면시간(30) + 단계 비중(35 + 35) - 방해 감점(최대 15) = 100 - 감점(최대 15)
```
