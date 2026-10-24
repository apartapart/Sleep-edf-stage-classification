# Results

`06_eval_cassette.py` → `cassette/` (per-night `*_true.npy`, `*_pred.npy`, `metrics.csv`)
`07_make_figures.py` → `figures/`

| 파일 | 내용 |
|---|---|
| `figures/metrics_by_protocol.png` | full / W-trimmed 6-class / 5-class × acc·mF1·κ, mean±std |
| `figures/cm_6class_trimmed.png` | 28밤 합산 confusion (counts + row-normalized), 6-class |
| `figures/cm_5class_trimmed.png` | 동일, N3+N4 병합 |
| `figures/cm_6class_full.png` | W 미제거 confusion (부풀림 비교용) |
| `figures/cm_SC4001E0_trimmed.png` | 단일 밤 confusion |
| `figures/per_class_prf_6class.png`, `_5class.png` | 클래스별 precision / recall / F1 |
| `figures/stage_proportion.png` | GT vs predicted 단계 분포 |
| `figures/kappa_per_night.png` | 밤별 κ, 평균선 |
| `figures/w_inflation.png` | W 비율 vs accuracy 부풀림 |
| `figures/hypnogram_SC4001E0.png` | 단일 밤 GT/Pred 오버레이 + mismatch 틱 |
| `figures/hypnogram_best_median_worst.png` | κ 최고/중앙/최저 밤 비교 |
| `figures/SC4001E0-PSG_*.png` | `05_predict` 노트북 스타일 출력 |
