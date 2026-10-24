# Setup

```bash
conda create -n pjh_skin python=3.10 -y   # 이미 있으면 생략
bash setup/setup_env.sh
```

- GPU: `tensorflow[and-cuda]` 가 CUDA 12 런타임을 pip 로 함께 설치. 드라이버 ≥ 525 필요.
- `model_4.keras` 는 Keras 3.10 포맷. Keras 2 / TF < 2.16 에서는 로드 불가.
- 실행 시 `conda activate pjh_skin` 후 repo root 에서 `python scripts/0X_*.py`.
