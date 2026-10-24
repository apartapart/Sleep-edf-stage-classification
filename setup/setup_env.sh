#!/usr/bin/env bash
# conda env `pjh_skin` (python 3.10) 에 의존성 설치.
#   bash setup/setup_env.sh            # 기본 env 이름 pjh_skin
#   ENV_NAME=other bash setup/setup_env.sh
# model_4.keras 는 Keras 3.10 으로 저장됨 -> tensorflow>=2.18, keras>=3.10 필수.
set -euo pipefail
ENV_NAME="${ENV_NAME:-pjh_skin}"
PY="$(conda run -n "$ENV_NAME" which python)"
echo "[setup] env=$ENV_NAME python=$PY"

"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$(dirname "$0")/../requirements.txt"

"$PY" - <<'PYEOF'
import tensorflow as tf, keras, mne, pyedflib, sklearn, imblearn
print("tensorflow", tf.__version__, "| keras", keras.__version__, "| mne", mne.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
PYEOF
echo "[setup] done"
