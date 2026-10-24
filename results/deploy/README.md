# Deploy-style prediction (scripts/08_stream_predict.py)

- `SC4001E0-PSG.predict.json` : PDF 설정 그대로 (band-pass + 후처리). full κ 0.18 — 필터 불일치로 붕괴
- `../deploy_nofilter/SC4001E0-PSG.predict.json` : `--no-filter --no-postprocess`. full κ 0.76
