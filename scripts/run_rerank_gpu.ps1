$ErrorActionPreference = "Stop"

python -m src.rerank_lgbm `
  --offline-db outputs\baseline\offline_baseline_work.db `
  --submit-db outputs\baseline\submit_baseline_work.db `
  --output-dir outputs\rerank `
  --candidate-topk 50 `
  --topk-grid 5,10,20,30 `
  --offline-eval-mode per_user `
  --device auto `
  --neg-pos-ratio 30 `
  --num-boost-round 300 `
  --predict-batch-size 2000000

Write-Host "Rerank GPU pipeline finished."

