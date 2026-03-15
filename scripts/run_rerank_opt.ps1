$ErrorActionPreference = "Stop"

python -m src.optimize_rerank_lgbm `
  --offline-db outputs\baseline\offline_baseline_work.db `
  --submit-db outputs\baseline\submit_baseline_work.db `
  --output-dir outputs\rerank_opt `
  --candidate-topk 50 `
  --topk-grid 5,10,20,30 `
  --optuna-trials 25 `
  --optimize-max-candidates 4000000 `
  --device auto `
  --predict-batch-size 2000000 `
  --log-eval 100

Write-Host "Rerank optimization pipeline finished."
