$ErrorActionPreference = "Stop"

python -m src.audit_data

python -m src.run_baseline `
  --mode offline `
  --eval-date 2014-12-18 `
  --chunksize 1000000 `
  --lookback-days 10 `
  --topk 20 `
  --topk-grid 10,20,30,50

python -m src.run_baseline `
  --mode submit `
  --predict-date 2014-12-19 `
  --chunksize 1000000 `
  --lookback-days 10 `
  --topk 20

Write-Host "Full pipeline finished."
