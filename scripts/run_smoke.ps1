$ErrorActionPreference = "Stop"

python -m src.audit_data `
  --sample-chunks 1 `
  --max-rows-per-file 200000 `
  --chunksize 200000

python -m src.run_baseline `
  --mode offline `
  --eval-date 2014-12-18 `
  --max-rows-per-file 200000 `
  --chunksize 200000 `
  --topk 20 `
  --topk-grid 10,20,30

python -m src.run_baseline `
  --mode submit `
  --predict-date 2014-12-19 `
  --max-rows-per-file 200000 `
  --chunksize 200000 `
  --topk 20

Write-Host "Smoke pipeline finished."
