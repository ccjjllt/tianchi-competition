param(
  [string]$BaselineDir = "outputs\baseline_30d_ab",
  [string]$Model14d = "outputs\rerank_v2\lgbm_rerank_model.txt",
  [string]$QuickOutDir = "outputs\fusion_stream_30d_quick",
  [string]$RerankOutDir = "outputs\rerank_30d_stream",
  [string]$FinalOutDir = "outputs\fusion_stream_30d_stream",
  [string]$CandidateTopkGrid = "20,30,40,50",
  [string]$GlobalTopnGrid = "30000,40000,50000,60000,70000,80000,90000,100000",
  [int]$RerankCandidateTopk = 50,
  [int]$TrainMaxPositives = 200000,
  [int]$TrainMaxNegatives = 400000,
  [double]$TrainHardNegRatio = 0.7,
  [int]$NumBoostRound = 300,
  [string]$Device = "auto",
  [int]$PredictBatchSize = 2000000,
  [int]$TrainReadChunksize = 1000000,
  [int]$SubmitReadChunksize = 1000000,
  [int]$MaxCandidates = 0,
  [switch]$SkipQuick,
  [switch]$SkipRetrain,
  [switch]$SkipFinalFusion
)

$ErrorActionPreference = "Stop"

$offlineDb = Join-Path $BaselineDir "offline_baseline_work.db"
$submitDb = Join-Path $BaselineDir "submit_baseline_work.db"
$baselineSummaryPath = Join-Path $BaselineDir "run_summary.json"

if (!(Test-Path $offlineDb)) { throw "offline db not found: $offlineDb" }
if (!(Test-Path $submitDb)) { throw "submit db not found: $submitDb" }
if (!(Test-Path $Model14d)) { throw "14d model not found: $Model14d" }

$maxCandidatesArg = @()
if ($MaxCandidates -gt 0) {
  $maxCandidatesArg = @("--max-candidates", "$MaxCandidates")
}

if (-not $SkipQuick) {
  python -m src.fusion_global_stream `
    --offline-db $offlineDb `
    --submit-db $submitDb `
    --model-file $Model14d `
    --output-dir $QuickOutDir `
    --candidate-topk-grid $CandidateTopkGrid `
    --global-topn-grid $GlobalTopnGrid `
    --predict-batch-size $PredictBatchSize `
    --read-chunksize $SubmitReadChunksize `
    --metric-protocol global_topn `
    --lookback-days 30 `
    --zombie-penalty 1.0 `
    --strong-bonus 0.0 `
    --loyalty-weight 0.0 `
    --cvr-weight 0.0 `
    --recency-weight 0.0 `
    @maxCandidatesArg
}

if (-not $SkipRetrain) {
  python -m src.rerank_lgbm `
    --offline-db $offlineDb `
    --submit-db $submitDb `
    --output-dir $RerankOutDir `
    --candidate-topk $RerankCandidateTopk `
    --offline-eval-mode global `
    --topk-grid $GlobalTopnGrid `
    --train-streaming `
    --train-max-positives $TrainMaxPositives `
    --train-max-negatives $TrainMaxNegatives `
    --train-hard-neg-ratio $TrainHardNegRatio `
    --train-read-chunksize $TrainReadChunksize `
    --submit-read-chunksize $SubmitReadChunksize `
    --predict-batch-size $PredictBatchSize `
    --num-boost-round $NumBoostRound `
    --device $Device `
    @maxCandidatesArg
}

if (-not $SkipFinalFusion) {
  $finalModel = Join-Path $RerankOutDir "lgbm_rerank_model.txt"
  if (!(Test-Path $finalModel)) {
    throw "retrained model not found for final fusion: $finalModel"
  }

  python -m src.fusion_global_stream `
    --offline-db $offlineDb `
    --submit-db $submitDb `
    --model-file $finalModel `
    --output-dir $FinalOutDir `
    --candidate-topk-grid $CandidateTopkGrid `
    --global-topn-grid $GlobalTopnGrid `
    --predict-batch-size $PredictBatchSize `
    --read-chunksize $SubmitReadChunksize `
    --metric-protocol global_topn `
    --lookback-days 30 `
    --zombie-penalty 1.0 `
    --strong-bonus 0.0 `
    --loyalty-weight 0.0 `
    --cvr-weight 0.0 `
    --recency-weight 0.0 `
    @maxCandidatesArg
}

$baseline = $null
if (Test-Path $baselineSummaryPath) {
  $baseline = Get-Content -Path $baselineSummaryPath -Encoding UTF8 -Raw | ConvertFrom-Json
}

$quick = $null
$quickSummaryPath = Join-Path $QuickOutDir "fusion_stream_summary.json"
if (Test-Path $quickSummaryPath) {
  $quick = Get-Content -Path $quickSummaryPath -Encoding UTF8 -Raw | ConvertFrom-Json
}

$rerank = $null
$rerankSummaryPath = Join-Path $RerankOutDir "rerank_summary.json"
if (Test-Path $rerankSummaryPath) {
  $rerank = Get-Content -Path $rerankSummaryPath -Encoding UTF8 -Raw | ConvertFrom-Json
}

$finalFusion = $null
$finalSummaryPath = Join-Path $FinalOutDir "fusion_stream_summary.json"
if (Test-Path $finalSummaryPath) {
  $finalFusion = Get-Content -Path $finalSummaryPath -Encoding UTF8 -Raw | ConvertFrom-Json
}

$pipelineSummary = [ordered]@{
  run_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  metric_protocol = "global_topn"
  lookback_days = if ($baseline -and $baseline.lookback_days) { [int]$baseline.lookback_days } else { 30 }
  baseline_dir = $BaselineDir
  offline_db = $offlineDb
  submit_db = $submitDb
  candidate_topk_grid = $CandidateTopkGrid
  global_topn_grid = $GlobalTopnGrid
  quick_baseline = if ($quick) {
    [ordered]@{
      output_dir = $QuickOutDir
      best_candidate_topk = $quick.best_candidate_topk
      best_topn = $quick.best_topn
      best_f1 = $quick.best_metrics.f1
      best_precision = $quick.best_metrics.precision
      best_recall = $quick.best_metrics.recall
    }
  } else { $null }
  retrain_rerank = if ($rerank) {
    [ordered]@{
      output_dir = $RerankOutDir
      candidate_topk = $rerank.candidate_topk
      best_topn = if ($rerank.offline_best_topn) { $rerank.offline_best_topn } else { $rerank.offline_best_topk }
      best_f1 = $null
      metric_protocol = $rerank.metric_protocol
      train_streaming = $rerank.train_streaming
      train_rows = $rerank.train_rows
    }
  } else { $null }
  final_fusion = if ($finalFusion) {
    [ordered]@{
      output_dir = $FinalOutDir
      best_candidate_topk = $finalFusion.best_candidate_topk
      best_topn = $finalFusion.best_topn
      best_f1 = $finalFusion.best_metrics.f1
      best_precision = $finalFusion.best_metrics.precision
      best_recall = $finalFusion.best_metrics.recall
    }
  } else { $null }
}

if ($rerank -and $rerank.offline_metrics_by_topk) {
  $bestKey = $null
  $bestF1 = -1.0
  foreach ($p in $rerank.offline_metrics_by_topk.PSObject.Properties) {
    $v = [double]$p.Value.f1
    if ($v -gt $bestF1) {
      $bestF1 = $v
      $bestKey = $p.Name
    }
  }
  if ($pipelineSummary.retrain_rerank) {
    $pipelineSummary.retrain_rerank.best_topn = [int]$bestKey
    $pipelineSummary.retrain_rerank.best_f1 = [double]$bestF1
  }
}

if (!(Test-Path $FinalOutDir)) { New-Item -ItemType Directory -Path $FinalOutDir -Force | Out-Null }
$summaryPath = Join-Path $FinalOutDir "pipeline_summary.json"
$pipelineSummary | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryPath -Encoding UTF8
Write-Host "[OK] 30d global pipeline summary: $summaryPath"

