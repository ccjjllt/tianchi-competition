param(
  [string]$BaselineDir = "outputs\baseline_v2",
  [string]$OutRoot = "outputs",
  [string]$RoundId = "",
  [string]$Device = "auto",
  [int]$PredictBatchSize = 200000,
  [int]$ReadChunksize = 500000,
  [int]$TrainMaxPositives = 200000,
  [string]$CandidateTopkFullGrid = "40,50,60",
  [string]$GlobalTopnGrid = "30000,40000,50000,60000,70000,80000",
  [string]$CvrGrid = "0.36,0.40,0.44,0.46,0.48,0.50,0.54,0.58",
  [int]$CvrPatience = 3,
  [double]$SprintGoal = 0.09,
  [double]$SubmitMinF1 = 0.082,
  [int]$MaxCandidates = 0,
  [switch]$EnableEnhancedFeatures,
  [switch]$SkipSubmit
)

$ErrorActionPreference = "Stop"

function Parse-IntGrid([string]$value) {
  $items = @()
  foreach ($p in $value.Split(',')) {
    $v = $p.Trim()
    if ([string]::IsNullOrWhiteSpace($v)) { continue }
    $items += [int]$v
  }
  return ($items | Sort-Object -Unique)
}

function Parse-DoubleGrid([string]$value) {
  $items = @()
  foreach ($p in $value.Split(',')) {
    $v = $p.Trim()
    if ([string]::IsNullOrWhiteSpace($v)) { continue }
    $items += [double]$v
  }
  return ($items | Sort-Object -Unique)
}

function Get-BestMetricFromRerank($summaryObj) {
  $bestTopn = $null
  $bestF1 = -1.0
  if ($summaryObj.offline_metrics_by_topk) {
    foreach ($p in $summaryObj.offline_metrics_by_topk.PSObject.Properties) {
      $f1 = [double]$p.Value.f1
      if ($f1 -gt $bestF1) {
        $bestF1 = $f1
        $bestTopn = [int]$p.Name
      }
    }
  }
  return [ordered]@{
    best_topn = $bestTopn
    best_f1 = [double]$bestF1
  }
}

function Test-SubmitFile([string]$path) {
  if (!(Test-Path $path)) {
    throw "submit file not found: $path"
  }
  $lines = Get-Content -Path $path -Encoding UTF8
  if ($lines.Count -eq 0) {
    throw "submit file is empty: $path"
  }
  $first = $lines[0]
  $cols = $first -split "`t"
  if ($cols.Count -ne 2) {
    throw "submit first row is not 2-column tab-separated: $path"
  }
  if ($first -match '^user_id\titem_id$') {
    throw "submit should not contain header: $path"
  }
  $set = New-Object 'System.Collections.Generic.HashSet[string]'
  foreach ($line in $lines) {
    if (-not $set.Add($line)) {
      throw "submit has duplicate pair rows: $path"
    }
  }
  return [ordered]@{
    path = $path
    line_count = [int]$lines.Count
    dedup_ok = $true
    no_header = $true
    tab_two_cols = $true
  }
}

$candidateTopkFull = Parse-IntGrid $CandidateTopkFullGrid
$topnGrid = Parse-IntGrid $GlobalTopnGrid
$cvrVals = Parse-DoubleGrid $CvrGrid

$offlineDb = Join-Path $BaselineDir "offline_baseline_work.db"
$submitDb = Join-Path $BaselineDir "submit_baseline_work.db"
if (!(Test-Path $offlineDb)) { throw "offline db not found: $offlineDb" }
if (!(Test-Path $submitDb)) { throw "submit db not found: $submitDb" }

if ([string]::IsNullOrWhiteSpace($RoundId)) {
  $RoundId = (Get-Date).ToString("yyyyMMdd_HHmmss")
}
$roundDir = Join-Path $OutRoot ("14d_sprint09_round_" + $RoundId)
New-Item -ItemType Directory -Path $roundDir -Force | Out-Null

$maxCandidatesArg = @()
if ($MaxCandidates -gt 0) {
  $maxCandidatesArg = @("--max-candidates", "$MaxCandidates")
}

# Tuned around current best and then push model capacity upward.
$modelDefs = @(
  [ordered]@{
    name = "S1";
    candidate_topk = 50;
    num_leaves = 255;
    min_leaf = 120;
    learning_rate = 0.03;
    num_boost_round = 1200;
    early_stopping_rounds = 120;
    valid_ratio = 0.10;
    train_max_negatives = 300000;
    train_hard_neg_ratio = 0.0
  },
  [ordered]@{
    name = "S2";
    candidate_topk = 50;
    num_leaves = 383;
    min_leaf = 120;
    learning_rate = 0.025;
    num_boost_round = 1600;
    early_stopping_rounds = 140;
    valid_ratio = 0.10;
    train_max_negatives = 300000;
    train_hard_neg_ratio = 0.0
  },
  [ordered]@{
    name = "S3";
    candidate_topk = 50;
    num_leaves = 511;
    min_leaf = 150;
    learning_rate = 0.02;
    num_boost_round = 2200;
    early_stopping_rounds = 160;
    valid_ratio = 0.10;
    train_max_negatives = 300000;
    train_hard_neg_ratio = 0.0
  },
  [ordered]@{
    name = "S4";
    candidate_topk = 60;
    num_leaves = 383;
    min_leaf = 120;
    learning_rate = 0.025;
    num_boost_round = 1600;
    early_stopping_rounds = 140;
    valid_ratio = 0.10;
    train_max_negatives = 300000;
    train_hard_neg_ratio = 0.0
  }
)

function Invoke-RerankRun([hashtable]$cfg) {
  $outDir = Join-Path $roundDir ("rerank_" + $cfg.name)
  New-Item -ItemType Directory -Path $outDir -Force | Out-Null

  $args = @(
    "-m", "src.rerank_lgbm",
    "--offline-db", $offlineDb,
    "--submit-db", $submitDb,
    "--output-dir", $outDir,
    "--candidate-topk", "$($cfg.candidate_topk)",
    "--offline-eval-mode", "global",
    "--topk-grid", $GlobalTopnGrid,
    "--train-streaming",
    "--train-max-positives", "$TrainMaxPositives",
    "--train-max-negatives", "$($cfg.train_max_negatives)",
    "--train-hard-neg-ratio", "$($cfg.train_hard_neg_ratio)",
    "--train-read-chunksize", "$ReadChunksize",
    "--submit-read-chunksize", "$ReadChunksize",
    "--predict-batch-size", "$PredictBatchSize",
    "--num-boost-round", "$($cfg.num_boost_round)",
    "--learning-rate", "$($cfg.learning_rate)",
    "--num-leaves", "$($cfg.num_leaves)",
    "--min-data-in-leaf", "$($cfg.min_leaf)",
    "--valid-ratio", "$($cfg.valid_ratio)",
    "--early-stopping-rounds", "$($cfg.early_stopping_rounds)",
    "--device", $Device
  )
  if (-not $EnableEnhancedFeatures) {
    $args += "--disable-enhanced-features"
  }
  if ($maxCandidatesArg.Count -gt 0) {
    $args += $maxCandidatesArg
  }

  $start = Get-Date
  & python @args
  if ($LASTEXITCODE -ne 0) {
    throw "rerank failed for $($cfg.name)"
  }
  $elapsed = (Get-Date) - $start

  $summaryPath = Join-Path $outDir "rerank_summary.json"
  if (!(Test-Path $summaryPath)) {
    throw "rerank summary missing: $summaryPath"
  }
  $summary = Get-Content -Path $summaryPath -Encoding UTF8 -Raw | ConvertFrom-Json
  if ($summary.metric_protocol -ne "global_topn") {
    throw "rerank metric_protocol mismatch: $($summary.metric_protocol)"
  }
  $bestMetric = Get-BestMetricFromRerank $summary

  return [ordered]@{
    output_dir = $outDir
    summary_path = $summaryPath
    model_file = (Join-Path $outDir "lgbm_rerank_model.txt")
    elapsed_seconds = [int][Math]::Round($elapsed.TotalSeconds)
    best_topn = $bestMetric.best_topn
    best_f1 = $bestMetric.best_f1
    summary = $summary
  }
}

function Invoke-FusionRun(
  [string]$RunName,
  [string]$ModelFile,
  [string]$CandidateTopkGridArg,
  [double]$CvrWeight,
  [bool]$DoSubmit
) {
  $outDir = Join-Path $roundDir ("fusion_" + $RunName)
  New-Item -ItemType Directory -Path $outDir -Force | Out-Null

  $args = @(
    "-m", "src.fusion_global_stream",
    "--offline-db", $offlineDb,
    "--submit-db", $submitDb,
    "--model-file", $ModelFile,
    "--output-dir", $outDir,
    "--candidate-topk-grid", $CandidateTopkGridArg,
    "--global-topn-grid", $GlobalTopnGrid,
    "--predict-batch-size", "$PredictBatchSize",
    "--read-chunksize", "$ReadChunksize",
    "--metric-protocol", "global_topn",
    "--lookback-days", "14",
    "--zombie-penalty", "1.0",
    "--strong-bonus", "0",
    "--loyalty-weight", "0",
    "--cvr-weight", "$CvrWeight",
    "--recency-weight", "0"
  )
  if ($maxCandidatesArg.Count -gt 0) {
    $args += $maxCandidatesArg
  }
  if (-not $DoSubmit -or $SkipSubmit) {
    $args += "--skip-submit"
  }

  $start = Get-Date
  & python @args
  if ($LASTEXITCODE -ne 0) {
    throw "fusion failed: $RunName"
  }
  $elapsed = (Get-Date) - $start

  $summaryPath = Join-Path $outDir "fusion_stream_summary.json"
  if (!(Test-Path $summaryPath)) {
    throw "fusion summary missing: $summaryPath"
  }
  $summary = Get-Content -Path $summaryPath -Encoding UTF8 -Raw | ConvertFrom-Json
  return [ordered]@{
    run_name = $RunName
    output_dir = $outDir
    summary_path = $summaryPath
    elapsed_seconds = [int][Math]::Round($elapsed.TotalSeconds)
    best_f1 = [double]$summary.best_metrics.f1
    best_topn = [int]$summary.best_topn
    best_candidate_topk = [int]$summary.best_candidate_topk
    cvr_weight = [double]$CvrWeight
    submit_file = $summary.submit_file
    submit_rows = $summary.submit_rows
    summary = $summary
  }
}

$roundStart = Get-Date
$modelResults = @()
$globalBest = $null

foreach ($cfg in $modelDefs) {
  if ($globalBest -and [double]$globalBest.best_f1 -ge $SprintGoal) {
    Write-Host "[EARLY-STOP] sprint goal reached, stop training more models."
    break
  }

  $name = [string]$cfg.name
  Write-Host "[MODEL] start $name" -ForegroundColor Cyan
  $modelStart = Get-Date

  $rerank = Invoke-RerankRun -cfg $cfg
  Write-Host ("[MODEL] rerank done {0} best_f1={1}" -f $name, [double]$rerank.best_f1)

  $scanRuns = @()
  $bestScan = $null
  $noImprove = 0
  $scanIdx = 0
  foreach ($cvr in $cvrVals) {
    $scanIdx += 1
    $cvrStr = ([double]$cvr).ToString().Replace('.', 'p')
    $scanRun = Invoke-FusionRun `
      -RunName ($name + "_scan_cvr_" + $cvrStr) `
      -ModelFile $rerank.model_file `
      -CandidateTopkGridArg ([string]$cfg.candidate_topk) `
      -CvrWeight ([double]$cvr) `
      -DoSubmit $false
    $scanRuns += $scanRun

    if ($bestScan -eq $null -or [double]$scanRun.best_f1 -gt [double]$bestScan.best_f1) {
      $bestScan = $scanRun
      $noImprove = 0
    } else {
      $noImprove += 1
    }

    if ($scanIdx -ge 4 -and $noImprove -ge $CvrPatience) {
      Write-Host "[MODEL] cvr scan early-stop by patience." -ForegroundColor Yellow
      break
    }
  }

  if ($bestScan -eq $null) {
    throw "cvr scan produced no valid run for $name"
  }

  $fullRun = Invoke-FusionRun `
    -RunName ($name + "_fullgrid") `
    -ModelFile $rerank.model_file `
    -CandidateTopkGridArg $CandidateTopkFullGrid `
    -CvrWeight ([double]$bestScan.cvr_weight) `
    -DoSubmit $false

  $finalRun = $fullRun
  if (-not $SkipSubmit -and [double]$fullRun.best_f1 -ge [double]$SubmitMinF1) {
    $finalRun = Invoke-FusionRun `
      -RunName ($name + "_submit") `
      -ModelFile $rerank.model_file `
      -CandidateTopkGridArg $CandidateTopkFullGrid `
      -CvrWeight ([double]$bestScan.cvr_weight) `
      -DoSubmit $true
  }

  $modelElapsed = (Get-Date) - $modelStart
  $modelResult = [ordered]@{
    model_name = $name
    model_config = $cfg
    rerank = [ordered]@{
      summary_path = $rerank.summary_path
      output_dir = $rerank.output_dir
      best_f1 = [double]$rerank.best_f1
      best_topn = [int]$rerank.best_topn
      best_iteration = $rerank.summary.best_iteration
      feature_count = [int]$rerank.summary.feature_count
      enhanced_features_enabled = [bool]$rerank.summary.enhanced_features_enabled
      elapsed_seconds = [int]$rerank.elapsed_seconds
    }
    fusion = [ordered]@{
      cvr_scan_runs = [int]$scanRuns.Count
      best_cvr = [double]$bestScan.cvr_weight
      best_scan_f1 = [double]$bestScan.best_f1
      fullgrid_summary = $fullRun.summary_path
      fullgrid_f1 = [double]$fullRun.best_f1
      fullgrid_best_topn = [int]$fullRun.best_topn
      fullgrid_best_candidate_topk = [int]$fullRun.best_candidate_topk
      final_summary = $finalRun.summary_path
      final_f1 = [double]$finalRun.best_f1
      final_best_topn = [int]$finalRun.best_topn
      final_best_candidate_topk = [int]$finalRun.best_candidate_topk
      submit_file = $finalRun.submit_file
      submit_rows = $finalRun.submit_rows
    }
    elapsed_seconds = [int][Math]::Round($modelElapsed.TotalSeconds)
  }
  $modelResults += $modelResult

  if ($globalBest -eq $null -or [double]$modelResult.fusion.final_f1 -gt [double]$globalBest.best_f1) {
    $globalBest = [ordered]@{
      model_name = $name
      best_f1 = [double]$modelResult.fusion.final_f1
      best_topn = [int]$modelResult.fusion.final_best_topn
      best_candidate_topk = [int]$modelResult.fusion.final_best_candidate_topk
      best_cvr = [double]$modelResult.fusion.best_cvr
      model_file = $rerank.model_file
      fusion_summary = $modelResult.fusion.final_summary
      submit_file = $modelResult.fusion.submit_file
    }
    Write-Host ("[BEST] update model={0} f1={1}" -f $globalBest.model_name, $globalBest.best_f1) -ForegroundColor Green
  }
}

if ($globalBest -eq $null) {
  throw "no valid model result produced"
}

$artifacts = [ordered]@{}
if (-not $SkipSubmit -and $globalBest.submit_file -and (Test-Path $globalBest.submit_file)) {
  $bestSubmit = Join-Path $roundDir "submit_best.txt"
  Copy-Item -Path $globalBest.submit_file -Destination $bestSubmit -Force
  $artifacts.best_submit = Test-SubmitFile $bestSubmit
}

$roundElapsed = (Get-Date) - $roundStart
$promotion = "continue"
if ([double]$globalBest.best_f1 -ge [double]$SprintGoal) {
  $promotion = "goal_reached"
} elseif ([double]$globalBest.best_f1 -ge 0.088) {
  $promotion = "near_goal"
}

$summary = [ordered]@{
  run_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  round_id = $RoundId
  round_dir = $roundDir
  baseline_dir = $BaselineDir
  offline_db = $offlineDb
  submit_db = $submitDb
  metric_protocol = "global_topn"
  lookback_days = 14
  sprint_goal = [double]$SprintGoal
  config = [ordered]@{
    candidate_topk_full_grid = $candidateTopkFull
    global_topn_grid = $topnGrid
    cvr_grid = $cvrVals
    cvr_patience = [int]$CvrPatience
    predict_batch_size = [int]$PredictBatchSize
    read_chunksize = [int]$ReadChunksize
    train_max_positives = [int]$TrainMaxPositives
    max_candidates = [int]$MaxCandidates
    enable_enhanced_features = [bool]$EnableEnhancedFeatures
    submit_min_f1 = [double]$SubmitMinF1
    device = $Device
    skip_submit = [bool]$SkipSubmit
  }
  models = $modelResults
  best = $globalBest
  promotion = $promotion
  artifacts = $artifacts
  elapsed_seconds = [int][Math]::Round($roundElapsed.TotalSeconds)
}

$summaryPath = Join-Path $roundDir "summary.json"
$summary | ConvertTo-Json -Depth 12 | Set-Content -Path $summaryPath -Encoding UTF8
Write-Host "[OK] sprint summary: $summaryPath"
Write-Host ("[BEST] model={0} f1={1} topn={2} ctopk={3} cvr={4}" -f `
  $summary.best.model_name, $summary.best.best_f1, $summary.best.best_topn, $summary.best.best_candidate_topk, $summary.best.best_cvr)
