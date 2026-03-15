param(
  [string]$BaselineDir = "outputs\baseline_v2",
  [string]$OutRoot = "outputs",
  [string]$RoundId = "",
  [int]$RerankCandidateTopk = 50,
  [string]$CandidateTopkGrid = "40,50,60",
  [string]$GlobalTopnGrid = "30000,40000,50000,60000,70000,80000",
  [string]$ZombiePenaltyGrid = "0.6,0.8,1.0",
  [string]$StrongBonusGrid = "0,0.05,0.1",
  [string]$LoyaltyWeightGrid = "0,0.1,0.2",
  [string]$CvrWeightGrid = "0,0.05,0.1",
  [string]$RecencyWeightGrid = "0,0.05",
  [string]$Device = "auto",
  [int]$PredictBatchSize = 200000,
  [int]$TrainReadChunksize = 500000,
  [int]$SubmitReadChunksize = 500000,
  [int]$TrainMaxPositives = 200000,
  [int]$TrainMaxNegatives = 600000,
  [double]$TrainHardNegRatio = 0.85,
  [int]$FallbackTrainMaxNegatives = 450000,
  [double]$RoundHoursCap = 4.0,
  [int]$MaxCandidates = 0,
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

function Copy-ParamMap([hashtable]$src) {
  $dst = [ordered]@{}
  foreach ($k in $src.Keys) {
    $dst[$k] = [double]$src[$k]
  }
  return $dst
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
  $lineCount = $lines.Count
  if ($lineCount -eq 0) {
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
    line_count = [int]$lineCount
    dedup_ok = $true
    no_header = $true
    tab_two_cols = $true
  }
}

$candidateGrid = Parse-IntGrid $CandidateTopkGrid
$topnGrid = Parse-IntGrid $GlobalTopnGrid
$zombieGrid = Parse-DoubleGrid $ZombiePenaltyGrid
$strongGrid = Parse-DoubleGrid $StrongBonusGrid
$loyaltyGrid = Parse-DoubleGrid $LoyaltyWeightGrid
$cvrGrid = Parse-DoubleGrid $CvrWeightGrid
$recencyGrid = Parse-DoubleGrid $RecencyWeightGrid

$offlineDb = Join-Path $BaselineDir "offline_baseline_work.db"
$submitDb = Join-Path $BaselineDir "submit_baseline_work.db"
if (!(Test-Path $offlineDb)) { throw "offline db not found: $offlineDb" }
if (!(Test-Path $submitDb)) { throw "submit db not found: $submitDb" }

if ([string]::IsNullOrWhiteSpace($RoundId)) {
  $RoundId = (Get-Date).ToString("yyyyMMdd_HHmmss")
}
$roundDir = Join-Path $OutRoot ("14d_push_round_" + $RoundId)
New-Item -ItemType Directory -Path $roundDir -Force | Out-Null

$maxCandidatesArg = @()
if ($MaxCandidates -gt 0) {
  $maxCandidatesArg = @("--max-candidates", "$MaxCandidates")
}

$baselineFeatureCount = 0
$baselineRerankSummary = "outputs\rerank_v2\rerank_summary.json"
if (Test-Path $baselineRerankSummary) {
  $baseObj = Get-Content -Path $baselineRerankSummary -Encoding UTF8 -Raw | ConvertFrom-Json
  if ($baseObj.feature_count) {
    $baselineFeatureCount = [int]$baseObj.feature_count
  }
}

$modelDefs = @(
  [ordered]@{ name = "P1"; num_leaves = 63; min_leaf = 50; learning_rate = 0.05; num_boost_round = 400 },
  [ordered]@{ name = "P2"; num_leaves = 127; min_leaf = 100; learning_rate = 0.03; num_boost_round = 600 }
)

function Invoke-RerankRun(
  [string]$ModelName,
  [int]$NumLeaves,
  [int]$MinLeaf,
  [double]$LearningRate,
  [int]$NumBoostRound,
  [int]$TrainNegatives,
  [bool]$DisableEnhanced,
  [string]$Tag
) {
  $outDir = Join-Path $roundDir ("rerank_" + $ModelName + "_" + $Tag)
  New-Item -ItemType Directory -Path $outDir -Force | Out-Null

  $args = @(
    "-m", "src.rerank_lgbm",
    "--offline-db", $offlineDb,
    "--submit-db", $submitDb,
    "--output-dir", $outDir,
    "--candidate-topk", "$RerankCandidateTopk",
    "--offline-eval-mode", "global",
    "--topk-grid", $GlobalTopnGrid,
    "--train-streaming",
    "--train-max-positives", "$TrainMaxPositives",
    "--train-max-negatives", "$TrainNegatives",
    "--train-hard-neg-ratio", "$TrainHardNegRatio",
    "--train-read-chunksize", "$TrainReadChunksize",
    "--submit-read-chunksize", "$SubmitReadChunksize",
    "--predict-batch-size", "$PredictBatchSize",
    "--num-boost-round", "$NumBoostRound",
    "--learning-rate", "$LearningRate",
    "--num-leaves", "$NumLeaves",
    "--min-data-in-leaf", "$MinLeaf",
    "--device", $Device
  )
  if ($DisableEnhanced) {
    $args += "--disable-enhanced-features"
  }
  if ($maxCandidatesArg.Count -gt 0) {
    $args += $maxCandidatesArg
  }

  $start = Get-Date
  & python @args
  if ($LASTEXITCODE -ne 0) {
    throw "rerank failed for $ModelName ($Tag)"
  }
  $elapsed = (Get-Date) - $start

  $summaryPath = Join-Path $outDir "rerank_summary.json"
  if (!(Test-Path $summaryPath)) {
    throw "rerank summary missing: $summaryPath"
  }
  $summary = Get-Content -Path $summaryPath -Encoding UTF8 -Raw | ConvertFrom-Json
  $bestMetric = Get-BestMetricFromRerank $summary

  return [ordered]@{
    model_name = $ModelName
    tag = $Tag
    output_dir = $outDir
    summary_path = $summaryPath
    model_file = (Join-Path $outDir "lgbm_rerank_model.txt")
    disable_enhanced_features = [bool]$DisableEnhanced
    train_max_negatives = [int]$TrainNegatives
    elapsed_seconds = [int][Math]::Round($elapsed.TotalSeconds)
    summary = $summary
    best_topn = $bestMetric.best_topn
    best_f1 = $bestMetric.best_f1
  }
}

function Invoke-FusionRun(
  [string]$RunName,
  [string]$ModelFile,
  [double]$ZombiePenalty,
  [double]$StrongBonus,
  [double]$LoyaltyWeight,
  [double]$CvrWeight,
  [double]$RecencyWeight,
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
    "--candidate-topk-grid", $CandidateTopkGrid,
    "--global-topn-grid", $GlobalTopnGrid,
    "--predict-batch-size", "$PredictBatchSize",
    "--read-chunksize", "$SubmitReadChunksize",
    "--metric-protocol", "global_topn",
    "--lookback-days", "14",
    "--zombie-penalty", "$ZombiePenalty",
    "--strong-bonus", "$StrongBonus",
    "--loyalty-weight", "$LoyaltyWeight",
    "--cvr-weight", "$CvrWeight",
    "--recency-weight", "$RecencyWeight"
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
    submit_file = $summary.submit_file
    submit_rows = $summary.submit_rows
    params = [ordered]@{
      zombie_penalty = [double]$ZombiePenalty
      strong_bonus = [double]$StrongBonus
      loyalty_weight = [double]$LoyaltyWeight
      cvr_weight = [double]$CvrWeight
      recency_weight = [double]$RecencyWeight
    }
    summary = $summary
  }
}

$modelResults = @()
$roundStart = Get-Date

foreach ($model in $modelDefs) {
  $name = [string]$model.name
  $modelStart = Get-Date

  $fallbackByTime = $false
  $fallbackByFeature = $false
  $oomOrFailure = $false

  $rerankRun = $null
  try {
    $rerankRun = Invoke-RerankRun `
      -ModelName $name `
      -NumLeaves ([int]$model.num_leaves) `
      -MinLeaf ([int]$model.min_leaf) `
      -LearningRate ([double]$model.learning_rate) `
      -NumBoostRound ([int]$model.num_boost_round) `
      -TrainNegatives $TrainMaxNegatives `
      -DisableEnhanced $false `
      -Tag "main"
  }
  catch {
    $oomOrFailure = $true
    Write-Warning "[$name] first rerank failed, retry with lower negatives. $_"
    $rerankRun = Invoke-RerankRun `
      -ModelName $name `
      -NumLeaves ([int]$model.num_leaves) `
      -MinLeaf ([int]$model.min_leaf) `
      -LearningRate ([double]$model.learning_rate) `
      -NumBoostRound ([int]$model.num_boost_round) `
      -TrainNegatives $FallbackTrainMaxNegatives `
      -DisableEnhanced $false `
      -Tag "retry_lowneg"
  }

  if ($rerankRun.summary.metric_protocol -ne "global_topn") {
    throw "[$name] rerank metric_protocol mismatch: $($rerankRun.summary.metric_protocol)"
  }

  $featureCount = [int]$rerankRun.summary.feature_count
  $featureNonFinite = [int]$rerankRun.summary.feature_non_finite_count
  if ($featureNonFinite -gt 0 -or ($baselineFeatureCount -gt 0 -and $featureCount -le $baselineFeatureCount)) {
    $fallbackByFeature = $true
    Write-Warning "[$name] feature anomaly detected, fallback to non-enhanced feature set."
    $rerankRun = Invoke-RerankRun `
      -ModelName $name `
      -NumLeaves ([int]$model.num_leaves) `
      -MinLeaf ([int]$model.min_leaf) `
      -LearningRate ([double]$model.learning_rate) `
      -NumBoostRound ([int]$model.num_boost_round) `
      -TrainNegatives ([int]$rerankRun.train_max_negatives) `
      -DisableEnhanced $true `
      -Tag "fallback_basic"
  }

  $elapsedHoursSinceModelStart = ((Get-Date) - $modelStart).TotalHours
  if ($elapsedHoursSinceModelStart -gt $RoundHoursCap -and [int]$rerankRun.train_max_negatives -gt $FallbackTrainMaxNegatives) {
    $fallbackByTime = $true
    Write-Warning "[$name] stage elapsed > $RoundHoursCap h, rerun with lower negatives ($FallbackTrainMaxNegatives)."
    $rerankRun = Invoke-RerankRun `
      -ModelName $name `
      -NumLeaves ([int]$model.num_leaves) `
      -MinLeaf ([int]$model.min_leaf) `
      -LearningRate ([double]$model.learning_rate) `
      -NumBoostRound ([int]$model.num_boost_round) `
      -TrainNegatives $FallbackTrainMaxNegatives `
      -DisableEnhanced ([bool]$rerankRun.disable_enhanced_features) `
      -Tag "fallback_time"
  }

  $bestParams = [ordered]@{
    zombie_penalty = 1.0
    strong_bonus = 0.0
    loyalty_weight = 0.0
    cvr_weight = 0.0
    recency_weight = 0.0
  }
  $fusionRuns = @()

  $baseFusion = Invoke-FusionRun `
    -RunName ($name + "_base") `
    -ModelFile $rerankRun.model_file `
    -ZombiePenalty $bestParams.zombie_penalty `
    -StrongBonus $bestParams.strong_bonus `
    -LoyaltyWeight $bestParams.loyalty_weight `
    -CvrWeight $bestParams.cvr_weight `
    -RecencyWeight $bestParams.recency_weight `
    -DoSubmit $false
  $fusionRuns += $baseFusion
  $bestFusion = $baseFusion

  $scanOrder = @(
    @{ key = "zombie_penalty"; grid = $zombieGrid },
    @{ key = "strong_bonus"; grid = $strongGrid },
    @{ key = "loyalty_weight"; grid = $loyaltyGrid },
    @{ key = "cvr_weight"; grid = $cvrGrid },
    @{ key = "recency_weight"; grid = $recencyGrid }
  )

  foreach ($entry in $scanOrder) {
    $key = [string]$entry.key
    $grid = $entry.grid
    foreach ($val in $grid) {
      $valD = [double]$val
      if ([double]$bestParams[$key] -eq $valD) { continue }

      $probe = Copy-ParamMap $bestParams
      $probe[$key] = $valD
      $runName = "{0}_{1}_{2}" -f $name, $key, ($valD.ToString().Replace('.', 'p'))

      $run = Invoke-FusionRun `
        -RunName $runName `
        -ModelFile $rerankRun.model_file `
        -ZombiePenalty ([double]$probe.zombie_penalty) `
        -StrongBonus ([double]$probe.strong_bonus) `
        -LoyaltyWeight ([double]$probe.loyalty_weight) `
        -CvrWeight ([double]$probe.cvr_weight) `
        -RecencyWeight ([double]$probe.recency_weight) `
        -DoSubmit $false

      $fusionRuns += $run
      if ([double]$run.best_f1 -gt [double]$bestFusion.best_f1) {
        $bestFusion = $run
        $bestParams = $probe
      }
    }
  }

  $finalFusion = Invoke-FusionRun `
    -RunName ($name + "_final") `
    -ModelFile $rerankRun.model_file `
    -ZombiePenalty ([double]$bestParams.zombie_penalty) `
    -StrongBonus ([double]$bestParams.strong_bonus) `
    -LoyaltyWeight ([double]$bestParams.loyalty_weight) `
    -CvrWeight ([double]$bestParams.cvr_weight) `
    -RecencyWeight ([double]$bestParams.recency_weight) `
    -DoSubmit $true

  $modelElapsed = (Get-Date) - $modelStart

  $modelResults += [ordered]@{
    model_name = $name
    model_config = $model
    rerank = [ordered]@{
      summary_path = $rerankRun.summary_path
      output_dir = $rerankRun.output_dir
      best_topn = $rerankRun.best_topn
      best_f1 = $rerankRun.best_f1
      metric_protocol = $rerankRun.summary.metric_protocol
      feature_count = [int]$rerankRun.summary.feature_count
      feature_non_finite_count = [int]$rerankRun.summary.feature_non_finite_count
      enhanced_features_enabled = [bool]$rerankRun.summary.enhanced_features_enabled
      train_max_negatives = [int]$rerankRun.train_max_negatives
      elapsed_seconds = [int]$rerankRun.elapsed_seconds
    }
    fusion = [ordered]@{
      runs_count = [int]$fusionRuns.Count
      best_offline_f1_in_scan = [double]$bestFusion.best_f1
      tuned_params = $bestParams
      final_summary_path = $finalFusion.summary_path
      final_f1 = [double]$finalFusion.best_f1
      final_best_topn = [int]$finalFusion.best_topn
      final_best_candidate_topk = [int]$finalFusion.best_candidate_topk
      submit_file = $finalFusion.submit_file
      submit_rows = $finalFusion.submit_rows
      elapsed_seconds = [int]$finalFusion.elapsed_seconds
    }
    status = [ordered]@{
      oom_or_failure = [bool]$oomOrFailure
      fallback_by_time = [bool]$fallbackByTime
      fallback_by_feature = [bool]$fallbackByFeature
      elapsed_seconds = [int][Math]::Round($modelElapsed.TotalSeconds)
    }
  }
}

$rankedModels = $modelResults | Sort-Object -Property { [double]$_.fusion.final_f1 } -Descending
$bestModel = $rankedModels[0]
$backupModel = $null
if ($rankedModels.Count -gt 1) {
  $backupModel = $rankedModels[1]
}

$bestF1 = [double]$bestModel.fusion.final_f1
$promotion = "continue_tune"
if ($bestF1 -ge 0.088) {
  $promotion = "sprint_submission"
}
elseif ($bestF1 -ge 0.080) {
  $promotion = "main_submission"
}
elseif ($bestF1 -ge 0.078) {
  $promotion = "next_round_tune"
}

$artifact = [ordered]@{}
if ($bestModel.fusion.submit_file -and (Test-Path $bestModel.fusion.submit_file)) {
  if ($bestF1 -ge 0.080) {
    $mainSubmit = Join-Path $roundDir "submit_main.tsv"
    Copy-Item -Path $bestModel.fusion.submit_file -Destination $mainSubmit -Force
    $artifact.main_submit = Test-SubmitFile $mainSubmit
  }
  if ($bestF1 -ge 0.088) {
    $sprintSubmit = Join-Path $roundDir "submit_sprint.tsv"
    Copy-Item -Path $bestModel.fusion.submit_file -Destination $sprintSubmit -Force
    $artifact.sprint_submit = Test-SubmitFile $sprintSubmit
  }
}
if ($backupModel -and $backupModel.fusion.submit_file -and (Test-Path $backupModel.fusion.submit_file)) {
  $backupSubmit = Join-Path $roundDir "submit_backup.tsv"
  Copy-Item -Path $backupModel.fusion.submit_file -Destination $backupSubmit -Force
  $artifact.backup_submit = Test-SubmitFile $backupSubmit
}

$roundElapsed = (Get-Date) - $roundStart

$summary = [ordered]@{
  run_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
  round_id = $RoundId
  round_dir = $roundDir
  baseline_dir = $BaselineDir
  offline_db = $offlineDb
  submit_db = $submitDb
  metric_protocol = "global_topn"
  lookback_days = 14
  config = [ordered]@{
    rerank_candidate_topk = $RerankCandidateTopk
    candidate_topk_grid = $candidateGrid
    global_topn_grid = $topnGrid
    zombie_penalty_grid = $zombieGrid
    strong_bonus_grid = $strongGrid
    loyalty_weight_grid = $loyaltyGrid
    cvr_weight_grid = $cvrGrid
    recency_weight_grid = $recencyGrid
    train_max_positives = $TrainMaxPositives
    train_max_negatives = $TrainMaxNegatives
    fallback_train_max_negatives = $FallbackTrainMaxNegatives
    train_hard_neg_ratio = $TrainHardNegRatio
    round_hours_cap = $RoundHoursCap
    predict_batch_size = $PredictBatchSize
    train_read_chunksize = $TrainReadChunksize
    submit_read_chunksize = $SubmitReadChunksize
    max_candidates = $MaxCandidates
    device = $Device
    skip_submit = [bool]$SkipSubmit
  }
  baseline_feature_count = [int]$baselineFeatureCount
  models = $modelResults
  best = [ordered]@{
    model_name = $bestModel.model_name
    final_f1 = [double]$bestModel.fusion.final_f1
    final_best_topn = [int]$bestModel.fusion.final_best_topn
    final_best_candidate_topk = [int]$bestModel.fusion.final_best_candidate_topk
    rerank_summary = $bestModel.rerank.summary_path
    fusion_summary = $bestModel.fusion.final_summary_path
  }
  promotion = $promotion
  artifacts = $artifact
  elapsed_seconds = [int][Math]::Round($roundElapsed.TotalSeconds)
}

$summaryPath = Join-Path $roundDir "summary.json"
$summary | ConvertTo-Json -Depth 10 | Set-Content -Path $summaryPath -Encoding UTF8
Write-Host "[OK] 14d push round summary: $summaryPath"
Write-Host "[BEST] model=$($summary.best.model_name) f1=$($summary.best.final_f1) promotion=$promotion"
