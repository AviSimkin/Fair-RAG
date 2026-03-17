# run_recreation.ps1
#
# Orchestrates the full experiment recreation for the Fair-RAG paper.
# Reproduces the EE-D interval / EU difference analysis from:
#   "Towards Fair RAG: On the Impact of Fair Ranking in Retrieval-Augmented Generation"
#
# Setup:
#   - Generator : flanT5Small  (google/flan-t5-small)
#   - LaMP task : 4  (news headline generation)
#   - Retrievers: bm25, gold
#   - Alphas    : 1, 2, 4, 8  (PL temperature; higher = fairer)
#   - N_samples : 100 (PL samples per query)
#   - Top-k     : 5
#
# Usage:
#   # Full experiment (833 queries, takes several hours on CPU)
#   .\run_recreation.ps1
#
#   # Quick smoke-test (10 queries, ~5 min on CPU)
#   .\run_recreation.ps1 -QuickTest
#
# After completion, run the analysis:
#   .venv\Scripts\python.exe analyze_results.py

param(
    [switch]$QuickTest,
    [string]$PythonExe = ".venv\Scripts\python.exe",
    [int]$MaxQueries,
    [int]$NumSamples,
    [int]$TopK = 5
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$GENERATOR   = "flanT5Small"
$LAMP_NUM    = 4
$ALPHAS      = @(1, 2, 4, 8)
$N_SAMPLES   = 100
$K           = $TopK

if ($QuickTest) {
    $MAX_QUERIES = 2
    $N_SAMPLES   = 2
    Write-Host "=== QUICK TEST MODE: $MAX_QUERIES queries, $N_SAMPLES PL samples ===" -ForegroundColor Yellow
} else {
    $MAX_QUERIES = $null   # all 833 queries
    Write-Host "=== FULL EXPERIMENT: all queries, $N_SAMPLES PL samples ===" -ForegroundColor Cyan
}

if ($PSBoundParameters.ContainsKey('MaxQueries')) {
    $MAX_QUERIES = $MaxQueries
}
if ($PSBoundParameters.ContainsKey('NumSamples')) {
    $N_SAMPLES = $NumSamples
}

Write-Host "Config => generator=$GENERATOR lamp=$LAMP_NUM k=$K n_samples=$N_SAMPLES max_queries=$MAX_QUERIES" -ForegroundColor DarkCyan

function Run-Experiment {
    param([string]$Retriever, [int]$Alpha, [int]$MaxQ)
    $resultDir = "experiment_results\$GENERATOR\lamp$LAMP_NUM\$Retriever"
    $resultFile = "$resultDir\alpha_$Alpha.json"
    if (Test-Path $resultFile) {
        Write-Host "  [SKIP] $resultFile already exists" -ForegroundColor DarkGray
        return
    }
    New-Item -ItemType Directory -Force -Path $resultDir | Out-Null
    $args_list = @(
        "experiment.py",
        "--generator_name", $GENERATOR,
        "--lamp_num", $LAMP_NUM,
        "--retriever_name", $Retriever,
        "--alpha", $Alpha,
        "--k", $K,
        "--n_samples", $N_SAMPLES,
        "--remove_temp_files"
    )
    if ($MaxQ) {
        $args_list += @("--max_queries", $MaxQ)
    }
    Write-Host "  Running: experiment.py --retriever $Retriever --alpha $Alpha" -ForegroundColor White
    & $PythonExe @args_list
    if ($LASTEXITCODE -ne 0) { throw "experiment.py failed for retriever=$Retriever alpha=$Alpha" }
}

function Run-NormalizeEU {
    param([string]$Retriever, [int]$Alpha)
    $normalizedFile = "experiment_results\$GENERATOR\lamp$LAMP_NUM\$Retriever\alpha_${Alpha}_normalized.json"
    if (Test-Path $normalizedFile) {
        Write-Host "  [SKIP] $normalizedFile already exists" -ForegroundColor DarkGray
        return
    }
    Write-Host "  Normalizing EU: retriever=$Retriever alpha=$Alpha" -ForegroundColor White
    & $PythonExe normalize_eu.py `
        --generator_name $GENERATOR `
        --lamp_num $LAMP_NUM `
        --retriever_name $Retriever `
        --alpha $Alpha
    if ($LASTEXITCODE -ne 0) { throw "normalize_eu.py failed for retriever=$Retriever alpha=$Alpha" }
}

# ------------------------------------------------------------------
# Step 1: Gold retriever (alpha=8 only, used as EU upper bound for norm)
# ------------------------------------------------------------------
Write-Host "`n--- Step 1: Gold retriever (alpha=8) ---" -ForegroundColor Cyan
Run-Experiment -Retriever "gold" -Alpha 8 -MaxQ $MAX_QUERIES

# ------------------------------------------------------------------
# Step 2: BM25 retriever for each alpha
# ------------------------------------------------------------------
Write-Host "`n--- Step 2: BM25 retriever (alpha = $($ALPHAS -join ', ')) ---" -ForegroundColor Cyan
foreach ($alpha in $ALPHAS) {
    Run-Experiment -Retriever "bm25" -Alpha $alpha -MaxQ $MAX_QUERIES
}

# ------------------------------------------------------------------
# Step 3: Normalize EU for each BM25 alpha
# ------------------------------------------------------------------
Write-Host "`n--- Step 3: Normalize EU ---" -ForegroundColor Cyan
foreach ($alpha in $ALPHAS) {
    Run-NormalizeEU -Retriever "bm25" -Alpha $alpha
}

# ------------------------------------------------------------------
# Step 4: Analysis
# ------------------------------------------------------------------
Write-Host "`n--- Step 4: Analyze results ---" -ForegroundColor Cyan
& $PythonExe analyze_results.py `
    --generator_name $GENERATOR `
    --lamp_num $LAMP_NUM `
    --retriever_name "bm25"
if ($LASTEXITCODE -ne 0) { throw "analyze_results.py failed" }

Write-Host "`n=== Recreation complete. Results in experiment_results\$GENERATOR\lamp$LAMP_NUM\ ===" -ForegroundColor Green
