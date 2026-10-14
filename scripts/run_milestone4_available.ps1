# Run the Milestone 4 experiments already supported by the repository.
# This does NOT run ColBERT, SPLADE, or MedCPT; those need separate validated
# runners for the full datasets. The final grid command will fail if cells remain.
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$Python = (Get-Command python).Source
$DataRoot = "E:\ir_project\data"

Write-Host "Repository: $RepoRoot"
Write-Host "Python: $Python"
Write-Host "Data root: $DataRoot"

# Verify the expected local files before spending time on model runs.
$required = @(
    "$DataRoot\limit-50k\corpus.jsonl",
    "$DataRoot\limit-50k\queries.jsonl",
    "$DataRoot\limit-50k\qrels\test.tsv",
    "$DataRoot\scidocs\corpus.jsonl",
    "$DataRoot\scidocs\queries.jsonl",
    "$DataRoot\scidocs\qrels\test.tsv",
    "$DataRoot\CSFCube\abstracts-csfcube-preds.jsonl",
    "$DataRoot\CSFCube\test-pid2pool-csfcube.json"
)
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Required data file not found: $path"
    }
}

# Full-budget lexical reference for all three datasets.
foreach ($dataset in @("CSFCube", "LIMIT-50k", "SciDocs")) {
    & $Python "scripts/run_bm25.py" --dataset $dataset
    if ($LASTEXITCODE -ne 0) { throw "BM25 failed for $dataset (exit $LASTEXITCODE)" }
}

# Contriever corpus embeddings, corpus-only PCA, and full rankings for BEIR data.
& $Python "scripts/run_contriever_beir.py"
if ($LASTEXITCODE -ne 0) { throw "Contriever retrieval failed (exit $LASTEXITCODE)" }

foreach ($dataset in @("LIMIT-50k", "SciDocs")) {
    if ($dataset -eq "LIMIT-50k") {
        $DatasetDir = "$DataRoot\limit-50k"
    } else {
        $DatasetDir = "$DataRoot\scidocs"
    }
    & $Python "scripts/evaluate_contriever_beir.py" --dataset $dataset --data-dir $DatasetDir --ranking-dir "$DataRoot\contriever_beir" --output-dir "results"
    if ($LASTEXITCODE -ne 0) { throw "Contriever evaluation failed for $dataset (exit $LASTEXITCODE)" }
}

# Re-evaluate CSFCube using the corrected NDCG@10 evaluator. Requires prior
# CSFCube Contriever rankings for dimensions 768/384/192/96/48.
& $Python "scripts/evaluate_contriever.py" --data-dir "$DataRoot\CSFCube" --ranking-dir "$DataRoot\CSFCube\contriever_rankings" --output "results\csfcube_contriever.json"
if ($LASTEXITCODE -ne 0) { throw "CSFCube Contriever evaluation failed (exit $LASTEXITCODE)" }

# The BM25 constant-control option is explicit. Missing ColBERT/SPLADE/MedCPT
# scores remain missing; the grid builder will report them rather than guess.
& $Python "scripts/build_full_grid.py" --results-dir "results" --output "results\full_grid_results.csv" --bm25-control-all-budgets
if ($LASTEXITCODE -ne 0) {
    Write-Warning "The grid is still incomplete. Use the MISSING lines above to identify the remaining experiments."
    exit $LASTEXITCODE
}

Write-Host "Milestone 4 available experiment set completed and grid written."
