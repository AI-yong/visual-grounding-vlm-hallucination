$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$cacheDir = Join-Path $projectRoot 'models\huggingface'
$logDir = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$env:PYTHONPATH = $projectRoot
$env:HF_HOME = $cacheDir
$env:TOKENIZERS_PARALLELISM = 'false'

$models = @(
  @{ Name = 'grounding_dino_tiny'; Id = 'IDEA-Research/grounding-dino-tiny' },
  @{ Name = 'grounding_dino_base'; Id = 'IDEA-Research/grounding-dino-base' }
)

$ErrorActionPreference = 'Continue'
foreach ($entry in $models) {
  $groundingOutput = Join-Path $projectRoot "outputs\grounding\$($entry.Name)_adversarial.jsonl"
  & $python -m src.grounding.run_grounding_dino `
    --pope-file (Join-Path $projectRoot 'data\raw\pope\coco_pope_adversarial.json') `
    --image-dir (Join-Path $projectRoot 'data\raw\coco\val2014') `
    --output $groundingOutput `
    --setting adversarial `
    --model-id $entry.Id `
    --cache-dir $cacheDir `
    --retry-errors 2>&1 | Tee-Object -FilePath (Join-Path $logDir "$($entry.Name)_adversarial.log")
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

  & $python -m src.evaluation.evaluate_grounded_gate `
    --baseline (Join-Path $projectRoot 'outputs\baseline\adversarial.jsonl') `
    --grounding $groundingOutput `
    --split-output (Join-Path $projectRoot 'data\splits\adversarial_image_split_seed42.json') `
    --detector-output (Join-Path $projectRoot "outputs\detector_only\$($entry.Name)_adversarial_test.jsonl") `
    --gated-output (Join-Path $projectRoot "outputs\gated\$($entry.Name)_adversarial_test.jsonl") `
    --metrics-output (Join-Path $projectRoot "outputs\tables\$($entry.Name)_adversarial_metrics.json") `
    --dev-ratio 0.2 `
    --seed 42 `
    --threshold-min 0.0 `
    --threshold-max 1.0 `
    --threshold-step 0.01
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
