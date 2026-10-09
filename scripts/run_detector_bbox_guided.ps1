$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$cacheDir = Join-Path $projectRoot 'models\huggingface'
$logDir = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$env:PYTHONPATH = $projectRoot
$env:HF_HOME = $cacheDir
$env:TOKENIZERS_PARALLELISM = 'false'

$detectors = @('grounding_dino_tiny', 'grounding_dino_base')
$ErrorActionPreference = 'Continue'
foreach ($detector in $detectors) {
  $guidedOutput = Join-Path $projectRoot "outputs\bbox_guided\${detector}_adversarial_bbox.jsonl"
  & $python -m src.inference.run_grounding_control `
    --intervention bbox_overlay `
    --pope-file (Join-Path $projectRoot 'data\raw\pope\coco_pope_adversarial.json') `
    --image-dir (Join-Path $projectRoot 'data\raw\coco\val2014') `
    --grounding (Join-Path $projectRoot "outputs\grounding\${detector}_adversarial.jsonl") `
    --output $guidedOutput `
    --setting adversarial `
    --split-file (Join-Path $projectRoot 'data\splits\adversarial_image_split_seed42.json') `
    --subset test `
    --model-id (Join-Path $projectRoot 'models\Qwen2.5-VL-3B-Instruct') `
    --cache-dir $cacheDir `
    --retry-errors 2>&1 | Tee-Object -FilePath (Join-Path $logDir "${detector}_bbox_guided.log")
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

  & $python -m src.evaluation.evaluate_bbox_guided `
    --baseline (Join-Path $projectRoot 'outputs\baseline\adversarial.jsonl') `
    --guided $guidedOutput `
    --split (Join-Path $projectRoot 'data\splits\adversarial_image_split_seed42.json') `
    --subset test `
    --output (Join-Path $projectRoot "outputs\tables\${detector}_bbox_guided_metrics.json")
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
