$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$logDir = Join-Path $projectRoot 'logs'
$outputDir = Join-Path $projectRoot 'outputs\controls'
$tableDir = Join-Path $projectRoot 'outputs\tables'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$env:PYTHONPATH = $projectRoot
$env:HF_HOME = Join-Path $projectRoot 'models\huggingface'
$env:TOKENIZERS_PARALLELISM = 'false'

$common = @(
  '--pope-file', (Join-Path $projectRoot 'data\raw\pope\coco_pope_adversarial.json'),
  '--image-dir', (Join-Path $projectRoot 'data\raw\coco\val2014'),
  '--grounding', (Join-Path $projectRoot 'outputs\grounding\adversarial.jsonl'),
  '--setting', 'adversarial',
  '--split-file', (Join-Path $projectRoot 'data\splits\adversarial_image_split_seed42.json'),
  '--subset', 'test',
  '--model-id', (Join-Path $projectRoot 'models\Qwen2.5-VL-3B-Instruct'),
  '--cache-dir', (Join-Path $projectRoot 'models\huggingface'),
  '--retry-errors'
)

$ErrorActionPreference = 'Continue'
foreach ($intervention in @('prompt_only', 'random_box', 'bbox_crop')) {
  $output = Join-Path $outputDir "adversarial_$intervention.jsonl"
  & $python -m src.inference.run_grounding_control `
    --intervention $intervention `
    --output $output `
    @common 2>&1 | Tee-Object -FilePath (Join-Path $logDir "control_$intervention.log")
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

  & $python -m src.evaluation.evaluate_bbox_guided `
    --baseline (Join-Path $projectRoot 'outputs\baseline\adversarial.jsonl') `
    --guided $output `
    --split (Join-Path $projectRoot 'data\splits\adversarial_image_split_seed42.json') `
    --subset test `
    --output (Join-Path $tableDir "control_${intervention}_adversarial_test_metrics.json")
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
