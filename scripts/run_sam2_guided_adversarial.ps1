param(
  [Parameter(Mandatory = $true)]
  [string]$GroundingStem
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$cacheDir = Join-Path $projectRoot 'models\huggingface'
$logDir = Join-Path $projectRoot 'logs'
$grounding = Join-Path $projectRoot "outputs\grounding\${GroundingStem}_adversarial.jsonl"
$output = Join-Path $projectRoot "outputs\segmentation_guided\${GroundingStem}_sam2_adversarial.jsonl"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$env:PYTHONPATH = $projectRoot
$env:HF_HOME = $cacheDir
$env:TOKENIZERS_PARALLELISM = 'false'

if (-not (Test-Path -LiteralPath $grounding)) {
  throw "Grounding file does not exist: $grounding"
}

$ErrorActionPreference = 'Continue'
& $python -m src.inference.run_sam2_guided `
  --pope-file (Join-Path $projectRoot 'data\raw\pope\coco_pope_adversarial.json') `
  --image-dir (Join-Path $projectRoot 'data\raw\coco\val2014') `
  --grounding $grounding `
  --output $output `
  --setting adversarial `
  --split-file (Join-Path $projectRoot 'data\splits\adversarial_image_split_seed42.json') `
  --subset test `
  --qwen-model-id (Join-Path $projectRoot 'models\Qwen2.5-VL-3B-Instruct') `
  --sam-model-id 'facebook/sam2.1-hiera-tiny' `
  --cache-dir $cacheDir `
  --retry-errors 2>&1 | Tee-Object -FilePath (Join-Path $logDir "${GroundingStem}_sam2_guided.log")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python -m src.evaluation.evaluate_bbox_guided `
  --baseline (Join-Path $projectRoot 'outputs\baseline\adversarial.jsonl') `
  --guided $output `
  --split (Join-Path $projectRoot 'data\splits\adversarial_image_split_seed42.json') `
  --subset test `
  --output (Join-Path $projectRoot "outputs\tables\${GroundingStem}_sam2_guided_metrics.json")
