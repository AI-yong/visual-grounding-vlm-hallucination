$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$popeFile = Join-Path $projectRoot 'data\raw\pope\coco_pope_adversarial.json'
$imageDir = Join-Path $projectRoot 'data\raw\coco\val2014'
$output = Join-Path $projectRoot 'outputs\baseline\adversarial.jsonl'
$cacheDir = Join-Path $projectRoot 'models\huggingface'
$modelDir = Join-Path $projectRoot 'models\Qwen2.5-VL-3B-Instruct'
$logDir = Join-Path $projectRoot 'logs'

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$env:PYTHONPATH = $projectRoot
$env:HF_HOME = $cacheDir
$env:TOKENIZERS_PARALLELISM = 'false'

& $python -m src.inference.run_mllm `
  --pope-file $popeFile `
  --image-dir $imageDir `
  --output $output `
  --setting adversarial `
  --model-id $modelDir `
  --cache-dir $cacheDir `
  --retry-errors 2>&1 | Tee-Object -FilePath (Join-Path $logDir 'qwen_adversarial.log')

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

& $python -m src.evaluation.evaluate_baseline `
  --input $output `
  --output (Join-Path $projectRoot 'outputs\tables\baseline_adversarial_metrics.json')
