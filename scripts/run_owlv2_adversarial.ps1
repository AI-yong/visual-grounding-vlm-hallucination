$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$popeFile = Join-Path $projectRoot 'data\raw\pope\coco_pope_adversarial.json'
$imageDir = Join-Path $projectRoot 'data\raw\coco\val2014'
$output = Join-Path $projectRoot 'outputs\grounding\adversarial.jsonl'
$cacheDir = Join-Path $projectRoot 'models\huggingface'
$modelDir = Join-Path $projectRoot 'models\owlv2-base-patch16-ensemble'
$logDir = Join-Path $projectRoot 'logs'

New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$env:PYTHONPATH = $projectRoot
$env:HF_HOME = $cacheDir
$env:TOKENIZERS_PARALLELISM = 'false'

# Transformers writes progress information to stderr during normal execution.
# Do not let PowerShell convert those messages into terminating errors.
$ErrorActionPreference = 'Continue'
& $python -m src.grounding.run_owlv2 `
  --pope-file $popeFile `
  --image-dir $imageDir `
  --output $output `
  --setting adversarial `
  --model-id $modelDir `
  --cache-dir $cacheDir `
  --postprocess-threshold 0.001 `
  --retry-errors 2>&1 | Tee-Object -FilePath (Join-Path $logDir 'owlv2_adversarial.log')
$nativeExitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'

if ($nativeExitCode -ne 0) {
  exit $nativeExitCode
}
