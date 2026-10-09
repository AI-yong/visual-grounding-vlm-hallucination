$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$env:PYTHONPATH = $projectRoot
$env:HF_HOME = Join-Path $projectRoot 'models\huggingface'
$env:TOKENIZERS_PARALLELISM = 'false'

& $python -m src.inference.run_bbox_guided `
  --pope-file (Join-Path $projectRoot 'data\raw\pope\coco_pope_adversarial.json') `
  --image-dir (Join-Path $projectRoot 'data\raw\coco\val2014') `
  --grounding (Join-Path $projectRoot 'outputs\grounding\adversarial.jsonl') `
  --output (Join-Path $projectRoot 'outputs\bbox_guided\adversarial_bbox_smoke.jsonl') `
  --setting adversarial `
  --model-id (Join-Path $projectRoot 'models\Qwen2.5-VL-3B-Instruct') `
  --cache-dir (Join-Path $projectRoot 'models\huggingface') `
  --limit 20 `
  --retry-errors

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python -m src.evaluation.evaluate_bbox_guided `
  --baseline (Join-Path $projectRoot 'outputs\baseline\adversarial.jsonl') `
  --guided (Join-Path $projectRoot 'outputs\bbox_guided\adversarial_bbox_smoke.jsonl') `
  --subset all `
  --output (Join-Path $projectRoot 'outputs\tables\bbox_guided_smoke_metrics.json')
