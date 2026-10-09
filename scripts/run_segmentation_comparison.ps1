$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$cacheDir = Join-Path $projectRoot 'models\huggingface'
$grounding = Join-Path $projectRoot 'outputs\grounding\grounding_dino_tiny_adversarial.jsonl'
$cropOutput = Join-Path $projectRoot 'outputs\controls\grounding_dino_tiny_bbox_crop_adversarial.jsonl'
$env:PYTHONPATH = $projectRoot
$env:HF_HOME = $cacheDir
$env:TOKENIZERS_PARALLELISM = 'false'

& $python -m src.inference.run_grounding_control `
  --intervention bbox_crop `
  --pope-file (Join-Path $projectRoot 'data\raw\pope\coco_pope_adversarial.json') `
  --image-dir (Join-Path $projectRoot 'data\raw\coco\val2014') `
  --grounding $grounding `
  --output $cropOutput `
  --setting adversarial `
  --split-file (Join-Path $projectRoot 'data\splits\adversarial_image_split_seed42.json') `
  --subset test `
  --model-id (Join-Path $projectRoot 'models\Qwen2.5-VL-3B-Instruct') `
  --cache-dir $cacheDir `
  --retry-errors
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python -m src.evaluation.evaluate_bbox_guided `
  --baseline (Join-Path $projectRoot 'outputs\baseline\adversarial.jsonl') `
  --guided $cropOutput `
  --split (Join-Path $projectRoot 'data\splits\adversarial_image_split_seed42.json') `
  --subset test `
  --output (Join-Path $projectRoot 'outputs\tables\grounding_dino_tiny_bbox_crop_metrics.json')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& (Join-Path $PSScriptRoot 'run_sam2_guided_adversarial.ps1') `
  -GroundingStem grounding_dino_tiny
exit $LASTEXITCODE
