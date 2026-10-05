$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$env:PYTHONPATH = $projectRoot

& $python -m src.evaluation.evaluate_grounded_gate `
  --baseline (Join-Path $projectRoot 'outputs\baseline\adversarial.jsonl') `
  --grounding (Join-Path $projectRoot 'outputs\grounding\adversarial.jsonl') `
  --split-output (Join-Path $projectRoot 'data\splits\adversarial_image_split_seed42.json') `
  --detector-output (Join-Path $projectRoot 'outputs\detector_only\adversarial_test.jsonl') `
  --gated-output (Join-Path $projectRoot 'outputs\gated\adversarial_test.jsonl') `
  --metrics-output (Join-Path $projectRoot 'outputs\tables\grounded_adversarial_metrics.json') `
  --dev-ratio 0.2 `
  --seed 42 `
  --threshold-min 0.0 `
  --threshold-max 0.5 `
  --threshold-step 0.01
