# Qwen baseline 실행 가이드

## 준비 상태 확인

프로젝트 루트에서 다음을 실행한다.

```powershell
cd D:\VIG_Grounded_Hallucination_Pilot
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Adversarial 본 실행

```powershell
cd D:\VIG_Grounded_Hallucination_Pilot
powershell -ExecutionPolicy Bypass -File .\scripts\run_adversarial_baseline.ps1
```

Qwen2.5-VL-3B 가중치는 `models/Qwen2.5-VL-3B-Instruct`에 미리 다운로드한다. 실행 중에는 이 로컬 경로를 사용하며, 질문마다 결과가 `outputs/baseline/adversarial.jsonl`에 즉시 추가된다.

실행이 중단되어도 동일 명령을 다시 실행하면 완료된 `question_id`를 건너뛴다. 오류 행만 다시 처리하려면 launch script에 이미 포함된 `--retry-errors`가 사용된다.

## 진행 상태 확인

```powershell
(Get-Content .\outputs\baseline\adversarial.jsonl | Measure-Object -Line).Lines
Get-Content .\logs\qwen_adversarial.log -Tail 20
```

## 완료 결과

- Raw predictions: `outputs/baseline/adversarial.jsonl`
- Metrics: `outputs/tables/baseline_adversarial_metrics.json`
- Execution log: `logs/qwen_adversarial.log`

별도의 20문항 smoke run은 만들지 않는다. 본 실행의 처음 10개 prediction을 출력하고 invalid 비율이 50%를 넘으면 자동 중단한다.
