# VLM 환각과 시각적 근거 분석

## 목적

이 프로젝트는 Qwen2.5-VL의 객체 존재 환각을 측정한다.

이 프로젝트는 COCO-POPE adversarial 데이터를 사용한다. 또한 OWLv2의 시각적 근거가 환각을 줄이는지 확인한다.

현재 방법은 Qwen의 가중치나 디코더를 변경하지 않는다. 사후 검증과 BBox 유도 재추론을 비교한다.

## 비교 방법

이 실험은 네 가지 방법을 비교한다.

1. **Qwen 단독**은 Qwen2.5-VL의 Yes 또는 No 답변을 사용한다.
2. **OWLv2 단독**은 OWLv2 탐지 점수를 Yes 또는 No로 변환한다.
3. **Qwen + OWLv2 gate**는 OWLv2 점수가 낮을 때 Qwen의 Yes를 No로 변경한다.
4. **BBox 유도 Qwen**은 OWLv2 후보 영역을 빨간 상자로 표시하고 Qwen이 다시 답하게 한다.

Gate는 다음 규칙을 사용한다.

```text
최종 Yes = Qwen Yes AND OWLv2 점수 >= threshold
```

Gate는 잘못된 Yes를 제거할 수 있다. Gate는 잘못된 No를 수정할 수 없다.

## 평가 방법

이 실험은 COCO-POPE adversarial 질문 3,000개를 사용한다. 이 질문들은 이미지 500장을 사용한다.

이 실험은 이미지 단위로 데이터를 분리한다. Dev에는 이미지 100장을 사용한다. Test에는 이미지 400장을 사용한다.

Dev는 threshold를 선택한다. Test는 최종 성능을 측정한다.

Test에는 질문 2,400개가 있다. Threshold 선택 기준은 balanced accuracy이다.

## Test 결과

표의 값은 백분율이다.

| 방법 | Accuracy | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Qwen 단독 | **86.54** | 93.98 | **78.08** | **85.30** | 5.00 |
| OWLv2 단독 | 85.79 | 85.18 | 86.67 | 85.91 | 15.08 |
| Qwen + OWLv2 gate | 86.08 | **95.01** | 76.17 | 84.55 | **4.00** |
| BBox 유도 Qwen | 84.00 | 85.11 | 82.42 | 83.74 | 14.42 |

Hard gate는 Qwen의 Yes 답변 35개를 No로 변경했다.

- Gate는 false positive 12개를 수정했다.
- Gate는 false negative 23개를 새로 만들었다.
- Gate는 accuracy를 0.46%p 낮췄다.

Gate는 false positive rate를 낮췄다. 그러나 gate는 recall과 전체 accuracy도 낮췄다.

BBox 유도 Qwen은 기존 답변 285개를 변경했다.

- BBox 유도는 오답 112개를 수정했다.
- BBox 유도는 정답 173개를 오답으로 변경했다.
- False negative 97개를 수정했다.
- True negative 128개를 false positive로 변경했다.
- Accuracy는 2.54%p 낮아졌다.

BBox 표시는 recall을 높였다. 그러나 빨간 상자가 객체 존재의 암시로 작동해 false positive가 크게 증가했다.

## 입력 대조 실험

BBox 효과를 분리하기 위해 같은 test 질문에서 입력 형태를 비교했다.

| 입력 | Accuracy | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| 기존 Qwen | **86.54** | 93.98 | 78.08 | 85.30 | 5.00 |
| 재확인 프롬프트만 | 86.17 | **94.56** | 76.75 | 84.73 | **4.42** |
| 임의 위치 BBox | 84.46 | 90.26 | 77.25 | 83.25 | 8.33 |
| OWLv2 BBox | 84.00 | 85.11 | 82.42 | 83.74 | 14.42 |
| 원본 + OWLv2 crop | 85.13 | 83.37 | **87.75** | **85.51** | 17.50 |

재확인 프롬프트만으로는 결과가 크게 변하지 않았다. 임의 BBox도 false positive를 늘렸다. 정확한 BBox와 crop은 recall을 높였지만 `Yes` 편향을 더 크게 만들었다.

자세한 내용은 [파일럿 결과](docs/results/adversarial_pilot.md)를 참고한다.
BBox 실험의 자세한 내용은 [BBox 유도 결과](docs/results/bbox_guided_adversarial.md)를 참고한다.
대조 실험의 자세한 내용은 [입력 대조 결과](docs/results/grounding_input_controls.md)를 참고한다.
디텍터 비교의 자세한 내용은 [디텍터 모델 비교 결과](docs/results/detector_model_comparison.md)를 참고한다.

## 디텍터 모델 비교

Qwen을 고정하고 OWLv2, Grounding DINO tiny, Grounding DINO base를 비교했다.

| 디텍터 | 디텍터 단독 Accuracy | BBox 유도 Qwen Accuracy | BBox 유도 Qwen F1 |
|---|---:|---:|---:|
| OWLv2 | **85.79** | 84.00 | 83.74 |
| Grounding DINO tiny | 81.88 | **85.00** | **84.80** |
| Grounding DINO base | 77.79 | 84.42 | 84.10 |

디텍터 단독 성능과 BBox 유도 성능의 순위는 일치하지 않았다. 따라서 더 좋은 객체 존재 판별 점수가 곧바로 더 좋은 VLM 추론을 만든다고 결론 내릴 수 없다.

## 실행 환경 준비

Python 3.11과 CUDA 지원 GPU를 사용한다.

가상환경을 만들고 필요한 패키지를 설치한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\environment\requirements.txt
```

POPE 파일을 다음 위치에 넣는다.

```text
data/raw/pope/
```

필요한 COCO 이미지를 다음 위치에 넣는다.

```text
data/raw/coco/val2014/
```

이 저장소는 모델 가중치를 포함하지 않는다. COCO 이미지, 전체 예측 결과, 로그도 포함하지 않는다.

## 실험 실행

저장소의 최상위 폴더에서 PowerShell을 연다.

Qwen baseline을 실행한다.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_adversarial_baseline.ps1
```

스크립트는 각 답변을 즉시 저장한다. 실행이 중단되면 같은 명령을 다시 실행한다.

OWLv2를 실행한다.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_owlv2_adversarial.ps1
```

기준선, detector, gate를 평가한다.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\evaluate_grounded_adversarial.ps1
```

BBox 유도 Qwen을 실행하고 평가한다.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_bbox_guided_adversarial.ps1
```

테스트 코드를 실행한다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

자세한 실행 방법은 [baseline 실행 가이드](docs/planning/run_baseline.md)를 참고한다.

## 주요 폴더

```text
configs/                 실험 설정
data/raw/                로컬 POPE 및 COCO 데이터
data/splits/             고정된 이미지 단위 split
docs/planning/           실험 계획 및 실행 가이드
docs/results/            실험 결과 보고서
environment/             Python 패키지 목록
experiments/pilot/       소규모 통합 테스트 기록
logs/                    로컬 실행 로그
models/                  로컬 모델 가중치
outputs/baseline/        Qwen 예측 결과
outputs/grounding/       OWLv2 점수와 box
outputs/detector_only/   OWLv2 단독 예측 결과
outputs/gated/           Gate 적용 결과
outputs/bbox_guided/     BBox 유도 Qwen 예측 결과
outputs/tables/          평가지표 파일
scripts/                 실행 스크립트
src/                     Python 소스 코드
tests/                   단위 테스트
```

## 실험 규칙

- Test 결과를 확인한 뒤에 threshold를 변경하지 않는다.
- 각 POPE 설정에 같은 이미지 단위 split을 사용한다.
- Qwen의 원본 답변과 OWLv2의 원본 점수를 보존한다.
- False positive 감소와 recall 손실을 함께 보고한다.
- 이 결과를 모든 VLM 환각에 적용하지 않는다.

POPE는 객체 존재 답변을 측정한다. POPE는 모든 개방형 환각을 측정하지 않는다.

## 다음 실험

디텍터 비교는 완료했다. 다음에는 Grounding DINO tiny의 같은 BBox를 사용해 두 입력을 비교한다.

1. 원본 이미지와 일반 BBox crop을 Qwen에 제공한다.
2. 같은 BBox로 SAM2 mask를 생성한다.
3. 원본 이미지와 mask crop을 Qwen에 제공한다.
4. 사각형 안의 배경을 제거한 픽셀 단위 근거가 도움이 되는지 확인한다.
