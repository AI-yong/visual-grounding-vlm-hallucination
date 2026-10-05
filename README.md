# VIG Grounded Hallucination Pilot

Qwen2.5-VL의 object-existence hallucination을 POPE에서 측정하고, OWLv2의 external object grounding signal이 hallucinated `Yes`를 식별하고 완화하는 데 도움이 되는지 검증하는 미니 empirical project다.

## 핵심 비교

1. **Qwen-only**: Qwen2.5-VL의 원래 Yes/No 답변
2. **OWLv2-only**: grounding score threshold만으로 만든 Yes/No 답변
3. **Qwen + OWLv2 gate**: Qwen의 Yes 중 grounding score가 낮은 답변을 No로 변경

현재 핵심 방법은 decoding 내부 수정이 아니라 **inference-time post-hoc verification**이다.

## 현재 결과

COCO-POPE adversarial 3,000문항을 실행하고, 이미지 단위로 dev 100장과 test 400장을 분리했다. Threshold는 dev의 balanced accuracy로 선택하고 test 2,400문항에서 한 번 평가했다.

| Method | Accuracy | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Qwen-only | **86.54** | 93.98 | **78.08** | **85.30** | 5.00 |
| OWLv2-only | 85.79 | 85.18 | 86.67 | 85.91 | 15.08 |
| Qwen + OWLv2 gate | 86.08 | **95.01** | 76.17 | 84.55 | **4.00** |

Hard gate는 Qwen의 false positive 12개를 수정했지만 true positive 23개도 제거했다. 따라서 객체 환각과 FPR은 감소했으나 Recall과 전체 정확도는 하락했다. 이 결과는 외부 grounding signal 자체보다 **결합 정책과 detector false negative 관리가 중요함**을 보여주는 파일럿 결과다.

자세한 프로토콜과 해석은 `docs/results/adversarial_pilot.md`에 정리한다.

## 권장 실행 순서

1. POPE와 필요한 COCO 이미지를 `data/raw/`에 준비한다.
2. Qwen baseline을 adversarial부터 실행한다.
3. 별도 smoke run 대신 본 실행 첫 10개에서 invalid/error sanity guard를 적용한다.
4. OWLv2 grounding과 detector-only prediction을 생성한다.
5. image-level dev/test split으로 gate threshold를 선택·평가한다.
6. `evaluate_grounded_adversarial.ps1`로 세 조건을 test split에서 비교한다.
7. 결과표와 그림은 `outputs/`, 발표 자료는 `presentation/`에 저장한다.

## Adversarial baseline 실행

```powershell
cd D:\VIG_Grounded_Hallucination_Pilot
powershell -ExecutionPolicy Bypass -File .\scripts\run_adversarial_baseline.ps1
```

질문마다 결과를 즉시 저장하므로 중단되어도 같은 명령으로 이어서 실행할 수 있다. 자세한 설명은 `docs/planning/run_baseline.md`를 참고한다.

## OWLv2와 gate 실행

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_owlv2_adversarial.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\evaluate_grounded_adversarial.ps1
```

모델 가중치, COCO 이미지, raw outputs는 저장소에 포함하지 않는다. `data/splits/`의 고정 split과 문서에 기록한 요약 지표로 실험 프로토콜을 공개한다.

## 디렉터리 안내

```text
configs/                 실험 설정 YAML
data/raw/                원본 POPE/COCO 데이터
data/processed/          전처리·병합 데이터
data/splits/             image-level dev/test split
docs/planning/           실행 계획과 실행 가이드
environment/             requirements, GPU 및 CUDA 정보
experiments/pilot/       통합 검증 기록
logs/                    실행 로그
models/                  로컬 모델 가중치
outputs/baseline/        Qwen-only raw prediction
outputs/grounding/       OWLv2 score와 box
outputs/detector_only/   OWLv2-only prediction
outputs/gated/           Qwen + OWLv2 gate prediction
outputs/figures/         최종 그래프
outputs/tables/          최종 결과표
outputs/cases/           성공·실패 정성 사례
presentation/            면담 슬라이드, 그림, 대본
scripts/                 단계별 실행 entry point
src/                     재사용 가능한 Python 코드
tests/                   parser와 metric 단위 테스트
third_party/             POPE 등 외부 저장소
```

## 실험 원칙

- Test 결과를 보고 threshold를 수정하지 않는다.
- 세 POPE setting에 동일한 image-level split을 적용한다.
- Raw answer와 raw grounding score를 보존한다.
- FP 감소와 함께 introduced FN 및 Recall 손실을 보고한다.
- POPE 개선을 일반적인 open-ended hallucination 해결로 과장하지 않는다.

상세 계획은 `docs/planning/experiment_plan.md`를 참고한다.
