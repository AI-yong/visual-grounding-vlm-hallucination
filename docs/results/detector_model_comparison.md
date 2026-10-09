# 디텍터 모델 비교 결과

## 질문

Qwen2.5-VL-3B를 고정하고 디텍터만 변경했을 때, 더 좋은 디텍터가 더 좋은 VLM 답변을 만드는지 확인했다.

비교한 디텍터는 OWLv2, Grounding DINO tiny, Grounding DINO base이다. 모든 결과는 같은 COCO-POPE adversarial test 질문 2,400개에서 측정했다. 각 디텍터의 threshold는 dev split의 balanced accuracy로 선택했다.

## 디텍터 단독 결과

표의 값은 백분율이다.

| 디텍터 | Threshold | Accuracy | Precision | Recall | F1 | FPR | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| OWLv2 | 0.11 | **85.79** | **85.18** | **86.67** | **85.91** | **15.08** | **91.55** |
| Grounding DINO tiny | 0.36 | 81.88 | 82.89 | 80.33 | 81.59 | 16.58 | 89.24 |
| Grounding DINO base | 0.38 | 77.79 | 77.77 | 77.83 | 77.80 | 22.25 | 85.70 |

이 설정에서는 큰 모델이 항상 더 좋지 않았다. Grounding DINO tiny가 base보다 모든 주요 지표에서 좋았다. 모델 크기 외에도 학습 특성, 짧은 객체 질의와의 적합성, 점수 보정이 결과에 영향을 줄 수 있다.

## Hard gate 결과

| 디텍터 | Accuracy | F1 | FPR |
|---|---:|---:|---:|
| OWLv2 gate | 86.08 | 84.55 | **4.00** |
| DINO tiny gate | **86.42** | 85.09 | 4.67 |
| DINO base gate | **86.42** | **85.11** | 4.83 |

세 gate 모두 Qwen 단독의 FPR 5.00%를 낮추거나 비슷하게 유지했다. 그러나 Yes를 No로만 바꾸는 구조이므로 recall 손실이 생겼다. 디텍터 단독 순위와 gate 순위도 일치하지 않았다. Qwen이 이미 대부분의 false positive를 적게 내기 때문에 디텍터 차이가 gate에서 크게 드러나지 않은 것으로 보인다.

## BBox를 Qwen에 제공한 결과

| 입력 박스 | Accuracy | Precision | Recall | F1 | FPR | 변경 / 수정 / 악화 |
|---|---:|---:|---:|---:|---:|---:|
| 박스 없음(Qwen 기준선) | **86.54** | **93.98** | 78.08 | **85.30** | **5.00** | - |
| OWLv2 BBox | 84.00 | 85.11 | 82.42 | 83.74 | 14.42 | 285 / 112 / 173 |
| DINO tiny BBox | 85.00 | 85.96 | **83.67** | 84.80 | 13.67 | 273 / 118 / 155 |
| DINO base BBox | 84.42 | 85.85 | 82.42 | 84.10 | 13.58 | 273 / 111 / 162 |

DINO tiny 박스가 세 박스 입력 중 가장 좋았다. 하지만 어떤 박스 입력도 Qwen 기준선의 accuracy와 F1을 넘지 못했다. 박스는 false negative를 고쳐 recall을 높였지만, 객체가 있다는 암시로 작동해 false positive도 크게 늘렸다.

## 결론

현재 결과는 “디텍터 단독 성능이 높을수록 BBox 유도 Qwen 성능도 높다”는 가설을 지지하지 않는다.

- 디텍터 단독 성능은 OWLv2가 가장 좋았다.
- BBox 유도 Qwen 성능은 DINO tiny가 가장 좋았다.
- DINO 계열 안에서는 tiny가 base보다 디텍터 단독과 BBox 유도 결과 모두 좋았다.
- 세 디텍터의 박스는 Qwen의 recall을 높였지만 FPR도 크게 높였다.

따라서 디텍터의 객체 존재 분류 성능만으로 VLM 개선 정도를 예측하기 어렵다. 박스의 위치 품질, 크기, score calibration, 박스가 생성되는 비율, 시각적 표시 방식도 함께 봐야 한다.

## 해석할 때 주의할 점

1. 비교 모델이 세 개뿐이므로 상관관계를 일반화할 수 없다.
2. POPE 정답은 객체 존재 여부이다. 박스의 IoU나 localization 품질을 직접 평가하지 않는다.
3. DINO 실험은 낮은 후처리 threshold에서 모든 질문에 후보 박스가 있었다. OWLv2는 2,400개 중 2,093개에만 박스가 있었다. 이 coverage 차이는 입력 조건의 차이다.
4. 모델마다 score 범위가 다르므로 raw score를 직접 비교할 수 없다.
5. 결과는 Qwen2.5-VL-3B와 COCO-POPE adversarial에 한정된다.

## 다음 실험

Grounding DINO tiny의 동일한 박스를 사용해 일반 BBox crop과 SAM2 mask crop을 비교한다. 이 비교는 사각형 안의 배경을 포함한 입력보다 픽셀 단위 객체 영역이 더 유용한지 확인한다.
