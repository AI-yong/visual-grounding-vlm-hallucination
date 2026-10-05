# VIG Lab 2차 면담 준비: MLLM Hallucination × Visual Grounding 실험 실행 계획

작성일: 2026-08-19  
목표 기간: 14일  
문서 용도: GPU 서버/워크스테이션으로 옮겨 실험을 구현하고, VIG Lab 2차 면담에서 5~7분 동안 설명할 수 있는 결과를 만든다.

---

## 0. 이 실험을 한 문장으로 설명하면

> 멀티모달 언어모델이 이미지에 없는 객체를 있다고 답하는 object hallucination을 측정하고, 별도의 open-vocabulary detector가 계산한 객체–이미지 grounding 점수를 이용해 근거가 약한 `Yes` 답변을 걸러냈을 때 hallucination이 감소하는지 확인한다.

이 실험은 새로운 대규모 모델을 학습하는 프로젝트가 아니다. 공개 모델과 공개 벤치마크를 사용하여 다음 세 가지를 보여주는 소규모 분석 프로젝트다.

1. MLLM의 object hallucination을 직접 재현한다.
2. hallucination과 visual grounding score의 관계를 분석한다.
3. 간단한 grounding gate를 적용해 hallucination을 일부 줄일 수 있는지 확인한다.

### 0.1 지금까지 읽은 논문을 기준으로 아주 쉽게 설명하면

이 실험에는 서로 다른 역할을 하는 두 모델이 등장한다.

- `Qwen2.5-VL`: LLaVA나 BLIP-2처럼 이미지와 질문을 받아 언어로 답하는 MLLM이다. 이 모델이 이번 실험의 **시험 대상**이다.
- `OWLv2`: 질문에 나온 객체가 이미지의 어느 영역에 있는지 찾는 별도의 open-vocabulary detector다. CLIP처럼 텍스트와 이미지를 연결하지만, 이미지 전체가 아니라 **객체가 있을 만한 영역과 점수**를 출력한다. 이 모델은 시험 대상이 아니라 **외부 증거를 제공하는 검사 도구**다.

POPE는 Qwen2.5-VL에게 다음과 같은 Yes/No 문제를 내는 시험지다.

```text
이미지에 냉장고가 있나요? → 실제 정답: No
Qwen의 답: Yes → object hallucination(FP)
```

실험은 크게 세 단계다.

1. **환각 재현:** Qwen2.5-VL이 POPE 질문에서 실제로 얼마나 자주 없는 객체를 있다고 답하는지 측정한다.
2. **외부 시각 근거 검사:** 같은 객체를 OWLv2에도 물어보고, 객체 영역이 검출되는지와 grounding score를 얻는다. Qwen이 Yes라고 한 사례 중 정답인 TP와 환각인 FP가 이 점수로 구분되는지 본다.
3. **간단한 완화:** Qwen은 Yes라고 했지만 OWLv2 점수가 낮으면 최종 답을 No로 바꾸고, 환각 감소와 실제 객체를 놓치는 부작용을 함께 측정한다.

비유하면 Qwen2.5-VL은 문제를 푸는 학생이고, OWLv2는 이미지에서 해당 물체의 근거를 다시 확인하는 보조 검사자다. `grounding gate`는 학생이 “있다”고 답했지만 검사자가 근거를 거의 찾지 못하면 그 답을 보류하는 규칙이다.

### 0.2 여기서 alignment가 의미하는 것과 의미하지 않는 것

여기서 다루는 alignment는 RLHF나 instruction alignment가 아니다. CLIP에서 배운 이미지–텍스트 대응을 더 세밀하게 내려가, **객체를 나타내는 텍스트와 그 객체가 있을 법한 이미지 영역이 대응하는가**라는 cross-modal grounding 의미에 가깝다.

또한 이 실험은 Qwen2.5-VL 내부의 attention이나 visual token–language token 정렬을 직접 측정하지 않는다. OWLv2의 점수를 외부 시각 근거의 proxy로 사용한다. 따라서 정확한 설명은 다음과 같다.

> MLLM의 object hallucination을 재현하고, 외부 object-level visual grounding signal이 hallucinated Yes를 식별하고 완화하는 데 도움이 되는지 분석한다.

### 0.3 이 실험에서 비교할 세 방법

OWLv2를 gate에 사용했다면 OWLv2만으로 답했을 때의 성능도 반드시 비교해야 한다.

1. `Qwen-only`: Qwen2.5-VL의 원래 Yes/No 답변
2. `OWLv2-only`: dev set에서 정한 threshold보다 score가 높으면 Yes, 아니면 No
3. `Qwen + OWLv2 gate`: Qwen의 Yes 답변 중 OWLv2 근거가 약한 것만 No로 변경

이 비교가 없으면 gate의 개선이 두 모델을 잘 결합한 결과인지, 단순히 detector가 POPE에 더 적합했기 때문인지 구분할 수 없다.

---

## 1. 연구 관심과 실험의 연결

### 1.1 큰 관심 분야

- Multimodal Large Language Model의 hallucination 감소
- 이미지 속 실제 시각적 근거와 생성 문장 사이의 cross-modal alignment
- 객체·속성·관계 수준의 fine-grained visual grounding

### 1.2 이번 2주 실험에서 좁힐 범위

Hallucination 전체를 한 번에 다루지 않는다. 이번에는 가장 측정하기 쉬운 **object existence hallucination**만 다룬다.

예시:

- 이미지에 `refrigerator`가 없는데 모델이 “Yes, there is a refrigerator.”라고 답함
- 이미지에 `person`이 있는데 모델이 “No.”라고 답함

첫 번째가 이번 실험의 핵심인 false positive, 즉 object hallucination이다.

### 1.3 Alignment를 어떻게 정의할 것인가

이번 실험에서 alignment는 다음 의미로 사용한다.

> 텍스트로 주어진 객체 이름과 이미지 안에서 그 객체에 해당할 가능성이 높은 영역 사이의 정렬 정도

이를 OWLv2의 text-conditioned object detection confidence로 근사한다. 예를 들어 질문이 “Is there a refrigerator in the image?”이면 OWLv2에 `a photo of a refrigerator`를 query로 주고, 이미지 영역 중 가장 높은 detection score를 `grounding_score`로 사용한다.

주의: 이 점수는 MLLM 내부의 alignment를 직접 측정한 것이 아니다. 별도의 detector가 제공하는 **외부 visual evidence proxy**다. 따라서 결과를 설명할 때 “MLLM 내부 정렬을 증명했다”고 말하면 안 된다.

---

## 2. 핵심 연구 질문과 가설

### RQ1. MLLM은 POPE에서 어느 정도 object hallucination을 보이는가?

- Baseline 모델: `Qwen/Qwen2.5-VL-3B-Instruct`
- Dataset: POPE COCO subset
- Metrics: Accuracy, Precision, Recall, F1, Yes ratio, False Positive Rate

### RQ2. MLLM이 `Yes`라고 답한 사례 안에서 grounding score가 TP와 FP를 구분하는가?

다음 네 집단의 OWLv2 grounding score 분포를 비교한다.

- TP: 객체가 있고 MLLM도 Yes
- FP: 객체가 없는데 MLLM이 Yes - 핵심 hallucination
- TN: 객체가 없고 MLLM도 No
- FN: 객체가 있는데 MLLM이 No

핵심 비교는 Qwen2.5-VL이 Yes라고 답한 사례에 조건을 걸고 TP와 FP를 비교하는 것이다. 전체 TP와 FP의 단순 차이는 OWLv2가 존재 객체와 부재 객체를 구분하는 일반적인 detector 성능만 반영할 수 있기 때문이다.

보조적으로 같은 정답 label을 공유하는 FP–TN과 TP–FN도 비교한다.

- FP vs TN: 둘 다 객체가 없는데 Qwen이 일부에서만 Yes라고 한 이유 탐색
- TP vs FN: 둘 다 객체가 있는데 Qwen이 일부를 놓친 시각적 조건 탐색

가설 H1:

> Qwen2.5-VL이 Yes라고 답한 사례 중 FP의 grounding score는 TP보다 낮고, 이 점수는 TP와 FP를 chance보다 잘 구분할 것이다.

### RQ3. Grounding score가 낮은 `Yes` 답변을 억제하면 hallucination이 감소하는가?

간단한 gate:

```text
if MLLM_answer == "yes" and grounding_score < threshold:
    final_answer = "no"
else:
    final_answer = MLLM_answer
```

가설 H2:

> Grounding gate는 baseline보다 Precision을 높이고 False Positive Rate와 Yes ratio를 낮출 것이다.

예상 trade-off:

> detector가 실제 객체를 놓치면 정답 Yes를 No로 바꿔 Recall이 감소할 수 있다.

따라서 “Accuracy가 올랐다”만 보고하지 말고 Precision–Recall trade-off를 함께 보고해야 한다.

또한 gate 결과는 `Qwen-only`뿐 아니라 `OWLv2-only`와도 비교한다. Gate가 OWLv2-only보다 낫지 않다면 “멀티모달 신호 결합이 효과적이었다”고 주장하지 않고, 해당 데이터에서는 detector가 더 강한 존재 판별기였다고 해석한다.

---

## 3. 전체 실험 파이프라인

```text
POPE question + COCO image
          │
          ├── Qwen2.5-VL ──> baseline yes/no answer
          │
          └── 질문에서 object phrase 추출
                    │
                    └── OWLv2 ──> max grounding score + bounding box

baseline answer + grounding score
          │
          ├── 오류 유형 분석(TP/FP/TN/FN)
          ├── grounding score 분포 분석
          ├── OWLv2-only answer 생성
          └── threshold gate 적용
                    │
                    └── Qwen-only vs OWLv2-only vs gated metrics 비교
```

---

## 4. 사용할 데이터와 모델

### 4.1 POPE

공식 논문: <https://aclanthology.org/2023.emnlp-main.20/>  
공식 코드: <https://github.com/AoiDragon/POPE>

POPE 질문 예시:

```json
{"question_id": 1, "image": "COCO_val2014_000000016631.jpg", "text": "Is there a person in the image?", "label": "yes"}
{"question_id": 2, "image": "COCO_val2014_000000016631.jpg", "text": "Is there a refrigerator in the image?", "label": "no"}
```

POPE에는 대표적으로 세 가지 negative sampling setting이 있다.

- `random`: 무작위 부재 객체
- `popular`: 데이터에 자주 등장하지만 해당 이미지에는 없는 객체
- `adversarial`: 이미지 속 실제 객체와 자주 함께 등장하지만 해당 이미지에는 없는 객체

우선순위:

1. 첫 실행은 `random` 100문항
2. 필수 결과는 `random`, `popular`, `adversarial` 각각 최소 500문항
3. 시간이 허용되면 공식 전체 질문 평가

### 4.2 Baseline MLLM

Primary:

- `Qwen/Qwen2.5-VL-3B-Instruct`
- 공식 모델 카드: <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>

선택 이유:

- 3B 규모라 제한된 GPU에서도 실행 가능성이 높다.
- Hugging Face Transformers에서 이미지–텍스트 입력을 지원한다.
- 이번 프로젝트의 목표는 SOTA 비교가 아니라 재현 가능한 hallucination 분석이다.

Optional extension:

- GPU 메모리가 24GB 이상이고 시간이 남으면 `Qwen/Qwen2.5-VL-7B-Instruct` 추가
- 모델 비교는 필수가 아니다. 한 모델을 제대로 분석하는 것이 우선이다.

### 4.3 Visual grounder

Primary:

- `google/owlv2-base-patch16-ensemble`
- 공식 모델 카드: <https://huggingface.co/google/owlv2-base-patch16-ensemble>

OWLv2는 텍스트 query에 대응하는 이미지 영역과 detection score를 출력하는 open-vocabulary detector다.

중요한 한계:

- OWLv2 자체도 완벽하지 않다.
- OWLv2가 COCO 계열 detection 데이터의 영향을 받았을 수 있으므로, COCO 기반 POPE에서 강한 성능을 보일 수 있다.
- 이 실험의 gate는 일반적인 hallucination 해결책을 증명하는 것이 아니라 **visual grounding signal이 유용할 수 있는지 보는 proof-of-concept**다.

---

## 5. GPU 환경 사전 점검

GPU 서버에서 가장 먼저 다음을 실행하고 결과를 기록한다.

```bash
nvidia-smi
python3 --version
which python3
df -h
```

기록할 항목:

- GPU 이름
- VRAM 용량
- CUDA driver version
- OS
- Python version
- 사용 가능한 디스크 공간

### GPU 메모리별 권장 설정

| VRAM | 권장 설정 |
|---|---|
| 8~12GB | Qwen2.5-VL-3B 4-bit, batch size 1, 낮은 `max_pixels` |
| 16GB | Qwen2.5-VL-3B BF16 또는 4-bit |
| 24GB | Qwen2.5-VL-3B BF16 권장, 여유 시 7B 4-bit |
| 40GB 이상 | 3B/7B BF16 비교 가능하지만 필수 아님 |

처음에는 무조건 batch size 1로 시작한다.

---

## 6. 권장 프로젝트 구조

```text
vig_hallucination_grounding/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   ├── coco/
│   │   └── val2014/
│   └── pope/
├── third_party/
│   └── POPE/
├── src/
│   ├── inspect_environment.py
│   ├── prepare_pope.py
│   ├── run_mllm.py
│   ├── run_grounder.py
│   ├── apply_gate.py
│   ├── evaluate.py
│   └── analyze_errors.py
├── outputs/
│   ├── baseline/
│   ├── grounding/
│   ├── gated/
│   ├── figures/
│   └── cases/
└── logs/
```

---

## 7. 환경 설치

아래는 Linux GPU 환경 기준이다.

### 7.1 가상환경

```bash
mkdir -p vig_hallucination_grounding
cd vig_hallucination_grounding

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Conda를 선호하면 venv 대신 Conda를 사용해도 된다.

### 7.2 PyTorch

CUDA driver에 맞는 공식 PyTorch 설치 명령을 사용한다. 설치 후 반드시 확인한다.

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("vram_gb:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY
```

`cuda available: True`가 아니면 모델 설치 전에 PyTorch/CUDA 문제를 먼저 해결한다.

### 7.3 Python package

```bash
pip install -U \
  transformers \
  accelerate \
  qwen-vl-utils \
  bitsandbytes \
  pillow \
  pandas \
  numpy \
  scipy \
  scikit-learn \
  matplotlib \
  seaborn \
  tqdm \
  pyyaml \
  pycocotools
```

설치가 성공한 뒤 환경을 고정한다.

```bash
pip freeze > requirements.lock.txt
```

Flash Attention은 선택 사항이다. 처음부터 설치 문제에 시간을 쓰지 말고 기본 attention으로 baseline을 먼저 성공시킨다.

---

## 8. 데이터 준비

### 8.1 POPE clone

```bash
mkdir -p third_party
git clone https://github.com/AoiDragon/POPE.git third_party/POPE
find third_party/POPE/output -type f -name '*pope*.json' | sort
```

공식 repository의 `output` 아래에 COCO POPE 파일이 있는지 먼저 확인한다.

예상 파일명:

```text
coco_pope_random.json
coco_pope_popular.json
coco_pope_adversarial.json
```

실제 경로가 다르면 `find` 결과를 기준으로 config에 입력한다.

### 8.2 COCO val2014 이미지

POPE 질문의 이미지 파일명이 `COCO_val2014_...jpg` 형식이므로 COCO 2014 validation images가 필요하다.

```bash
mkdir -p data/coco
cd data/coco
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
cd ../..
```

다운로드 후 확인:

```bash
find data/coco/val2014 -type f -name '*.jpg' | head
```

### 8.3 데이터 sanity check

반드시 다음을 확인한다.

- JSONL 한 줄이 정상적으로 파싱되는가?
- `image` filename이 실제 COCO 이미지와 매칭되는가?
- label이 `yes` 또는 `no`인가?
- 동일 이미지가 여러 질문에 반복되는가?

Train/dev/test split은 **질문 단위가 아니라 image 단위**로 나눈다. 동일 이미지의 질문이 dev와 test에 동시에 들어가면 threshold tuning leakage가 발생한다.

권장 split:

- dev: 고유 이미지의 20%
- test: 고유 이미지의 80%
- random seed: 42

이 image split은 random/popular/adversarial 각각에서 따로 만들지 않고, 세 setting의 전체 고유 이미지에 대해 한 번 만든 뒤 공통으로 적용한다. 그래야 같은 이미지가 한 setting의 dev와 다른 setting의 test에 동시에 들어가는 leakage를 막을 수 있다.

---

## 9. 1단계: MLLM baseline

### 9.1 Prompt

POPE 원문의 질문을 사용하되 출력 형식을 제한한다.

```text
Answer the following question using only the visible content of the image.
Respond with exactly one word: Yes or No.

Question: Is there a refrigerator in the image?
```

이 prompt는 모든 샘플에서 동일하게 유지한다.

### 9.2 Generation setting

```text
do_sample = False
max_new_tokens = 4
temperature = 사용하지 않음
batch_size = 1부터 시작
seed = 42
```

Yes/No 평가에서 긴 reasoning을 생성시키지 않는다. 이번 실험은 chain-of-thought가 아니라 visual evidence 반영 여부를 본다.

### 9.3 출력 JSONL schema

`outputs/baseline/{setting}.jsonl`

```json
{
  "question_id": 1,
  "image": "COCO_val2014_000000016631.jpg",
  "question": "Is there a person in the image?",
  "label": "yes",
  "raw_answer": "Yes",
  "pred_answer": "yes",
  "setting": "random",
  "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
  "inference_seconds": 0.83
}
```

### 9.4 Answer normalization

권장 규칙:

```text
lowercase(raw_answer.strip())
starts with "yes" -> yes
starts with "no"  -> no
otherwise         -> invalid
```

`invalid` 비율을 반드시 기록한다. invalid가 많으면 prompt나 decoding을 수정하되, 수정 후 전체 실험을 동일 설정으로 다시 돌린다.

### 9.5 첫 성공 기준

처음에는 20개만 실행한다.

```bash
python src/run_mllm.py \
  --pope-file data/pope/coco_pope_random.json \
  --image-dir data/coco/val2014 \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --limit 20 \
  --output outputs/baseline/smoke_random_20.jsonl
```

확인할 것:

- GPU가 실제로 사용되는가?
- 이미지가 정상적으로 열리는가?
- 답변이 Yes/No로만 나오는가?
- 한 샘플당 추론 시간이 얼마인가?
- 예상 전체 실행 시간이 얼마인가?

### 9.6 OOM 대처 순서

1. batch size를 1로 고정
2. `max_new_tokens=4`
3. processor의 `max_pixels` 감소
4. BF16 사용 가능 여부 확인
5. 4-bit quantization 사용
6. 그래도 실패하면 더 작은 샘플로 CPU offload를 검토

OOM 때문에 7B 모델로 확장하지 못해도 프로젝트에는 문제가 없다.

---

## 10. 2단계: 질문에서 객체명 추출

POPE 기본 template은 다음과 같다.

```text
Is there a {object} in the image?
```

예상 parser:

```python
import re

def extract_object_phrase(question: str) -> str:
    q = question.strip().lower()
    patterns = [
        r"is there an? (.+?) in the image\?*$",
        r"is there (.+?) in the image\?*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, q)
        if match:
            return match.group(1).strip()
    raise ValueError(f"Unsupported question template: {question}")
```

주의:

- `a`와 `an` 처리
- 복수형 여부
- punctuation 제거
- parser 실패 샘플 수 기록

OWLv2 text query는 다음처럼 만든다.

```text
a photo of a {object_phrase}
```

---

## 11. 3단계: OWLv2 grounding score

### 11.1 핵심 정의

각 POPE 질문마다 OWLv2가 반환한 모든 box 중 가장 높은 score를 사용한다.

```text
grounding_score = max(all detection scores for queried object)
```

탐지 결과가 하나도 없으면:

```text
grounding_score = 0.0
```

### 11.2 출력 JSONL schema

`outputs/grounding/{setting}.jsonl`

```json
{
  "question_id": 1,
  "image": "COCO_val2014_000000016631.jpg",
  "question": "Is there a refrigerator in the image?",
  "object_phrase": "refrigerator",
  "grounding_query": "a photo of a refrigerator",
  "grounding_score": 0.031,
  "best_box_xyxy": [123.4, 54.2, 280.1, 310.9],
  "grounder_id": "google/owlv2-base-patch16-ensemble"
}
```

### 11.3 중요한 구현 주의사항

Threshold를 너무 일찍 적용하지 않는다. 처음에는 OWLv2 raw score를 저장하고, threshold 결정은 dev set 분석 단계에서 한다.

`post_process_object_detection(..., threshold=0.0)` 또는 매우 낮은 threshold를 사용해 가능한 한 raw max score를 보존한다. 모델/라이브러리에서 0.0이 문제를 일으키면 0.001처럼 매우 낮은 값을 사용하고 문서에 기록한다.

### 11.4 첫 성공 기준

- 20개 smoke sample에서 object phrase가 정상 추출됨
- OWLv2 score가 0~1 범위로 저장됨
- 최소 몇 개 이미지에서 bounding box가 생성됨
- 실제 객체가 있는 사례와 없는 사례의 score가 육안으로 어느 정도 구분됨

---

## 12. 4단계: Baseline 오류 분석

각 샘플을 다음 네 유형으로 분류한다.

| Label | MLLM prediction | Type | 의미 |
|---|---|---|---|
| yes | yes | TP | 객체를 올바르게 확인 |
| no | yes | FP | object hallucination |
| no | no | TN | 부재 객체를 올바르게 거절 |
| yes | no | FN | 실제 객체를 놓침 |

### 필수 수치

- Accuracy
- Precision
- Recall
- F1
- Yes ratio
- False Positive Rate: `FP / (FP + TN)`
- Invalid answer ratio

### 필수 그림

1. POPE setting별 baseline Accuracy/F1/FPR 막대그래프
2. TP/FP/TN/FN별 grounding score boxplot 또는 violin plot
3. Qwen이 Yes라고 답한 사례에서 grounding score로 TP와 FP를 구분하는 ROC/PR curve와 AUROC/AUPRC
4. 전체 샘플에서 OWLv2-only의 object existence 판별 성능

가능하면 metric은 이미지 단위 bootstrap 95% confidence interval과 함께 보고한다. Baseline과 gate는 같은 질문에 대한 paired prediction이므로 paired bootstrap 또는 McNemar test를 사용할 수 있다.

### 정성 분석

최소 다음 사례를 저장한다.

- grounding score가 낮은 FP 5개
- grounding score가 높은 FP 5개
- grounding score가 낮은 TP 5개
- gate가 올바르게 수정한 사례 5개
- gate가 오히려 틀리게 만든 사례 5개

각 사례에는 이미지, 질문, 정답, MLLM 답변, grounding score, box를 포함한다.

---

## 13. 5단계: Grounding gate

### 13.0 Detector-only control

Gate를 평가하기 전에 같은 dev threshold를 이용해 OWLv2-only 답변을 만든다.

```text
if grounding_score >= threshold:
    owlv2_answer = "yes"
else:
    owlv2_answer = "no"
```

Test set에서 `Qwen-only`, `OWLv2-only`, `Qwen + OWLv2 gate`를 동일한 metric으로 비교한다. 이는 gate가 단순히 detector의 판단을 빌린 것인지, Qwen과 detector의 상호보완성이 있는지를 확인하는 필수 control이다.

### 13.1 Threshold 선택

Test set 결과를 보고 threshold를 선택하면 안 된다.

절차:

1. image-level dev split에서 threshold 탐색
2. candidate threshold 예: `0.00, 0.01, 0.02, ..., 0.50`
3. dev F1 또는 balanced accuracy가 최대인 threshold 선택
4. threshold를 고정
5. test set에 한 번 적용

Threshold 목적을 미리 정한다.

- hallucination 억제를 우선하면 Precision 또는 FPR 개선 중심
- 전체 균형을 원하면 F1 또는 balanced accuracy 중심

면담용 프로젝트에서는 **FPR 감소를 주목표**로 두고, Recall 손실을 함께 보고하는 것이 자연스럽다.

### 13.2 Gate 결과에 반드시 추가할 통계

- baseline FP 수
- gate 이후 FP 수
- corrected FP: FP → TN
- introduced FN: TP → FN
- intervention count: Yes를 No로 바꾼 총 횟수
- precision 변화
- recall 변화
- FPR 변화

### 13.3 결과 표 template

| Setting | Method | Acc | Precision | Recall | F1 | FPR | Yes ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | MLLM baseline |  |  |  |  |  |  |
| Random | OWLv2-only |  |  |  |  |  |  |
| Random | + Grounding gate |  |  |  |  |  |  |
| Popular | MLLM baseline |  |  |  |  |  |  |
| Popular | OWLv2-only |  |  |  |  |  |  |
| Popular | + Grounding gate |  |  |  |  |  |  |
| Adversarial | MLLM baseline |  |  |  |  |  |  |
| Adversarial | OWLv2-only |  |  |  |  |  |  |
| Adversarial | + Grounding gate |  |  |  |  |  |  |

---

## 14. 결과 해석 가이드

### 결과 A: FP의 grounding score가 TP보다 확실히 낮음

해석:

- 외부 visual grounding signal이 hallucination 탐지에 유용할 가능성
- MLLM의 Yes 답변이 항상 충분한 시각적 근거를 동반하지는 않음

말하면 안 되는 것:

- “alignment 부족이 hallucination의 유일한 원인이다.”
- “OWLv2 score가 MLLM 내부 alignment를 정확히 측정한다.”

### 결과 B: Gate가 FPR을 낮추지만 Recall도 크게 낮춤

해석:

- 단순 hard threshold는 시각적 근거가 약한 hallucination을 줄일 수 있지만 detector miss로 실제 객체까지 제거함
- 향후에는 uncertainty-aware soft gating이나 MLLM 내부 token confidence와 결합할 필요

### 결과 C: FP와 TP score가 거의 구분되지 않음

이것도 실패가 아니라 연구 결과다.

가능한 원인:

- OWLv2 score가 MLLM hallucination을 설명하는 적절한 proxy가 아님
- 객체명 query가 detector vocabulary/표현과 맞지 않음
- COCO 객체가 작거나 가려져 있음
- hallucination의 주원인이 vision-language alignment보다 language prior/decoder에 있을 수 있음
- MLLM과 detector가 서로 다른 시각적 오류를 보임

다음 실험 제안:

- query template 여러 개 비교
- MLLM에게 bounding box를 먼저 출력하게 한 뒤 답변하게 하기
- GroundingDINO 등 다른 grounder와 비교
- object attribute hallucination으로 확장

### 결과 D: Baseline MLLM이 POPE에서 거의 틀리지 않음

POPE가 현재 모델에 너무 쉬울 수 있다.

축소하지 말고 다음 순서로 난도를 올린다.

1. adversarial setting 우선 분석
2. POPEv2/HOPE/H-POPE 같은 더 어려운 benchmark 검토
3. object attribute 질문으로 확장

단, 2주 내에는 benchmark를 여러 개 벌리지 않는다. baseline이 포화됐을 때만 확장한다.

---

### 14.1 선택 확장: 시각적 근거에 대한 간단한 causal intervention

시간이 남으면 대표 TP/FP 사례에 한해 다음 세 입력에서 Qwen2.5-VL의 답변 변화를 비교한다.

1. 원본 이미지
2. OWLv2가 찾은 영역을 확대한 crop
3. 해당 영역을 mask 또는 blur한 이미지

실제 객체를 근거로 답했다면 관련 영역을 제거했을 때 Yes 응답이 줄어들 가능성이 있다. 반대로 근거 영역을 제거해도 hallucinated Yes가 유지된다면 language prior나 다른 문맥 신호의 영향을 의심할 수 있다.

이 분석은 correlation만 보는 grounding score 실험보다 alignment 문제와 더 직접적으로 연결되지만, 완전한 인과 증명은 아니다. Mask가 비자연적인 이미지를 만들고 crop은 global context를 잃게 한다는 한계를 함께 보고한다. 본 실험의 필수 단계가 아니라 세 방법 비교가 완료된 뒤 수행하는 선택 확장이다.

---

## 15. Visual Funnel과 연결하는 방법

VIG Lab의 Visual Funnel:

<https://openaccess.thecvf.com/content/CVPR2026F/html/Jung_Visual_Funnel_Resolving_Contextual_Blindness_in_Multimodal_Large_Language_Models_CVPRF_2026_paper.html>

연결 논리:

- Visual Funnel은 단순히 crop을 많이 넣는 것이 아니라 focal detail과 global context 사이의 구조를 유지해야 한다고 본다.
- 이번 실험은 MLLM의 생성 답변이 실제 이미지 속 객체 근거와 일치하는지 확인한다.
- 두 연구는 동일한 방법을 쓰지는 않지만, **모델이 필요한 시각 정보를 실제 답변에 올바르게 연결하는가**라는 공통 문제의식을 가진다.

면담 표현:

> Visual Funnel이 다루는 contextual blindness를 읽으면서, 이미지에 정보가 존재하는 것과 모델이 그 정보를 답변의 근거로 활용하는 것은 다를 수 있다고 생각했습니다. 그래서 우선 object hallucination을 재현하고, 별도의 visual grounding score가 근거 없는 Yes 답변을 구분하는 데 도움이 되는지 확인하는 작은 실험을 진행했습니다.

주의:

> Visual Funnel이 object hallucination을 직접 해결하는 논문이라고 설명하지 않는다.

---

## 16. 14일 일정

### Day 1: GPU 환경 확인

- `nvidia-smi`, Python, CUDA 확인
- 프로젝트 폴더와 가상환경 생성
- PyTorch GPU 동작 확인
- 환경 정보 저장

완료 조건: `torch.cuda.is_available() == True`

### Day 2: 데이터 준비

- POPE clone
- COCO val2014 다운로드
- JSONL/image path sanity check
- random 20문항 subset 생성

완료 조건: Python에서 이미지와 질문 1개를 함께 열 수 있음

### Day 3: Qwen2.5-VL smoke test

- 이미지 1개 질의
- POPE 20문항 실행
- Yes/No normalization 확인

완료 조건: baseline JSONL 20줄 생성

### Day 4: Baseline 확장

- random 100~500문항
- 추론 시간 및 VRAM 기록
- 기본 metric 계산

완료 조건: Accuracy/F1/FPR 출력

### Day 5: 세 setting baseline

- random/popular/adversarial 실행
- baseline 결과 표 생성

완료 조건: 세 setting 비교표

### Day 6: OWLv2 smoke test

- 객체명 parser
- OWLv2 query
- 20문항 grounding score 저장
- bounding box 시각화

완료 조건: score와 box가 포함된 JSONL

### Day 7: OWLv2 전체 실행

- baseline과 동일한 질문에 grounder 실행
- 결과 merge

완료 조건: 각 baseline row에 grounding score 연결

### Day 8: 오류 유형/분포 분석

- TP/FP/TN/FN 분류
- boxplot/violin plot
- AUROC 계산

완료 조건: 그림 2개 이상

### Day 9: Dev threshold 선택

- image-level dev/test split
- threshold sweep
- 선택 기준과 threshold 기록

완료 조건: test를 보지 않고 threshold 고정

### Day 10: Gate test 평가

- test에 gate 적용
- baseline vs gate 비교
- corrected FP / introduced FN 계산

완료 조건: 핵심 결과 표 완성

### Day 11: 정성 사례 분석

- 성공 사례와 실패 사례 저장
- 각 사례에 한두 문장 해석 작성

완료 조건: 발표용 사례 최소 6개

### Day 12: 연구 해석/한계 정리

- 결과가 가설을 지지하는지 판단
- detector proxy와 COCO overlap 한계 작성
- Visual Funnel과 연결하되 과장하지 않기

완료 조건: 결론 3문장, 한계 3개, 다음 연구 3개

### Day 13: 발표 자료

- 5~7장 슬라이드
- 5분 발표 대본
- 예상 기술 질문 답변

완료 조건: 시간 재고 5분 이내 발표

### Day 14: 재현성 및 면담 준비

- config, seed, commit, package version 확인
- 결과 백업
- 모의 면담 2회

완료 조건: 새 환경에서 실행 순서를 설명할 수 있음

---

## 17. 최소/목표/확장 성공 기준

### 최소 성공 - 시간이 부족해도 반드시 완료

- Qwen2.5-VL로 POPE 300문항 이상 평가
- baseline metric 1개 표
- FP hallucination 사례 10개 분석
- OWLv2 score를 최소 일부 사례에 계산
- 5장 발표 자료

### 목표 성공 - 권장

- POPE 세 setting 각각 500문항 이상
- OWLv2 grounding score 전체 계산
- dev threshold와 held-out test 분리
- baseline vs gate 정량 비교
- 그림 3개, 정성 사례 10개 이상

### 확장 성공 - 시간이 남을 때만

- Qwen2.5-VL-7B 추가 비교
- H-POPE 또는 attribute hallucination 일부 평가
- hard gate 대신 logistic calibration
- MLLM confidence와 OWLv2 score 결합
- 대표 사례의 original/crop/mask intervention 비교

확장 실험은 목표 성공이 끝난 뒤에만 시작한다.

---

## 18. 재현성 체크리스트

- [ ] random seed 42 고정
- [ ] 모델 ID와 revision 기록
- [ ] `pip freeze` 저장
- [ ] `nvidia-smi` 저장
- [ ] POPE git commit hash 기록
- [ ] 질문/이미지 단위 데이터 개수 기록
- [ ] image-level dev/test split 파일 저장
- [ ] threshold 선택 기준을 사전에 기록
- [ ] raw model output 보존
- [ ] raw OWLv2 score 보존
- [ ] 실패/invalid sample을 삭제하지 않고 별도 표시
- [ ] 모든 결과 JSONL에 config 정보 기록

권장 명령:

```bash
nvidia-smi > logs/nvidia_smi.txt
pip freeze > logs/requirements.lock.txt
git -C third_party/POPE rev-parse HEAD > logs/pope_commit.txt
```

---

## 19. 자주 생길 문제와 해결 순서

### `KeyError: qwen2_5_vl`

```bash
pip install -U transformers accelerate
```

그래도 실패하면 공식 Qwen2.5-VL 모델 카드가 안내하는 Transformers 설치법을 확인한다.

### CUDA OOM

- batch size 1
- `max_new_tokens=4`
- `max_pixels` 감소
- BF16 확인
- 4-bit quantization
- 7B가 아니라 3B 사용

### 모델이 Yes/No 이외의 긴 문장을 출력

- “Respond with exactly one word: Yes or No.” 추가
- `do_sample=False`
- `max_new_tokens=4`
- normalization parser 적용

### POPE 이미지가 없음

- 질문 파일의 `image` 값 확인
- COCO `val2014`가 맞는지 확인
- `COCO_val2014_...jpg` 파일명 보존 여부 확인

### OWLv2 score가 거의 모두 0

- query가 `a photo of a {object}` 형식인지 확인
- post-processing threshold가 너무 높지 않은지 확인
- 이미지 RGB 변환 확인
- object parser 결과 확인
- 샘플 이미지에 대한 box 시각화 수행

### Gate가 모든 Yes를 No로 바꿈

- threshold leakage/scale 확인
- dev score histogram 확인
- threshold 범위를 실제 score 분포에 맞게 재설정
- detector score를 확률처럼 해석하지 않기

---

## 20. 발표 자료 구성

### Slide 1. 배경과 관심

- MLLM hallucination
- visual grounding / fine-grained alignment 관심
- 왜 이 문제를 선택했는가

### Slide 2. 연구 질문

- MLLM이 이미지에 없는 객체를 생성하는가?
- grounding score가 hallucination을 구분하는가?
- grounding gate가 FP를 줄이는가?

### Slide 3. 방법

- POPE + Qwen2.5-VL + OWLv2
- pipeline diagram
- gate 수식/규칙

### Slide 4. Baseline 결과

- setting별 Accuracy/F1/FPR

### Slide 5. Alignment 분석

- TP/FP grounding score 분포
- 대표 hallucination 사례

### Slide 6. Gate 결과

- baseline vs gate
- corrected FP와 introduced FN

### Slide 7. 한계와 다음 방향

- external detector proxy
- COCO overlap
- attribute/relation hallucination 확장
- MLLM 내부 region-token alignment 분석 가능성

---

## 21. 면담용 60초 설명 template

결과 숫자가 나온 뒤 빈칸을 채운다.

> 1차 면담 이후에는 멀티모달 모델의 hallucination을 visual grounding과 alignment 관점에서 공부하고 있습니다. 우선 범위를 object hallucination으로 좁혀, Qwen2.5-VL을 POPE에서 평가했습니다. 그 결과 [setting]에서 false positive가 [수치]만큼 나타났습니다. 이후 질문에 포함된 객체와 이미지 영역 사이의 grounding score를 OWLv2로 계산해 보니, hallucinated Yes 사례는 정상적인 True Positive보다 score가 [낮았다/유의미한 차이가 없었다]는 결과를 확인했습니다. 이를 바탕으로 grounding score가 낮은 Yes 답변을 억제하는 간단한 gate를 적용했고, FPR이 [A]에서 [B]로 변한 대신 Recall은 [C]에서 [D]로 변했습니다. 아직 외부 detector를 proxy로 사용했다는 한계가 있지만, 향후에는 MLLM 내부의 region-token alignment를 직접 분석하고 객체뿐 아니라 속성과 관계 hallucination으로 확장하고 싶습니다.

---

## 22. 예상 면담 질문

### 왜 POPE를 선택했나?

> 2주 안에 hallucination을 정량적으로 재현하기 위해 object existence를 Yes/No로 평가할 수 있는 POPE를 선택했습니다. 전체 hallucination을 대표한다고 보지는 않으며, 이후에는 attribute와 relation 오류로 확장할 필요가 있습니다.

### OWLv2를 사용하는 것이 alignment를 개선한 것인가?

> MLLM 자체의 representation을 학습하거나 수정한 것은 아닙니다. 별도의 detector가 주는 region-text grounding score를 external evidence로 사용한 proof-of-concept입니다.

### 왜 CLIP global similarity가 아니라 OWLv2인가?

> global similarity는 이미지 전체와 문장의 유사도를 주기 때문에 특정 객체가 실제 어느 영역에 존재하는지 확인하기 어렵습니다. OWLv2는 text query에 대응하는 영역과 score를 제공하므로 object-level grounding proxy로 더 직접적이라고 판단했습니다.

### Gate가 성능을 높였으면 hallucination 문제가 해결된 것인가?

> 아닙니다. detector가 놓친 객체를 제거하면서 Recall이 감소할 수 있고, COCO와 detector training data의 overlap 가능성도 있습니다. 이번 결과는 visual grounding signal의 가능성을 확인한 소규모 분석입니다.

### Hallucination은 alignment 문제만으로 발생하나?

> 아닙니다. language prior, instruction-tuning data, visual encoder의 해상도, decoding 등 여러 원인이 있습니다. 저는 그중 시각적 근거와 생성 언어 사이의 fine-grained alignment 문제에 관심이 있습니다.

### Visual Funnel과 어떻게 연결되는가?

> 동일한 방법은 아니지만, 이미지에 필요한 정보가 존재해도 모델이 세부 정보와 전체 맥락을 구조적으로 연결하지 못하면 잘못된 답을 만들 수 있다는 문제의식에서 연결됩니다.

---

## 23. 이 실험에서 절대 과장하지 말아야 할 주장

다음 표현은 피한다.

- “Hallucination의 원인이 alignment 부족임을 증명했다.”
- “OWLv2 score가 MLLM의 내부 attention 또는 alignment다.”
- “새로운 SOTA hallucination 해결 방법을 제안했다.”
- “Visual Funnel을 재현했다.”
- “Grounding gate가 일반적인 MLLM hallucination을 해결한다.”

대신 다음처럼 말한다.

- “Object hallucination을 재현하고 오류 사례를 분석했다.”
- “External grounding score와 hallucination의 관계를 살펴봤다.”
- “간단한 proof-of-concept gate의 효과와 trade-off를 확인했다.”
- “향후 MLLM 내부 region-token alignment 분석으로 확장하고 싶다.”

---

## 24. GPU 환경으로 넘길 때 사용할 구현 요청문

아래 내용을 GPU 환경의 코딩 에이전트 또는 개발자에게 그대로 전달할 수 있다.

```text
첨부된 실행 계획 문서를 기준으로 `vig_hallucination_grounding` 프로젝트를 구현해줘.

우선순위:
1. 환경/GPU 점검
2. POPE random 20문항 Qwen2.5-VL-3B smoke test
3. baseline JSONL과 metric 생성
4. 질문에서 object phrase 추출
5. OWLv2 grounding score와 bounding box 저장
6. TP/FP/TN/FN 분포 분석
7. image-level dev/test split을 사용한 grounding threshold gate
8. baseline vs gated result table과 figure 생성

중요 조건:
- 처음에는 batch size 1과 20문항만 사용한다.
- 모든 raw answer와 raw grounding score를 보존한다.
- threshold는 dev에서만 선택하고 test 결과를 보고 조정하지 않는다.
- OOM 발생 시 3B 모델, 낮은 max_pixels, 4-bit 순서로 축소한다.
- 각 단계가 끝날 때 생성된 파일, metric, 남은 문제를 보고한다.
- 실행 계획에 없는 대규모 학습이나 benchmark 확장은 하지 않는다.
```

---

## 25. 최종 체크

면담 전 다음 질문에 막힘없이 답할 수 있으면 준비가 된 것이다.

- 내가 측정한 hallucination은 정확히 어떤 종류인가?
- POPE의 random/popular/adversarial 차이는 무엇인가?
- Precision, Recall, FPR 중 무엇이 왜 변했는가?
- OWLv2 score를 왜 alignment proxy로 사용했는가?
- hard gate가 만든 부작용은 무엇인가?
- 이 결과로 말할 수 있는 것과 말할 수 없는 것은 무엇인가?
- Visual Funnel의 문제의식과 어떻게 연결되는가?
- 다음 연구에서는 무엇을 더 직접적으로 측정하고 싶은가?

이 프로젝트의 성공 기준은 높은 점수가 아니다. **문제를 정의하고, 재현 가능한 실험을 만들고, 결과와 한계를 정직하게 설명하는 것**이다.
