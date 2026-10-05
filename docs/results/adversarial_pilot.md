# Adversarial pilot result

## Research question

Can an external open-vocabulary detector reduce Qwen2.5-VL object-existence hallucinations through inference-time post-hoc verification?

## Setup

- VLM: Qwen2.5-VL-3B-Instruct
- Grounder: OWLv2-base-patch16-ensemble
- Benchmark: COCO-POPE adversarial, 3,000 questions over 500 images
- Split: image-level 20% dev / 80% test, seed 42
- Selection: thresholds chosen only on dev by balanced accuracy
- Test size: 2,400 questions over 400 images

The hard gate keeps `Yes` only when both Qwen and OWLv2 support object presence:

```text
gated_yes = qwen_yes AND (owlv2_score >= threshold)
```

Selected dev thresholds were 0.11 for OWLv2-only and 0.07 for the Qwen+OWLv2 gate.

## Test results

| Method | Accuracy | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Qwen-only | **86.54** | 93.98 | **78.08** | **85.30** | 5.00 |
| OWLv2-only | 85.79 | 85.18 | 86.67 | 85.91 | 15.08 |
| Qwen + OWLv2 gate | 86.08 | **95.01** | 76.17 | 84.55 | **4.00** |

The gate changed 35 Qwen `Yes` predictions to `No`:

- 12 were corrected false positives.
- 23 were introduced false negatives.
- Net change: 11 fewer correct answers, or -0.46 percentage points in accuracy.

## Interpretation

The external grounding score is useful for suppressing some hallucinated positive answers, but a one-way hard gate cannot recover Qwen false negatives. Detector misses also remove correct Qwen positives. The pilot therefore supports a precision/recall trade-off rather than an overall performance gain.

This is a narrow object-existence result on POPE. It does not establish improvement for open-ended captions, attributes, counts, relations, or general VLM hallucination.

## Planned extension

Compare the current hard gate with bbox-guided second-pass reasoning:

1. Overlay the best OWLv2 box on the full image and ask Qwen to verify the object.
2. Provide a box crop alongside the original image and ask Qwen to reconsider.
3. Add a random-box or matched-area crop control to separate localization benefit from simple zoom benefit.
4. Reuse the existing image-level dev/test split without tuning on test.

The key extension question is whether region-guided re-reasoning can correct false positives without the recall loss caused by the hard gate.
