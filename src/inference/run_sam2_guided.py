from __future__ import annotations

import argparse
import gc
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

from src.inference.run_bbox_guided import clamp_box, latest_grounding_rows
from src.inference.run_grounding_control import generate_one, select_rows
from src.inference.run_mllm import load_model as load_qwen
from src.data.pope import normalize_binary_answer
from src.utils.jsonl import append_jsonl, completed_question_ids


MASK_PROMPT_TEMPLATE = """Answer the question using only visible image evidence.
The first image is the full image. The second image is a segmented candidate region for "{object_phrase}". Gray pixels are outside the predicted mask.
The segmented region alone does not prove that the object exists. Verify it against the full image.
Respond with exactly one word: Yes or No.

Question: {question}"""

NO_REGION_TEMPLATE = """Answer the question using only visible image evidence.
No usable detector candidate region is available for "{object_phrase}".
This detector result can be wrong. Inspect the full image and make the final decision yourself.
Respond with exactly one word: Yes or No.

Question: {question}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen with SAM2-segmented detector regions on POPE."
    )
    parser.add_argument("--pope-file", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--subset", choices=("all", "dev", "test"), default="all")
    parser.add_argument("--qwen-model-id", required=True)
    parser.add_argument("--sam-model-id", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    return parser.parse_args()


def padded_bounds(
    box: tuple[int, int, int, int], width: int, height: int, ratio: float = 0.1
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    pad_x = max(2, round((right - left) * ratio))
    pad_y = max(2, round((bottom - top) * ratio))
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x + 1),
        min(height, bottom + pad_y + 1),
    )


def make_masked_crop(
    image: Image.Image,
    mask: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[Image.Image, float]:
    if mask.shape != (image.height, image.width):
        raise ValueError(f"Mask shape {mask.shape} does not match image size {image.size}")
    binary = mask.astype(bool)
    area_ratio = float(binary.mean())
    if not binary.any():
        raise ValueError("SAM2 returned an empty mask")
    bounds = padded_bounds(box, image.width, image.height)
    crop = image.convert("RGB").crop(bounds)
    mask_crop = Image.fromarray((binary.astype(np.uint8) * 255)).crop(bounds)
    isolated = Image.new("RGB", crop.size, (127, 127, 127))
    isolated.paste(crop, mask=mask_crop)
    return isolated, area_ratio


def load_models(args: argparse.Namespace):
    sam_processor = Sam2Processor.from_pretrained(
        args.sam_model_id, cache_dir=args.cache_dir
    )
    sam_model = Sam2Model.from_pretrained(
        args.sam_model_id,
        cache_dir=args.cache_dir,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to("cuda")
    sam_model.eval()
    qwen_args = SimpleNamespace(
        model_id=args.qwen_model_id,
        cache_dir=args.cache_dir,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    qwen_model, qwen_processor = load_qwen(qwen_args)
    return sam_model, sam_processor, qwen_model, qwen_processor


def group_pending(rows, completed):
    grouped = defaultdict(list)
    for row in rows:
        if row.question_id not in completed:
            grouped[row.image].append(row)
    return dict(grouped)


def segment_boxes(model, processor, image: Image.Image, boxes):
    inputs = processor(
        images=image,
        input_boxes=[[list(box) for box in boxes]],
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"]
    )[0]
    masks = masks[:, 0].numpy()
    scores = outputs.iou_scores.detach().float().cpu()[0, :, 0].tolist()
    del inputs, outputs
    return masks, scores


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start a long CPU run.")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rows = select_rows(args)
    grounding = latest_grounding_rows(args.grounding)
    completed = completed_question_ids(args.output, args.retry_errors)
    grouped = group_pending(rows, completed)
    pending_count = sum(len(items) for items in grouped.values())
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"sam={args.sam_model_id} subset={args.subset} rows={len(rows)} "
        f"completed={len(completed)} pending={pending_count} pending_images={len(grouped)}",
        flush=True,
    )
    if not grouped:
        print("nothing_to_do=true", flush=True)
        return
    for row in rows:
        candidate = grounding.get(row.question_id)
        if candidate is None or candidate.get("status", "ok") != "ok":
            raise ValueError(f"Question {row.question_id} has unusable grounding output")

    sam_model, sam_processor, qwen_model, qwen_processor = load_models(args)
    successes = 0
    errors = 0
    consecutive_errors = 0
    run_started = time.perf_counter()
    for image_position, (image_name, image_rows) in enumerate(grouped.items(), start=1):
        try:
            with Image.open(args.image_dir / image_name) as opened:
                full_image = opened.convert("RGB")
            valid_rows = []
            valid_boxes = []
            fallback = {}
            for row in image_rows:
                raw_box = grounding[row.question_id].get("best_box")
                if raw_box is None:
                    fallback[row.question_id] = "missing"
                    continue
                try:
                    box = clamp_box(raw_box, *full_image.size)
                except ValueError:
                    fallback[row.question_id] = "invalid_after_clamp"
                    continue
                valid_rows.append(row)
                valid_boxes.append(box)

            masks = []
            mask_scores = []
            if valid_boxes:
                masks, mask_scores = segment_boxes(
                    sam_model, sam_processor, full_image, valid_boxes
                )
            segmented = {
                row.question_id: (box, mask, score)
                for row, box, mask, score in zip(
                    valid_rows, valid_boxes, masks, mask_scores
                )
            }

            for row in image_rows:
                started = time.perf_counter()
                candidate = grounding[row.question_id]
                try:
                    if row.question_id in segmented:
                        box, mask, mask_score = segmented[row.question_id]
                        masked_crop, area_ratio = make_masked_crop(full_image, mask, box)
                        images = [full_image, masked_crop]
                        template = MASK_PROMPT_TEMPLATE
                        mask_status = "ok"
                    else:
                        images = [full_image]
                        template = NO_REGION_TEMPLATE
                        box = None
                        mask_score = None
                        area_ratio = None
                        mask_status = fallback[row.question_id]
                    prompt = template.format(
                        question=row.question,
                        object_phrase=candidate["object_phrase"],
                    )
                    raw_answer = generate_one(
                        qwen_model,
                        qwen_processor,
                        images,
                        prompt,
                        args.max_new_tokens,
                    )
                    prediction = normalize_binary_answer(raw_answer)
                    elapsed = time.perf_counter() - started
                    append_jsonl(
                        args.output,
                        {
                            "question_id": row.question_id,
                            "image": row.image,
                            "question": row.question,
                            "label": row.label,
                            "object_phrase": candidate["object_phrase"],
                            "grounding_score": candidate["max_score"],
                            "candidate_box": candidate.get("best_box"),
                            "rendered_box": list(box) if box else None,
                            "has_box": box is not None,
                            "mask_status": mask_status,
                            "mask_iou_score": mask_score,
                            "mask_area_ratio": area_ratio,
                            "raw_answer": raw_answer,
                            "pred_answer": prediction,
                            "intervention": "sam2_masked_crop",
                            "setting": args.setting,
                            "qwen_model_id": args.qwen_model_id,
                            "sam_model_id": args.sam_model_id,
                            "status": "ok",
                            "inference_seconds": round(elapsed, 4),
                            "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    successes += 1
                    consecutive_errors = 0
                except Exception as error:
                    elapsed = time.perf_counter() - started
                    append_jsonl(
                        args.output,
                        {
                            "question_id": row.question_id,
                            "image": row.image,
                            "question": row.question,
                            "label": row.label,
                            "intervention": "sam2_masked_crop",
                            "setting": args.setting,
                            "status": "error",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "inference_seconds": round(elapsed, 4),
                            "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    errors += 1
                    consecutive_errors += 1
                    print(
                        f"ERROR qid={row.question_id} {type(error).__name__}: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if consecutive_errors >= args.max_consecutive_errors:
                        raise
                finally:
                    if (successes + errors) % 25 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
            if image_position <= 3 or image_position % 25 == 0:
                print(
                    f"progress_images={image_position}/{len(grouped)} image={image_name} "
                    f"questions={len(image_rows)} masks={len(valid_boxes)}",
                    flush=True,
                )
        except Exception as error:
            raise RuntimeError(
                f"Failed while processing image {image_name}: {type(error).__name__}: {error}"
            ) from error

    total_elapsed = time.perf_counter() - run_started
    print(
        f"run_complete=true processed={pending_count} successes={successes} errors={errors} "
        f"total_seconds={total_elapsed:.1f} python={platform.python_version()} "
        f"torch={torch.__version__}",
        flush=True,
    )


if __name__ == "__main__":
    main()
