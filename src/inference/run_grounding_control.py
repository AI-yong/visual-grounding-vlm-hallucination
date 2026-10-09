from __future__ import annotations

import argparse
import gc
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from src.data.pope import load_pope_rows, normalize_binary_answer
from src.inference.run_bbox_guided import (
    clamp_box,
    draw_candidate_box,
    latest_grounding_rows,
)
from src.inference.run_mllm import load_model
from src.utils.jsonl import append_jsonl, completed_question_ids


PROMPT_ONLY_TEMPLATE = """Answer the question using only visible image evidence.
Recheck the full image carefully. Do not assume that an object exists because it is common in similar scenes.
Respond with exactly one word: Yes or No.

Question: {question}"""

BOX_TEMPLATE = """Answer the question using only visible image evidence.
The red rectangle is a candidate region proposed by an object detector for "{object_phrase}".
Verify the pixels inside and around the rectangle. The rectangle alone does not prove that the object exists.
Respond with exactly one word: Yes or No.

Question: {question}"""

CROP_TEMPLATE = """Answer the question using only visible image evidence.
The first image is the full image. The second image is an enlarged candidate region proposed by an object detector for "{object_phrase}".
The crop alone does not prove that the object exists. Verify it against the full image.
Respond with exactly one word: Yes or No.

Question: {question}"""

NO_REGION_TEMPLATE = """Answer the question using only visible image evidence.
No usable detector candidate region is available for "{object_phrase}".
This detector result can be wrong. Inspect the full image and make the final decision yourself.
Respond with exactly one word: Yes or No.

Question: {question}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled Qwen visual-grounding interventions on POPE."
    )
    parser.add_argument(
        "--intervention",
        choices=("bbox_overlay", "prompt_only", "random_box", "bbox_crop"),
        required=True,
    )
    parser.add_argument("--pope-file", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--subset", choices=("all", "dev", "test"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    return parser.parse_args()


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def matched_random_box(
    source: tuple[int, int, int, int], width: int, height: int, seed: int
) -> tuple[int, int, int, int]:
    box_width = source[2] - source[0]
    box_height = source[3] - source[1]
    max_left = max(0, width - 1 - box_width)
    max_top = max(0, height - 1 - box_height)
    rng = random.Random(seed)
    candidates = []
    for _ in range(32):
        left = rng.randint(0, max_left) if max_left else 0
        top = rng.randint(0, max_top) if max_top else 0
        candidate = (left, top, left + box_width, top + box_height)
        candidates.append(candidate)
    return min(candidates, key=lambda candidate: box_iou(source, candidate))


def padded_crop(
    image: Image.Image, box: tuple[int, int, int, int], padding_ratio: float = 0.1
) -> Image.Image:
    left, top, right, bottom = box
    pad_x = max(2, round((right - left) * padding_ratio))
    pad_y = max(2, round((bottom - top) * padding_ratio))
    crop_box = (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(image.width, right + pad_x + 1),
        min(image.height, bottom + pad_y + 1),
    )
    return image.crop(crop_box).convert("RGB")


def select_rows(args: argparse.Namespace):
    rows = load_pope_rows(args.pope_file)
    if args.subset != "all":
        if args.split_file is None:
            raise ValueError("--split-file is required for dev or test")
        split = json.loads(args.split_file.read_text(encoding="utf-8"))
        allowed = set(split[f"{args.subset}_images"])
        rows = [row for row in rows if row.image in allowed]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    return rows


def validate_inputs(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start a long CPU run.")
    rows = select_rows(args)
    grounding = latest_grounding_rows(args.grounding)
    for row in rows:
        if not (args.image_dir / row.image).is_file():
            raise FileNotFoundError(args.image_dir / row.image)
        candidate = grounding.get(row.question_id)
        if candidate is None or candidate.get("status", "ok") != "ok":
            raise ValueError(f"Question {row.question_id} has unusable grounding output")
        if candidate.get("image") != row.image or candidate.get("label") != row.label:
            raise ValueError(f"Question {row.question_id} metadata differs from grounding")
    return rows, grounding


def prepare_intervention(
    image: Image.Image,
    candidate: dict,
    intervention: str,
    seed: int,
):
    full_image = image.convert("RGB").copy()
    phrase = str(candidate["object_phrase"])
    if intervention == "prompt_only":
        has_box = candidate.get("best_box") is not None
        return [full_image], PROMPT_ONLY_TEMPLATE, None, has_box, "available_not_shown" if has_box else "missing"

    raw_box = candidate.get("best_box")
    if raw_box is None:
        return [full_image], NO_REGION_TEMPLATE, None, False, "missing"
    try:
        source_box = clamp_box(raw_box, *full_image.size)
    except ValueError:
        return [full_image], NO_REGION_TEMPLATE, None, False, "invalid_after_clamp"

    if intervention == "bbox_overlay":
        boxed_image, rendered_box = draw_candidate_box(full_image, list(source_box))
        return [boxed_image], BOX_TEMPLATE, rendered_box, True, "rendered"
    if intervention == "random_box":
        shown_box = matched_random_box(source_box, *full_image.size, seed)
        boxed_image, rendered_box = draw_candidate_box(full_image, list(shown_box))
        return [boxed_image], BOX_TEMPLATE, rendered_box, True, "random_rendered"
    if intervention == "bbox_crop":
        crop = padded_crop(full_image, source_box)
        return [full_image, crop], CROP_TEMPLATE, source_box, True, "crop_rendered"
    raise ValueError(f"Unsupported intervention: {intervention}")


def generate_one(model, processor, images, prompt: str, max_new_tokens: int) -> str:
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True
        )
    generated_only = generated_ids[:, inputs.input_ids.shape[1] :]
    answer = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    del inputs, generated_ids, generated_only
    return answer


def main() -> None:
    args = parse_args()
    rows, grounding = validate_inputs(args)
    completed = completed_question_ids(args.output, args.retry_errors)
    pending = [row for row in rows if row.question_id not in completed]
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"intervention={args.intervention} subset={args.subset} rows={len(rows)} "
        f"completed={len(completed)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        print("nothing_to_do=true", flush=True)
        return

    model, processor = load_model(args)
    consecutive_errors = 0
    success_count = 0
    error_count = 0
    run_started = time.perf_counter()
    for position, row in enumerate(pending, start=1):
        candidate = grounding[row.question_id]
        started = time.perf_counter()
        try:
            with Image.open(args.image_dir / row.image) as image:
                images, template, shown_box, has_box, box_status = prepare_intervention(
                    image,
                    candidate,
                    args.intervention,
                    args.seed * 1_000_003 + row.question_id,
                )
            prompt = template.format(
                question=row.question, object_phrase=candidate["object_phrase"]
            )
            raw_answer = generate_one(
                model, processor, images, prompt, args.max_new_tokens
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
                    "shown_box": list(shown_box) if shown_box else None,
                    "has_box": has_box,
                    "box_status": box_status,
                    "raw_answer": raw_answer,
                    "pred_answer": prediction,
                    "intervention": args.intervention,
                    "setting": args.setting,
                    "model_id": args.model_id,
                    "status": "ok",
                    "inference_seconds": round(elapsed, 4),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            consecutive_errors = 0
            success_count += 1
            if position <= 10 or position % 25 == 0:
                print(
                    f"progress={position}/{len(pending)} qid={row.question_id} "
                    f"label={row.label} pred={prediction} raw={raw_answer!r} sec={elapsed:.2f}",
                    flush=True,
                )
        except Exception as error:
            elapsed = time.perf_counter() - started
            append_jsonl(
                args.output,
                {
                    "question_id": row.question_id,
                    "image": row.image,
                    "question": row.question,
                    "label": row.label,
                    "intervention": args.intervention,
                    "setting": args.setting,
                    "model_id": args.model_id,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "inference_seconds": round(elapsed, 4),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            consecutive_errors += 1
            error_count += 1
            print(
                f"ERROR qid={row.question_id} {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            if consecutive_errors >= args.max_consecutive_errors:
                raise RuntimeError(
                    f"Stopping after {consecutive_errors} consecutive errors; rerun with --retry-errors"
                ) from error
        finally:
            if position % 25 == 0:
                gc.collect()

    if success_count == 0 and error_count > 0:
        raise RuntimeError(f"All {error_count} attempted rows failed; inspect {args.output}")
    total_elapsed = time.perf_counter() - run_started
    print(
        f"run_complete=true processed={len(pending)} successes={success_count} "
        f"errors={error_count} total_seconds={total_elapsed:.1f} "
        f"python={platform.python_version()} torch={torch.__version__}",
        flush=True,
    )


if __name__ == "__main__":
    main()
