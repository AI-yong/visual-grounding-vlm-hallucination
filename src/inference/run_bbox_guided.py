from __future__ import annotations

import argparse
import gc
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info

from src.data.pope import load_pope_rows, normalize_binary_answer
from src.inference.run_mllm import load_model
from src.utils.jsonl import append_jsonl, completed_question_ids, iter_jsonl


GUIDED_PROMPT_TEMPLATE = """Answer the question using only visible image evidence.
The red rectangle is a candidate region proposed by an object detector for "{object_phrase}".
Verify the pixels inside and around the rectangle. The rectangle alone does not prove that the object exists.
Respond with exactly one word: Yes or No.

Question: {question}"""

NO_BOX_PROMPT_TEMPLATE = """Answer the question using only visible image evidence.
No usable detector candidate region is available for "{object_phrase}".
This detector result can be wrong. Inspect the full image and make the final decision yourself.
Respond with exactly one word: Yes or No.

Question: {question}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable bbox-guided Qwen2.5-VL inference on POPE."
    )
    parser.add_argument("--pope-file", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--sanity-window", type=int, default=10)
    parser.add_argument("--max-invalid-ratio", type=float, default=0.5)
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    return parser.parse_args()


def latest_grounding_rows(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["question_id"]): row for row in iter_jsonl(path)}


def clamp_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError(f"Expected four box coordinates, got {box}")
    x1, y1, x2, y2 = (float(value) for value in box)
    left = max(0, min(width - 1, round(min(x1, x2))))
    top = max(0, min(height - 1, round(min(y1, y2))))
    right = max(0, min(width - 1, round(max(x1, x2))))
    bottom = max(0, min(height - 1, round(max(y1, y2))))
    if right <= left or bottom <= top:
        raise ValueError(f"Degenerate box after clamping: {(left, top, right, bottom)}")
    return left, top, right, bottom


def draw_candidate_box(image: Image.Image, box: list[float]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    rendered = image.convert("RGB").copy()
    clamped = clamp_box(box, *rendered.size)
    line_width = max(3, round(min(rendered.size) * 0.008))
    draw = ImageDraw.Draw(rendered)
    for offset in range(line_width):
        left, top, right, bottom = clamped
        expanded = (
            max(0, left - offset),
            max(0, top - offset),
            min(rendered.width - 1, right + offset),
            min(rendered.height - 1, bottom + offset),
        )
        draw.rectangle(expanded, outline=(255, 0, 0), width=1)
    return rendered, clamped


def validate_inputs(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start a long CPU run.")
    rows = load_pope_rows(args.pope_file)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    grounding = latest_grounding_rows(args.grounding)
    for row in rows:
        if not (args.image_dir / row.image).is_file():
            raise FileNotFoundError(args.image_dir / row.image)
        candidate = grounding.get(row.question_id)
        if candidate is None:
            raise ValueError(f"Missing grounding row for question {row.question_id}")
        if candidate.get("status", "ok") != "ok":
            raise ValueError(f"Question {row.question_id} has unusable grounding output")
        if candidate.get("image") != row.image or candidate.get("label") != row.label:
            raise ValueError(f"Question {row.question_id} metadata differs from grounding")
    return rows, grounding


def generate_one(
    model,
    processor,
    boxed_image: Image.Image,
    question: str,
    object_phrase: str,
    has_box: bool,
    max_new_tokens: int,
) -> str:
    template = GUIDED_PROMPT_TEMPLATE if has_box else NO_BOX_PROMPT_TEMPLATE
    prompt = template.format(
        question=question, object_phrase=object_phrase
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": boxed_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
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
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
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
    print(f"rows={len(rows)} completed={len(completed)} pending={len(pending)}", flush=True)
    if not pending:
        print("nothing_to_do=true", flush=True)
        return

    model, processor = load_model(args)
    invalid_in_window = 0
    observed_in_window = 0
    consecutive_errors = 0
    success_count = 0
    error_count = 0
    run_started = time.perf_counter()

    for position, row in enumerate(pending, start=1):
        candidate = grounding[row.question_id]
        image_path = args.image_dir / row.image
        started = time.perf_counter()
        try:
            with Image.open(image_path) as image:
                if candidate.get("best_box") is None:
                    boxed_image = image.convert("RGB").copy()
                    clamped_box = None
                    box_status = "missing"
                else:
                    try:
                        boxed_image, clamped_box = draw_candidate_box(
                            image, candidate["best_box"]
                        )
                        box_status = "rendered"
                    except ValueError:
                        boxed_image = image.convert("RGB").copy()
                        clamped_box = None
                        box_status = "invalid_after_clamp"
            has_box = clamped_box is not None
            raw_answer = generate_one(
                model,
                processor,
                boxed_image,
                row.question,
                str(candidate["object_phrase"]),
                has_box,
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
                    "best_box": candidate["best_box"],
                    "rendered_box": list(clamped_box) if clamped_box else None,
                    "has_box": has_box,
                    "box_status": box_status,
                    "raw_answer": raw_answer,
                    "pred_answer": prediction,
                    "intervention": "bbox_overlay",
                    "setting": args.setting,
                    "model_id": args.model_id,
                    "status": "ok",
                    "inference_seconds": round(elapsed, 4),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            consecutive_errors = 0
            success_count += 1
            if observed_in_window < args.sanity_window:
                observed_in_window += 1
                invalid_in_window += int(prediction == "invalid")
            if position <= args.sanity_window or position % 25 == 0:
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
                    "intervention": "bbox_overlay",
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
            gc.collect()

        if observed_in_window == args.sanity_window:
            invalid_ratio = invalid_in_window / observed_in_window
            if invalid_ratio > args.max_invalid_ratio:
                raise RuntimeError(
                    f"Sanity guard stopped the run: invalid_ratio={invalid_ratio:.3f} "
                    f"> {args.max_invalid_ratio:.3f}"
                )
            observed_in_window += 1

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
