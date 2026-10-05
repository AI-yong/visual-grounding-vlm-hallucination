from __future__ import annotations

import argparse
import gc
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.data.pope import load_pope_rows, normalize_binary_answer
from src.utils.jsonl import append_jsonl, completed_question_ids


PROMPT_TEMPLATE = """Answer the following question using only the visible content of the image.
Respond with exactly one word: Yes or No.

Question: {question}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable Qwen2.5-VL baseline on a POPE JSONL file."
    )
    parser.add_argument("--pope-file", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
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


def validate_inputs(args: argparse.Namespace) -> list:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start a long CPU run.")
    rows = load_pope_rows(args.pope_file)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    missing = [row.image for row in rows if not (args.image_dir / row.image).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} referenced images are missing. First: {preview}")
    return rows


def load_model(args: argparse.Namespace):
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, processor


def generate_one(model, processor, image_path: Path, question: str, max_new_tokens: int) -> str:
    prompt = PROMPT_TEMPLATE.format(question=question)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
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
    input_length = inputs.input_ids.shape[1]
    generated_only = generated_ids[:, input_length:]
    answer = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    del inputs, generated_ids, generated_only
    return answer


def main() -> None:
    args = parse_args()
    rows = validate_inputs(args)
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
        image_path = args.image_dir / row.image
        started = time.perf_counter()
        try:
            with Image.open(image_path) as image:
                image.verify()
            raw_answer = generate_one(
                model, processor, image_path, row.question, args.max_new_tokens
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
                    "raw_answer": raw_answer,
                    "pred_answer": prediction,
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
                    f"label={row.label} pred={prediction} raw={raw_answer!r} "
                    f"sec={elapsed:.2f}",
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
            observed_in_window += 1  # Evaluate this guard only once.

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
