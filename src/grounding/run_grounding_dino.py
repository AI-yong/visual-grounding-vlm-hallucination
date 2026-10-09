from __future__ import annotations

import argparse
import gc
import math
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from src.data.pope import PopeRow, extract_object_phrase, load_pope_rows
from src.utils.jsonl import append_jsonl, completed_question_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable Grounding DINO grounding on POPE questions."
    )
    parser.add_argument("--pope-file", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--postprocess-threshold", type=float, default=0.001)
    parser.add_argument("--text-threshold", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> list[PopeRow]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start a long CPU run.")
    for name in ("postprocess_threshold", "text_threshold"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    rows = load_pope_rows(args.pope_file)
    if args.limit is not None:
        rows = rows[: args.limit]
    missing = [row.image for row in rows if not (args.image_dir / row.image).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} referenced images are missing: {missing[:5]}")
    for row in rows:
        extract_object_phrase(row.question)
    return rows


def load_model(args: argparse.Namespace):
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        dtype=torch.float32,
        device_map={"": "cuda"},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, processor


def summarize_detections(
    scores: Iterable[float],
    labels: Iterable[int],
    boxes: Iterable[Iterable[float]],
    query_count: int,
) -> list[dict[str, Any]]:
    summaries = [
        {"max_score": 0.0, "best_box": None, "detection_count": 0}
        for _ in range(query_count)
    ]
    for raw_score, raw_label, raw_box in zip(scores, labels, boxes):
        score = float(raw_score)
        label = int(raw_label)
        box = [float(value) for value in raw_box]
        if not 0 <= label < query_count:
            raise ValueError(f"Grounding DINO returned out-of-range label {label}")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Grounding DINO returned invalid score {score}")
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError(f"Grounding DINO returned invalid box {box}")
        summary = summaries[label]
        summary["detection_count"] += 1
        if score > summary["max_score"]:
            summary["max_score"] = score
            summary["best_box"] = box
    return summaries


def ground_image(
    model,
    processor,
    image_path: Path,
    rows: list[PopeRow],
    postprocess_threshold: float,
    text_threshold: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    phrases = [extract_object_phrase(row.question) for row in rows]
    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB")
    summaries = []
    for start in range(0, len(phrases), batch_size):
        phrase_batch = phrases[start : start + batch_size]
        images = [image] * len(phrase_batch)
        text_queries = [[phrase] for phrase in phrase_batch]
        target_sizes = [(image.height, image.width)] * len(phrase_batch)
        inputs = processor(
            text=text_queries, images=images, return_tensors="pt", padding=True
        ).to("cuda")
        with torch.inference_mode():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=postprocess_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
        for result in results:
            scores = result["scores"].detach().cpu()
            boxes = result["boxes"].detach().cpu()
            if len(scores) == 0:
                summaries.append(
                    {"max_score": 0.0, "best_box": None, "detection_count": 0}
                )
                continue
            best_index = int(torch.argmax(scores).item())
            summaries.append(
                {
                    "max_score": float(scores[best_index].item()),
                    "best_box": [float(value) for value in boxes[best_index].tolist()],
                    "detection_count": len(scores),
                }
            )
        del inputs, outputs, results
    return [
        {"object_phrase": phrase, "query": f"{phrase}.", **summary}
        for phrase, summary in zip(phrases, summaries)
    ]


def group_rows_by_image(rows: Iterable[PopeRow]) -> dict[str, list[PopeRow]]:
    grouped: dict[str, list[PopeRow]] = defaultdict(list)
    for row in rows:
        grouped[row.image].append(row)
    return dict(grouped)


def main() -> None:
    args = parse_args()
    rows = validate_inputs(args)
    completed = completed_question_ids(args.output, args.retry_errors)
    pending = [row for row in rows if row.question_id not in completed]
    grouped = group_rows_by_image(pending)
    print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"model={args.model_id} rows={len(rows)} completed={len(completed)} "
        f"pending={len(pending)} pending_images={len(grouped)}",
        flush=True,
    )
    if not pending:
        print("nothing_to_do=true", flush=True)
        return

    model, processor = load_model(args)
    success_count = 0
    error_count = 0
    consecutive_errors = 0
    run_started = time.perf_counter()
    for image_position, (image_name, image_rows) in enumerate(grouped.items(), start=1):
        started = time.perf_counter()
        try:
            detections = ground_image(
                model,
                processor,
                args.image_dir / image_name,
                image_rows,
                args.postprocess_threshold,
                args.text_threshold,
                args.batch_size,
            )
            elapsed = time.perf_counter() - started
            for row, detection in zip(image_rows, detections):
                append_jsonl(
                    args.output,
                    {
                        "question_id": row.question_id,
                        "image": row.image,
                        "question": row.question,
                        "label": row.label,
                        **detection,
                        "setting": args.setting,
                        "model_id": args.model_id,
                        "postprocess_threshold": args.postprocess_threshold,
                        "text_threshold": args.text_threshold,
                        "status": "ok",
                        "inference_seconds": round(elapsed / len(image_rows), 4),
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            success_count += len(image_rows)
            consecutive_errors = 0
            if image_position <= 3 or image_position % 25 == 0:
                preview = ", ".join(
                    f"{item['object_phrase']}={item['max_score']:.3f}" for item in detections
                )
                print(
                    f"progress_images={image_position}/{len(grouped)} image={image_name} "
                    f"questions={len(image_rows)} sec={elapsed:.2f} scores=[{preview}]",
                    flush=True,
                )
        except Exception as error:
            elapsed = time.perf_counter() - started
            for row in image_rows:
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
                        "inference_seconds": round(elapsed / len(image_rows), 4),
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            error_count += len(image_rows)
            consecutive_errors += 1
            print(
                f"ERROR image={image_name} {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            if consecutive_errors >= args.max_consecutive_errors:
                raise RuntimeError(
                    f"Stopping after {consecutive_errors} consecutive image errors; rerun with --retry-errors"
                ) from error
        finally:
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
