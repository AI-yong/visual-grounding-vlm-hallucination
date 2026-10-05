from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image


BASE_URL = "http://images.cocodataset.org/val2014/{filename}"


def collect_images(pope_dir: Path) -> list[str]:
    images: set[str] = set()
    for path in sorted(pope_dir.glob("coco_pope_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    images.add(json.loads(line)["image"])
    if not images:
        raise RuntimeError(f"No POPE rows found in {pope_dir}")
    return sorted(images)


def is_valid_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def download_one(filename: str, output_dir: Path, retries: int) -> tuple[str, str]:
    destination = output_dir / filename
    partial = destination.with_suffix(destination.suffix + ".part")
    if is_valid_image(destination):
        return filename, "cached"

    url = BASE_URL.format(filename=filename)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(10, 30)) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            partial.replace(destination)
            if not is_valid_image(destination):
                destination.unlink(missing_ok=True)
                raise RuntimeError("downloaded file is not a valid image")
            return filename, "downloaded"
        except Exception as error:
            last_error = error
            partial.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2**attempt)
    return filename, f"failed: {last_error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pope-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filenames = collect_images(args.pope_dir)
    print(f"required_images={len(filenames)}", flush=True)

    counts = {"downloaded": 0, "cached": 0, "failed": 0}
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, name, args.output_dir, args.retries): name
            for name in filenames
        }
        for index, future in enumerate(as_completed(futures), start=1):
            filename, status = future.result()
            category = status if status in {"downloaded", "cached"} else "failed"
            counts[category] += 1
            if category == "failed":
                failures.append((filename, status))
            if index % 25 == 0 or index == len(futures):
                print(
                    f"progress={index}/{len(futures)} "
                    f"downloaded={counts['downloaded']} cached={counts['cached']} "
                    f"failed={counts['failed']}",
                    flush=True,
                )

    if failures:
        for filename, status in failures:
            print(f"FAIL {filename}: {status}")
        raise SystemExit(f"{len(failures)} image downloads failed")
    print("all_required_images_ready=true")


if __name__ == "__main__":
    main()
