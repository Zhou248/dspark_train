#!/usr/bin/env python3
"""Generate verifier hidden states with no overlap between adjacent samples."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import openai
from datasets import load_from_disk
from safetensors.torch import load_file
from speculators.data_generation.vllm_client import (
    generate_hidden_states,
    wait_for_lock,
)
from speculators.train.data import build_client_item
from tqdm import tqdm

from repair_hidden_states import check_hidden_states


def wait_until_ready(handle: str, timeout: float) -> None:
    """Wait until vLLM finishes writing the generated safetensors file."""
    lock_path = f"{handle}.lock"
    if Path(lock_path).exists():
        wait_for_lock(lock_path, timeout=timeout)


def discard_generated(handle: str | None, timeout: float) -> None:
    if handle is None:
        return
    try:
        wait_until_ready(handle, timeout)
    except Exception:  # noqa: BLE001 - preserve a live producer's files
        return
    Path(handle).unlink(missing_ok=True)


def main() -> None:  # noqa: C901 - linear command-line workflow
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--preprocessed-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-consecutive-errors", type=int, default=20)
    args = parser.parse_args()

    dataset = load_from_disk(args.preprocessed_data)
    limit = len(dataset)
    if args.max_samples is not None:
        limit = min(limit, args.max_samples)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = [i for i in range(limit) if not (output_dir / f"hs_{i}.safetensors").exists()]

    client = openai.OpenAI(base_url=args.endpoint, api_key="EMPTY", max_retries=0)
    served_model = client.models.list().data[0].id
    if served_model != args.model:
        raise ValueError(
            f"Requested model {args.model!r} does not match served model "
            f"{served_model!r}"
        )

    skipped: list[int] = []
    saved = 0
    consecutive_errors = 0
    progress = tqdm(indices, desc="Sequential hidden states")
    for index in progress:
        target = output_dir / f"hs_{index}.safetensors"
        item = build_client_item(dataset[index])
        last_error: Exception | None = None

        for attempt in range(1, args.max_retries + 2):
            handle: str | None = None
            try:
                handle = generate_hidden_states(
                    client,
                    served_model,
                    item,
                    timeout=args.request_timeout,
                    max_retries=0,
                )
                # This wait remains inside the per-sample loop. The next HTTP
                # request cannot start while the current file is still locked.
                wait_until_ready(handle, args.request_timeout)
                shutil.move(handle, target)
                check_hidden_states(load_file(target), item["input_ids"])
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - retry all sample failures
                last_error = exc
                target.unlink(missing_ok=True)
                discard_generated(handle, args.request_timeout)
                print(
                    f"sample={index} attempt={attempt}/{args.max_retries + 1} "
                    f"ERROR {type(exc).__name__}: {exc}"
                )

        if last_error is None:
            saved += 1
            consecutive_errors = 0
        else:
            skipped.append(index)
            consecutive_errors += 1
            if consecutive_errors >= args.max_consecutive_errors:
                raise RuntimeError(
                    f"Aborting after {consecutive_errors} consecutive failures; "
                    f"latest sample={index}: {last_error}"
                )
        progress.set_postfix(ok=saved, err=len(skipped))

    print(f"saved={saved} skipped={len(skipped)}")
    if skipped:
        print(f"skipped_indices={skipped}")


if __name__ == "__main__":
    main()
