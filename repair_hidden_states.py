#!/usr/bin/env python3
"""Validate cached verifier hidden states and optionally remove invalid files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk
from safetensors.torch import load_file


def check_hidden_states(data: dict, tokens: list[int]) -> None:
    """Local validator so this workflow does not patch msModelSpec-Dev."""
    token_ids = data["token_ids"].tolist()
    if token_ids != tokens:
        raise ValueError("token IDs do not match the prepared dataset")

    hidden_states = data["hidden_states"]
    nan_mask = hidden_states.isnan()
    if nan_mask.any():
        first = nan_mask.nonzero()[0].tolist()
        details = ""
        if hidden_states.ndim == 3:
            counts = nan_mask.sum(dim=(0, 2)).tolist()
            bad_slots = {i: int(n) for i, n in enumerate(counts) if n}
            details = f", extracted-layer slots={bad_slots}"
        raise ValueError(
            f"{int(nan_mask.sum())} NaN values; "
            f"shape={list(hidden_states.shape)}, first={first}{details}"
        )
    if hidden_states.isinf().any():
        raise ValueError("hidden states contain Inf values")
    if hidden_states.shape[0] != len(tokens):
        raise ValueError(
            f"hidden-state length {hidden_states.shape[0]} != token length {len(tokens)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", required=True)
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument("--delete-invalid", action="store_true")
    args = parser.parse_args()

    dataset = load_from_disk(args.prepared_data)
    hidden_states_dir = Path(args.hidden_states)
    paths = sorted(hidden_states_dir.glob("hs_*.safetensors"))
    invalid = 0

    for path in paths:
        index: int | None = None
        try:
            index = int(path.stem.removeprefix("hs_"))
            if index < 0 or index >= len(dataset):
                raise IndexError(
                    f"sample index {index} is outside dataset size {len(dataset)}"
                )
            row = dataset[index]
            check_hidden_states(load_file(path), row["input_ids"])
        except Exception as exc:  # noqa: BLE001 - this tool must repair corrupt files
            invalid += 1
            print(f"INVALID {path.name}: {exc}")
            if index is not None and 0 <= index < len(dataset):
                messages = dataset[index].get("messages")
                if messages is not None:
                    print(
                        "  messages=",
                        json.dumps(messages, ensure_ascii=False)[:2000],
                    )
            if args.delete_invalid:
                path.unlink(missing_ok=True)
                print(f"  deleted={path}")

    print(
        f"checked={len(paths)} invalid={invalid} "
        f"deleted={invalid if args.delete_invalid else 0}"
    )


if __name__ == "__main__":
    main()
