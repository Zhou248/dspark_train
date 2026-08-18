#!/usr/bin/env python3
"""Request one prepared sample repeatedly and diagnose verifier hidden states."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import unquote, urlparse

import openai
from datasets import load_from_disk
from PIL import Image
from safetensors.torch import load_file
from speculators.data_generation.vllm_client import generate_hidden_states
from speculators.train.data import build_client_item

from repair_hidden_states import check_hidden_states


def describe_images(messages: list[dict]) -> None:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            image_url = part.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url", "")
            parsed = urlparse(url)
            if parsed.scheme != "file":
                print(f"image={url}")
                continue
            path = Path(unquote(parsed.path))
            with Image.open(path) as image:
                print(
                    f"image={path} size={image.size} mode={image.mode} "
                    f"format={image.format} bytes={path.stat().st_size}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", required=True)
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--layer-ids", type=int, nargs="+", default=[2, 20, 37, 40]
    )
    args = parser.parse_args()

    dataset = load_from_disk(args.prepared_data)
    row = dataset[args.index]
    item = build_client_item(row)
    print(
        f"index={args.index} seq_len={len(item['input_ids'])} "
        f"assistant_tokens={int(row['loss_mask'].sum())}"
    )
    if "messages" in item:
        describe_images(item["messages"])

    client = openai.OpenAI(base_url=args.endpoint, api_key="EMPTY", max_retries=0)
    for attempt in range(1, args.repeat + 1):
        handle: str | None = None
        try:
            handle = generate_hidden_states(
                client,
                args.model,
                item,
                timeout=600,
                max_retries=0,
            )
            data = load_file(handle)
            hidden_states = data["hidden_states"]
            print(
                f"attempt={attempt} handle={handle} shape={list(hidden_states.shape)}"
            )
            for slot in range(hidden_states.shape[1]):
                layer_id = (
                    args.layer_ids[slot] if slot < len(args.layer_ids) else slot
                )
                layer = hidden_states[:, slot]
                print(
                    f"  slot={slot} layer={layer_id} "
                    f"nan={int(layer.isnan().sum())} inf={int(layer.isinf().sum())} "
                    f"absmax={float(layer.nan_to_num().abs().max()):.6g}"
                )
            check_hidden_states(data, item["input_ids"])
            print(f"attempt={attempt} VALID")
        except Exception as exc:  # noqa: BLE001 - diagnostic must print all failures
            print(f"attempt={attempt} ERROR {type(exc).__name__}: {exc}")
        finally:
            if handle is not None:
                Path(handle).unlink(missing_ok=True)
                Path(f"{handle}.lock").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
