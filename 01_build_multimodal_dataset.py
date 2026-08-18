#!/usr/bin/env python3
"""构建多模态 conversations 格式数据集（msModelSpec-Dev prepare_data.py 输入格式）。

支持的原始数据：
1. parquet，包含 image(bytes)/text 列（如 human_handwrite 手写公式数据）
2. jsonl，每行 {"image": "/abs/path.jpg", "prompt": "...", "response": "..."}
   （可自行扩展 video/audio: {"type": "video", "path": ...}）

输出：
  ${DATASET_DIR}/conversations.jsonl  每行 {"conversations": [...]}
  ${DATASET_DIR}/images/              从 parquet 导出的图像

conversations 格式（见 msModelSpec-Dev src/speculators/data_generation/preprocessing.py:
_normalize_conversation / _adapt_part_for_vllm）：
  用户多模态 content 为列表: [{"type":"text","text":...}, {"type":"image","path":...}]
"""
import argparse
import base64
import io
import json
import os


def export_parquet(parquet_path: str, out_dir: str, instruction: str, limit: int | None):
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    n = len(df) if limit is None else min(limit, len(df))
    for i in range(n):
        r = df.iloc[i]
        img, text = r["image"], r["text"]
        img_bytes = img["bytes"] if isinstance(img, dict) else img
        if isinstance(img_bytes, str):  # base64 字符串
            img_bytes = base64.b64decode(img_bytes)
        ext = "png" if img_bytes[:4] == b"\x89PNG" else "jpg"
        img_path = os.path.abspath(os.path.join(img_dir, f"sample_{i:06d}.{ext}"))
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        rows.append(
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": [
                            {"type": "image", "path": img_path},
                            {"type": "text", "text": instruction},
                        ],
                    },
                    {"from": "gpt", "value": str(text)},
                ]
            }
        )
    return rows


def load_jsonl(jsonl_path: str, limit: int | None):
    rows = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            d = json.loads(line)
            user_parts = []
            if d.get("image"):
                user_parts.append({"type": "image", "path": os.path.abspath(d["image"])})
            if d.get("video"):
                user_parts.append({"type": "video", "path": os.path.abspath(d["video"])})
            user_parts.append({"type": "text", "text": d.get("prompt", "描述这个输入。")})
            rows.append(
                {
                    "conversations": [
                        {"from": "human", "value": user_parts},
                        {"from": "gpt", "value": d["response"]},
                    ]
                }
            )
    return rows


def is_json_array(path: str) -> bool:
    if not path.endswith(".json"):
        return False
    with open(path) as f:
        return f.read(1).lstrip().startswith("[")


def normalize_conversations_array(json_path: str, out_dir: str, limit: int | None):
    """加载 json 数组（如 ALLaVA .mm.json），把所有 content/value 统一为列表类型。

    Arrow 加载 conversations 列时无法混合 str 和 list 类型
    ("cannot mix list and non-list, non-null values")，因此把字符串
    value 包装为 [{"type":"text","text": ...}]，并输出为 jsonl。
    """
    with open(json_path) as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    n_img = 0
    with open(os.path.join(out_dir, "conversations.jsonl"), "w") as f:
        for item in data:
            conv = item["conversations"]
            for turn in conv:
                v = turn.get("value", turn.get("content", ""))
                if isinstance(v, str):
                    turn["value"] = [{"type": "text", "text": v}]
                else:
                    for part in v:
                        if part.get("type") in ("image", "video", "audio"):
                            part["path"] = os.path.abspath(part["path"])
                            n_img += 1
                turn.pop("content", None)
            f.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")
    print(f"normalized {len(data)} samples ({n_img} media parts) -> conversations.jsonl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="parquet/jsonl/json 数组路径")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--instruction",
        default="请识别图中的手写内容，并用 LaTeX 表示。",
        help="用户文本指令（parquet 数据无 prompt 列时使用）",
    )
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if args.input.endswith(".parquet"):
        rows = export_parquet(args.input, args.output_dir, args.instruction, args.limit)
    elif args.input.endswith(".mm.json") or is_json_array(args.input):
        normalize_conversations_array(args.input, args.output_dir, args.limit)
        return
    else:
        rows = load_jsonl(args.input, args.limit)

    out = os.path.join(args.output_dir, "conversations.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} samples -> {out}")


if __name__ == "__main__":
    main()
