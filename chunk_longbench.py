#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LongBench 数据集切块工具。

作用：
1. 读取 `data/longbench/<dataset>_e.jsonl` 原始样本。
2. 使用指定模型的 tokenizer 对 context 做 token 级切分。
3. 当 context 超过 `max_length` 时，按头尾保留策略截断。
4. 输出下游评测可直接使用的 chunk 数据文件和对应切分日志。

可用示例：
    python ./chunk_longbench.py \
      --model /data/ykw/models/Meta-Llama-3.1-8B-Instruct \
      --dataset 2wikimqa \
      --max_length 15000 \
      --chunk_size 512 \
      --output_root ./inputs

    python ./chunk_longbench.py \
      --model /data/ykw/models/Meta-Llama-3.1-8B-Instruct \
      --dataset all \
      --max_length 15000 \
      --chunk_size 512 \
      --output_root ./inputs
"""

import argparse
import json
import os
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# precompute.py 当前支持的 LongBench 数据集列表。
LONG_BENCH_PRECOMPUTE_DATASETS = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "lcc",
]


DEFAULT_MODEL = "/data/ykw/models/Meta-Llama-3-8B-Instruct"
DEFAULT_DATASET = "2wikimqa"
DEFAULT_MAX_LENGTH = 15000
DEFAULT_CHUNK_SIZE = 512
DEFAULT_OUTPUT_ROOT = "./inputs"
DEFAULT_DATA_ROOT = "./data/longbench"


def parse_args() -> argparse.Namespace:
    # 保持旧默认行为，同时允许通过 CLI 覆盖参数。
    parser = argparse.ArgumentParser(
        description="Chunk LongBench contexts into fixed-size segments."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="Model path/id for tokenizer."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset name, comma-separated datasets, or 'all'.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Max context tokens before head-tail truncation.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size in tokens.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for chunk json files.",
    )
    return parser.parse_args()


def parse_dataset_arg(dataset_arg: str, available: List[str]) -> List[str]:
    # 支持单个数据集、逗号分隔列表、或 all。
    if dataset_arg == "all":
        missing = [d for d in LONG_BENCH_PRECOMPUTE_DATASETS if d not in available]
        if missing:
            raise ValueError(f"missing dataset templates in config: {missing}")
        return LONG_BENCH_PRECOMPUTE_DATASETS

    datasets = []
    seen = set()
    for raw in dataset_arg.split(","):
        dataset = raw.strip()
        if not dataset:
            continue
        if dataset not in available:
            raise ValueError(
                f"unsupported dataset '{dataset}'. available: {sorted(available)}"
            )
        if dataset not in seen:
            datasets.append(dataset)
            seen.add(dataset)

    if not datasets:
        raise ValueError("--dataset resolved to empty set")
    return datasets


def chunk_single_dataset(
    dataset: str,
    tokenizer: AutoTokenizer,
    questions_format: dict,
    max_length: int,
    chunk_size: int,
    data_root: str,
    output_dir: str,
) -> None:
    # 将每条原始样本切块，并保存为下游加载器可直接使用的字段结构。
    question_format = questions_format[dataset]
    chunk_data = []
    chunk_log = []

    input_file = os.path.join(data_root, f"{dataset}_e.jsonl")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"missing input file: {input_file}")

    with open(input_file, "r", encoding="utf-8") as file:
        for item_idx, line in enumerate(file):
            item = json.loads(line)
            chunk_list = []
            chunk_lengths = []

            question = question_format.format(**item)
            context = item["context"]
            answers = item["answers"]
            classes = item.get("all_classes", [])

            tokenized_context = tokenizer(
                context, truncation=False, return_tensors="pt"
            ).input_ids[0]
            original_context_tokens = int(tokenized_context.shape[0])
            truncated = original_context_tokens > max_length

            if truncated:
                # 头尾截断：同时保留开头和结尾语义。
                half = int(max_length / 2)
                tokenized_context = torch.cat(
                    [tokenized_context[:half], tokenized_context[-half:]],
                    dim=0,
                )
            kept_context_tokens = int(tokenized_context.shape[0])

            for i in range(0, tokenized_context.shape[0], chunk_size):
                # 先在 token 维度分块，再解码回文本，兼容现有 loader。
                chunk_tokens = tokenized_context[i : i + chunk_size]
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunk_list.append(chunk_text)
                chunk_lengths.append(int(chunk_tokens.shape[0]))

            chunk_data.append(
                {
                    "chunks": chunk_list,
                    "question": question,
                    "answers": answers,
                    "all_classes": classes,
                }
            )

            chunk_log.append(
                {
                    "item_idx": item_idx,
                    "original_context_tokens": original_context_tokens,
                    "kept_context_tokens": kept_context_tokens,
                    "max_length": max_length,
                    "chunk_size": chunk_size,
                    "num_chunks": len(chunk_lengths),
                    "chunk_token_lengths": chunk_lengths,
                    "truncated": truncated,
                    "truncation_strategy": "head_tail" if truncated else None,
                    "discarded_context_tokens": (
                        original_context_tokens - kept_context_tokens
                        if truncated
                        else 0
                    ),
                    "question_tokens": int(
                        tokenizer(
                            question, truncation=False, return_tensors="pt"
                        ).input_ids.shape[-1]
                    ),
                    "answer_count": len(answers),
                }
            )

    output_file = os.path.join(output_dir, f"{dataset}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=4, ensure_ascii=False)

    log_dir = os.path.join(output_dir, "chunk_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{dataset}_chunk_log.json")
    log_payload = {
        "dataset": dataset,
        "model": tokenizer.name_or_path,
        "max_length": max_length,
        "chunk_size": chunk_size,
        "field_descriptions": {
            "dataset": "当前日志对应的 LongBench 数据集名称。",
            "model": "用于切分文本的 tokenizer 对应模型路径或模型名。",
            "max_length": "上下文允许保留的最大 token 数。超过该值时会触发截断。",
            "chunk_size": "上下文分块时每个 chunk 的目标 token 数上限。",
            "summary": {
                "total_samples": "该数据集总样本数。",
                "truncated_samples": "原始上下文长度超过 max_length、因此发生截断的样本数。",
                "max_original_context_tokens": "该数据集中原始 context 的最大 token 长度。",
                "max_kept_context_tokens": "截断后实际保留的最大 context token 长度，通常不超过 max_length。",
                "max_num_chunks": "该数据集中单条样本切分后得到的最大 chunk 数。",
            },
            "items": {
                "item_idx": "样本在原始数据集中的下标，从 0 开始。",
                "original_context_tokens": "原始 context 在截断前的 token 数。",
                "kept_context_tokens": "参与分块的 context token 数；若发生截断，则为截断后的长度。",
                "max_length": "本次切分使用的最大 context token 长度限制。",
                "chunk_size": "本次切分使用的 chunk token 长度上限。",
                "num_chunks": "该样本最终被切成的 chunk 数量。",
                "chunk_token_lengths": "每个 chunk 的实际 token 长度列表，顺序与输出 chunks 一一对应。",
                "truncated": "该样本是否因超过 max_length 而被截断。",
                "truncation_strategy": "发生截断时使用的策略；当前为保留头尾的 head_tail。",
                "discarded_context_tokens": "因截断被丢弃的 context token 数；未截断时为 0。",
                "question_tokens": "对应 question 文本的 token 数，仅作参考统计。",
                "answer_count": "该样本标准答案的数量。",
            },
        },
        "summary": {
            "total_samples": len(chunk_log),
            "truncated_samples": sum(1 for item in chunk_log if item["truncated"]),
            "max_original_context_tokens": max(
                (item["original_context_tokens"] for item in chunk_log), default=0
            ),
            "max_kept_context_tokens": max(
                (item["kept_context_tokens"] for item in chunk_log), default=0
            ),
            "max_num_chunks": max(
                (item["num_chunks"] for item in chunk_log), default=0
            ),
        },
        "items": chunk_log,
    }
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=4, ensure_ascii=False)

    print(
        f"[chunk] done dataset={dataset} samples={len(chunk_data)} "
        f"output={output_file} log={log_file}"
    )


def main() -> None:
    args = parse_args()

    if args.max_length <= 0:
        raise ValueError("--max_length must be > 0")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be > 0")

    # question_format 定义每个数据集的问题拼接模板。
    questions_format = json.load(open("./config/longbench/question_format.json", "r"))
    available_datasets = sorted(questions_format.keys())
    datasets = parse_dataset_arg(args.dataset, available_datasets)

    # tokenizer 必须与后续 precompute/eval 所用模型一致。
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    output_dir = os.path.join(args.output_root, os.path.basename(args.model))
    os.makedirs(output_dir, exist_ok=True)

    print("[chunk] start")
    print(f"[chunk] model={args.model}")
    print(f"[chunk] datasets={datasets}")
    print(f"[chunk] max_length={args.max_length} chunk_size={args.chunk_size}")
    print(f"[chunk] output_dir={output_dir}")

    for dataset in tqdm(datasets, desc="chunk datasets"):
        chunk_single_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            questions_format=questions_format,
            max_length=args.max_length,
            chunk_size=args.chunk_size,
            data_root=DEFAULT_DATA_ROOT,
            output_dir=output_dir,
        )

    print(f"[chunk] all done datasets={len(datasets)} output_dir={output_dir}")


if __name__ == "__main__":
    main()
