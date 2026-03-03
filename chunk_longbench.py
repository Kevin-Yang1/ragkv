#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LongBench 数据集切块工具。"""

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
DEFAULT_MAX_LENGTH = 7000
DEFAULT_CHUNK_SIZE = 512
DEFAULT_OUTPUT_ROOT = "./inputs"
DEFAULT_DATA_ROOT = "./data/longbench"


def parse_args() -> argparse.Namespace:
    # 保持旧默认行为，同时允许通过 CLI 覆盖参数。
    parser = argparse.ArgumentParser(description="Chunk LongBench contexts into fixed-size segments.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model path/id for tokenizer.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset name, comma-separated datasets, or 'all'.",
    )
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH, help="Max context tokens before head-tail truncation.")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in tokens.")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Output root for chunk json files.")
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

    input_file = os.path.join(data_root, f"{dataset}_e.jsonl")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"missing input file: {input_file}")

    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            chunk_list = []

            question = question_format.format(**item)
            context = item["context"]
            answers = item["answers"]
            classes = item.get("all_classes", [])

            tokenized_context = tokenizer(
                context, truncation=False, return_tensors="pt"
            ).input_ids[0]

            if len(tokenized_context) > max_length:
                # 头尾截断：同时保留开头和结尾语义。
                half = int(max_length / 2)
                tokenized_context = torch.cat(
                    [tokenized_context[:half], tokenized_context[-half:]],
                    dim=0,
                )

            for i in range(0, tokenized_context.shape[0], chunk_size):
                # 先在 token 维度分块，再解码回文本，兼容现有 loader。
                chunk_tokens = tokenized_context[i : i + chunk_size]
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunk_list.append(chunk_text)

            chunk_data.append(
                {
                    "chunks": chunk_list,
                    "question": question,
                    "answers": answers,
                    "all_classes": classes,
                }
            )

    output_file = os.path.join(output_dir, f"{dataset}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=4, ensure_ascii=False)

    print(f"[chunk] done dataset={dataset} samples={len(chunk_data)} output={output_file}")


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
