#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
LongBench 数据集分块脚本
=============================================================================
功能：将 LongBench 数据集的长上下文分割成固定大小的 chunks
用途：为预计算 KV Cache 做准备，将长文档切分便于处理
输出：JSON 格式的分块数据，包含 chunks、问题、答案等信息

使用方法：
    python chunk_longbench.py

输出位置：
    ./inputs/{model_name}/{dataset}.json

依赖：
    - transformers
    - torch
    - tqdm

作者：参考 A³ 项目
=============================================================================
"""

import json
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import os

# =============================================================================
# 配置参数
# =============================================================================

# 最大上下文长度（tokens）
# 如果文档超过此长度，会截取开头和结尾各一半
# MAXLENGTH = 3500  # 适用于 Llama-3（8K 上下文）
MAXLENGTH = 7000  # 适用于更长上下文的模型（如 Qwen2.5）

# 分块大小（tokens）
# 每个 chunk 包含的 token 数量
CHUNKSIZE = 512

# =============================================================================
# 数据集列表
# =============================================================================
# LongBench 基准测试中的 11 个数据集
# 涵盖问答、摘要、分类等多种长上下文任务
datasets = [
    "2wikimqa",  # Wikipedia 多跳问答
    # "qasper",          # 科学论文问答
    # "multifieldqa_en", # 多领域问答（英文）
    # "hotpotqa",        # 多跳推理问答
    # "gov_report",      # 政府报告摘要
    # "multi_news",      # 多文档新闻摘要
    # "trec",            # TREC 问题分类
    # "triviaqa",        # 琐事问答
    # "samsum",          # 对话摘要
    # "passage_count",   # 段落计数
    # "lcc",             # 长上下文代码补全
]

# =============================================================================
# 模型路径配置
# =============================================================================
# 选择用于 tokenization 的模型
# 不同模型的 tokenizer 可能产生不同的 token 化结果
# path = '../Models/LLMs/Qwen/Qwen2.5-7B-Instruct'
path = "/data/ykw/models/Meta-Llama-3-8B-Instruct"
# path = '../Models/LLMs/Mistral-7B-Instruct-v0.2'

# 注意：path 必须与后续 precompute 和 eval 中使用的模型一致

# =============================================================================
# 加载配置和 Tokenizer
# =============================================================================
# 加载问题格式模板
# question_format.json 定义了每个数据集的问题格式化方式
questions_format = json.load(open("./config/longbench/question_format.json", "r"))

# 加载对应模型的 tokenizer
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

# =============================================================================
# 输出路径配置
# =============================================================================
# 输出目录：保存分块后的数据
# 格式：inputs/{model_name}/{dataset}.json
# chunk_path = 'inputs/Qwen2.5-7B-Instruct'
chunk_path = "inputs/Meta-Llama-3-8B-Instruct"
# chunk_path = 'inputs/Mistral-7B-Instruct-v0.2'

# 创建输出目录（如果不存在）
os.makedirs(chunk_path, exist_ok=True)

# =============================================================================
# 主处理循环
# =============================================================================
print("开始处理 LongBench 数据集...")
print(f"最大长度: {MAXLENGTH} tokens")
print(f"分块大小: {CHUNKSIZE} tokens")
print(f"输出目录: {chunk_path}")
print("-" * 80)

for dataset in tqdm(datasets, desc="处理数据集"):
    # 获取当前数据集的问题格式模板
    question_format = questions_format[dataset]
    chunk_data = []

    # 读取原始数据文件
    # 文件格式：每行一个 JSON 对象（JSONL）
    input_file = f"./data/longbench/{dataset}_e.jsonl"

    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            # 解析每一行的 JSON 数据
            item = json.loads(line)

            # -------------------------------------------------------------------------
            # 数据结构说明：
            # item = {
            #     'context': str,        # 长上下文文本
            #     'answers': list,       # 标准答案列表
            #     'all_classes': list,   # 分类任务的所有类别
            #     其他字段...             # 用于格式化问题的字段
            # }
            # -------------------------------------------------------------------------

            chunk_list = []

            # 使用模板格式化问题
            # 例如："Question: {input}\nAnswer:"
            question = question_format.format(**item)

            # 提取答案和分类信息
            context = item["context"]
            answers = item["answers"]
            classes = item["all_classes"]

            # -------------------------------------------------------------------------
            # Tokenization 和截断处理
            # -------------------------------------------------------------------------
            # 将文本转换为 token IDs
            tokenized_context = tokenizer(
                context, truncation=False, return_tensors="pt"
            ).input_ids[0]

            # 如果超过最大长度，进行截断
            if len(tokenized_context) > MAXLENGTH:
                # 策略：保留开头和结尾各一半
                # 这样可以保留文档的开始和结束部分，通常包含重要信息
                half = int(MAXLENGTH / 2)
                tokenized_context = torch.cat(
                    [
                        tokenized_context[:half],  # 前半部分
                        tokenized_context[-half:],  # 后半部分
                    ],
                    dim=0,
                )

            # -------------------------------------------------------------------------
            # 分块处理
            # -------------------------------------------------------------------------
            # 将 tokenized context 分割成固定大小的 chunks
            # 例如：CHUNKSIZE=512，则每个 chunk 包含 512 个 tokens
            for i in range(0, tokenized_context.shape[0], CHUNKSIZE):
                # 提取当前 chunk 的 token IDs
                chunk_tokens = tokenized_context[i : i + CHUNKSIZE]

                # 解码回文本
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

                # 添加到 chunk 列表
                chunk_list.append(chunk_text)

            # -------------------------------------------------------------------------
            # 保存分块数据
            # -------------------------------------------------------------------------
            # 数据结构：
            # {
            #     'chunks': [chunk1, chunk2, ...],  # 分块后的文本列表
            #     'question': str,                   # 格式化后的问题
            #     'answers': list,                   # 标准答案
            #     'all_classes': list                # 分类类别
            # }
            chunk_data.append(
                {
                    "chunks": chunk_list,
                    "question": question,
                    "answers": answers,
                    "all_classes": classes,
                }
            )

    # 保存到文件
    output_file = os.path.join(chunk_path, f"{dataset}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=4, ensure_ascii=False)

    # tqdm 会自动显示进度条

print("-" * 80)
print(f"✓ 所有数据集处理完成！")
print(f"✓ 输出目录: {chunk_path}")
print(f"✓ 共处理 {len(datasets)} 个数据集")

# =============================================================================
# 使用说明
# =============================================================================
"""
1. 基本使用：
   python chunk_longbench.py

2. 修改配置：
   - 调整 MAXLENGTH：修改第 26 行（根据模型上下文长度）
   - 调整 CHUNKSIZE：修改第 30 行（根据实验需求）
   - 切换模型：修改第 52 行和第 60 行

3. 自定义数据集：
   在 datasets 列表（第 39-50 行）中添加或删除数据集

4. 输出结构：
   inputs/
   └── Qwen2.5-7B-Instruct/
       ├── qasper.json
       ├── hotpotqa.json
       └── ...

5. 下一步：
   运行 scripts/precompute.sh 预计算 KV Cache

6. 常见问题：
   Q: 为什么要分块？
   A: 长文档分块后可以分段预计算 KV Cache，便于管理和复用

   Q: MAXLENGTH 和 CHUNKSIZE 如何设置？
   A: MAXLENGTH 应小于模型最大上下文长度
      CHUNKSIZE 一般设置为 512 或 1024

   Q: 可以用不同模型的 tokenizer 吗？
   A: 可以，但必须与后续实验使用的模型一致
"""
