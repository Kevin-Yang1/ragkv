import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import transformers
from metrics import (
    classification_score,
    code_sim_score,
    count_score,
    qa_f1_score,
    retrieval_score,
    rouge_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaConfig, MistralConfig
from models.loader import load_model

from data.longbench.loader import LongBench
from utils import *

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

METRIC_NAME = {
    "qasper": "F1",
    "multifieldqa_en": "F1",
    "hotpotqa": "F1",
    "2wikimqa": "F1",
    "gov_report": "RL",
    "multi_news": "RL",
    "trec": "Acc",
    "triviaqa": "F1",
    "samsum": "RL",
    "passage_retrieval_en": "Retrieve",
    "passage_count": "Count",
    "lcc": "Sim",
    "repobench-p": "Sim",
}

data2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}


def scorer_e(dataset, predictions, answers, all_classes):
    scores = []
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for gound_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, gound_truth, all_classes=all_classes
                ),
            )

        scores.append(score)

    scores = round(100 * np.mean(scores), 2)
    return scores


def get_stop_tokens(args, tokenizer):
    lst = [tokenizer.bos_token_id]
    if "llama-3" in args.model.lower():
        lst.append(128009)
        lst.append(128006)

        if args.dataset == "samsum":
            lst.append(tokenizer.encode("Dialogue", add_special_tokens=False)[-1])
        if args.dataset == "triviaqa":
            lst.append(tokenizer.encode("Passage", add_special_tokens=False)[-1])

    if "mistral" in args.model.lower():
        if args.dataset in ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa"]:
            lst.append(tokenizer.encode("\n", add_special_tokens=False)[-1])
        if args.dataset == "samsum":
            lst.append(tokenizer.encode("Dialogue", add_special_tokens=False)[-1])
        if args.dataset == "triviaqa":
            lst.append(tokenizer.encode("Passage", add_special_tokens=False)[-1])

    if "qwen" in args.model.lower():
        if args.dataset in ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa"]:
            lst.append(tokenizer.encode("\n", add_special_tokens=False)[-1])
        if args.dataset == "samsum":
            lst.append(tokenizer.encode("Dialogue", add_special_tokens=False)[-1])
        if args.dataset == "triviaqa":
            lst.append(tokenizer.encode("Passage", add_special_tokens=False)[-1])

    return lst


def build_recomputed_tokens(args, extra_config, prompt_ids, tokenizer):
    indices = extract_recomputed_indices(args, extra_config, len(prompt_ids))
    if indices is None:
        return None

    recomputed_tokens = {}
    for idx in indices:
        token_id = int(prompt_ids[idx])
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        recomputed_tokens[str(int(idx))] = token_text

    return recomputed_tokens


def extract_recomputed_indices(args, extra_config, prompt_len):
    if args.reuse == "no":
        return None
    if not extra_config or "reuse_config" not in extra_config:
        return None

    reuse_config = extra_config["reuse_config"]
    if reuse_config is None:
        return None

    imp_indices = reuse_config.get("imp_indices", None)
    if imp_indices is None:
        return None

    if isinstance(imp_indices, torch.Tensor):
        indices = imp_indices.detach().to("cpu", dtype=torch.long).reshape(-1)
    else:
        indices = torch.as_tensor(imp_indices, dtype=torch.long).reshape(-1)

    if indices.numel() == 0:
        return None

    indices = torch.unique(indices, sorted=True)
    if torch.any(indices < 0) or torch.any(indices >= prompt_len):
        raise ValueError(
            f"invalid recomputed token indices: min={int(indices.min())}, "
            f"max={int(indices.max())}, prompt_len={prompt_len}"
        )
    return indices.tolist()


def build_segment_lengths(doc_ids, params):
    """
    Segment layout aligned with flattened prompt_ids:
    [prefix] + [doc chunks (drop doc_start_len)] + [question]
    """
    if len(doc_ids) < 2:
        raise ValueError(f"invalid doc_ids length: {len(doc_ids)}")

    seg_lens = [len(doc_ids[0])]
    doc_start_len = int(params["doc_start_len"])
    for doc_index in range(1, len(doc_ids) - 1):
        seg_len = len(doc_ids[doc_index]) - doc_start_len
        if seg_len < 0:
            raise ValueError(
                f"invalid segment length at doc_index={doc_index}: {seg_len}"
            )
        seg_lens.append(seg_len)
    seg_lens.append(len(doc_ids[-1]))
    return seg_lens


def build_recomputed_chunks(args, extra_config, doc_ids, params, prompt_ids):
    """
    Return per-segment recompute trace:
    [{"len": seg_len, "indices": [local_idx, ...]}, ...]
    """
    seg_lens = build_segment_lengths(doc_ids, params)
    if sum(seg_lens) != len(prompt_ids):
        raise ValueError(
            f"segment length mismatch: sum(seg_lens)={sum(seg_lens)} "
            f"vs prompt_len={len(prompt_ids)}"
        )

    indices = extract_recomputed_indices(args, extra_config, len(prompt_ids))
    if indices is None:
        return None

    by_chunk = []
    ptr = 0
    cursor = 0
    for seg_len in seg_lens:
        seg_end = cursor + seg_len
        local_indices = []
        while ptr < len(indices) and indices[ptr] < seg_end:
            local_indices.append(int(indices[ptr] - cursor))
            ptr += 1
        by_chunk.append({"len": int(seg_len), "indices": local_indices})
        cursor = seg_end

    return by_chunk


def flush_result_json(result_json_path, saved_results):
    """Write current results immediately and atomically."""
    tmp_path = f"{result_json_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(saved_results, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, result_json_path)


def load_existing_results(result_json_path):
    """Load existing result.json for resume. Returns [] if not found/empty."""
    if not os.path.exists(result_json_path):
        return []
    if os.path.getsize(result_json_path) == 0:
        return []

    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"invalid result.json format (expect list): {result_json_path}"
        )
    return data


def init_recomputed_chunks_txt(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("")


def append_recomputed_chunks_txt(path, item_id, chunks):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"item_id: {item_id}\n")
        if chunks is None:
            f.write("null\n\n")
            return
        for chunk in chunks:
            line = json.dumps(chunk, ensure_ascii=False, separators=(", ", ": "))
            f.write(f"{line}\n")
        f.write("\n")


def load_reuse_payload(args, item_idx):
    """在 CPU 上预加载当前样本复用所需文件。"""
    payload = {"cat_kv": load_kv(args, None, None, None, None, item_idx)}
    if args.reuse == "surprisal_chunk":
        payload["surprisal_info"] = load_surprisal(args, item_idx)
    return payload


def parse_bool_arg(v):
    if isinstance(v, bool):
        return v
    value = str(v).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")


def parse_args():
    parse = argparse.ArgumentParser(description="")
    parse.add_argument("--model", type=str, default=None)
    parse.add_argument("--reuse", type=str, default="fp16")
    parse.add_argument("--blend_gap_source", type=str, choices=["v", "k"], default="v")
    parse.add_argument(
        "--blend_debug_fusion", type=str, choices=["mul", "sum", "rank"], default="mul"
    )
    parse.add_argument("--output_path", type=str, default=None)
    parse.add_argument("--dataset", type=str, default=None)
    parse.add_argument("--kv_path", type=str, default=None)
    parse.add_argument("--drop", type=str, default=False)
    parse.add_argument("--drop_config", type=str, default=None)
    parse.add_argument("--rate", type=float, default=0.15)
    parse.add_argument(
        "--resume",
        type=parse_bool_arg,
        default=True,
        help="Whether to resume from existing result.json (default: True).",
    )

    args = parse.parse_args()
    return args


if __name__ == "__main__":
    # 固定随机种子，确保评测过程（如采样、随机初始化等）可复现。
    seed_everything(42)
    args = parse_args()
    # 输出目录是必填项：所有中间结果和最终统计都会写到这里。
    if not args.output_path:
        raise ValueError("--output_path is required")
    os.makedirs(args.output_path, exist_ok=True)
    # `result.json` 保存逐条样本的预测结果；`recomputed_chunks.txt` 记录重算 chunk 信息。
    result_json_path = os.path.join(args.output_path, "result.json")
    recomputed_chunks_path = os.path.join(args.output_path, "recomputed_chunks.txt")

    # 读取历史结果，支持断点续跑：
    # - `Saved` 中每个元素对应一个样本结果
    # - `resume_idx` 表示下一条要处理的样本下标
    if args.resume:
        Saved = load_existing_results(result_json_path)
        resume_idx = len(Saved)
    else:
        Saved = []
        resume_idx = 0
        print("resume disabled: start from scratch and overwrite outputs")

    # 加载数据集与 DataLoader。
    print(f"loading {args.dataset}...")

    # LongBench 数据项本身已是结构化字典；这里不做堆叠，按原样返回 batch。
    def custom_collate_fn(batch):
        return batch

    dataset = LongBench(args)
    dataloader = DataLoader(dataset, collate_fn=custom_collate_fn)
    # 不同子任务允许的最大生成长度不同，由 data2maxlen 统一配置。
    max_new_tokens = data2maxlen[args.dataset]
    dataset_len = len(dataset)
    # 若历史结果条数超过数据集长度，说明结果文件与当前数据集不匹配，直接报错。
    if resume_idx > dataset_len:
        raise ValueError(
            f"resume index out of range: resume_idx={resume_idx}, dataset_len={dataset_len}"
        )

    # 首次运行时初始化输出文件；断点续跑时保留已完成结果并继续追加。
    if resume_idx == 0:
        flush_result_json(result_json_path, [])
        init_recomputed_chunks_txt(recomputed_chunks_path)
    else:
        print(f"resume enabled: skip first {resume_idx} finished items")
        # 兼容旧目录只有 result.json 没有 recomputed_chunks.txt 的情况。
        if not os.path.exists(recomputed_chunks_path):
            print(
                "warning: recomputed_chunks.txt not found, create a new file and append from resumed index"
            )
            init_recomputed_chunks_txt(recomputed_chunks_path)

    # 加载模型与分词器。
    print(f"loading {args.model}")
    model, tokenizer = load_model(args)

    # 主循环统计项：
    # - TTFT: time to first token
    # - TPOT: time per output token
    # - LEN: 生成长度（token 数）
    TTFT, TPOT, LEN = [], [], []
    # 若是断点续跑，把已完成样本中的历史统计先恢复进来，避免最终均值被低估。
    for item in Saved:
        if "ttft" in item:
            TTFT.append(float(item["ttft"]))
        if "tpot" in item:
            TPOT.append(float(item["tpot"]))
        if "pred_len" in item:
            LEN.append(int(item["pred_len"]))
        elif "prediction" in item:
            LEN.append(len(tokenizer.encode(item["prediction"])))

    stop_list = get_stop_tokens(args, tokenizer)

    # 为 KV 复用场景准备“下一条样本预读”线程：
    # 目标是将磁盘读取与当前样本的 GPU 解码并行，减少等待。
    prefetch_executor = None
    prefetch_future = None
    if args.reuse != "no" and resume_idx < dataset_len:
        # 先提交 resume_idx 对应样本的预读任务。
        prefetch_executor = ThreadPoolExecutor(max_workers=1)
        prefetch_future = prefetch_executor.submit(load_reuse_payload, args, resume_idx)

    # `islice(..., resume_idx, None)` 跳过已完成样本，实现断点继续。
    data_iter = enumerate(
        itertools.islice(dataloader, resume_idx, None), start=resume_idx
    )
    try:
        for i, batch in tqdm(data_iter, total=dataset_len, initial=resume_idx):
            data = {}
            # 当前实现默认 batch_size=1，因此取 batch[0]。
            doc_ids, prompt_ids, answers, params, classes = (
                batch[0]["doc_ids"],
                batch[0]["prompt_ids"],
                batch[0]["answer"],
                batch[0]["params"],
                batch[0]["all_classes"],
            )
            # 仅在 decode 路径使用（含复用/丢弃等策略）；vanilla 路径可为 None。
            extra_config = None

            # 构造模型输入（统一放到 CUDA）。
            input_ids = torch.tensor([prompt_ids]).to("cuda")

            past_key_values = None
            position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0).to("cuda")
            input = {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }  # pos 这里只支持bs 1

            # generate
            if args.reuse == "no" and args.drop == "False":  # 全部重算+全kv
                # 纯基线路径：不做 KV 复用，不做 KV 丢弃，完整重算。
                continuation, ttft, tpot = vanilla(
                    args, model, tokenizer, input, stop_list, max_new_tokens, {}
                )

            else:
                # 复用/丢弃类策略统一走 decode，并通过 extra_config 传递控制参数。
                extra_config = initialize_config(args)
                # 注入当前样本的数据参数，供后续 chunk 切分和策略判断使用。
                extra_config["other_config"]["data_params"] = params

                if args.reuse != "no":
                    # 优先消费后台预读结果；若无预读任务则同步读取当前样本复用文件。
                    if prefetch_future is not None:
                        reuse_payload = prefetch_future.result()
                    else:
                        reuse_payload = load_reuse_payload(args, i)

                    # 立即预读下一条，尽量持续形成“读文件 + 解码”的流水线。
                    next_i = i + 1
                    if prefetch_executor is not None and next_i < dataset_len:
                        prefetch_future = prefetch_executor.submit(
                            load_reuse_payload, args, next_i
                        )
                    else:
                        prefetch_future = None

                    # 复用路径的核心输入：拼接后的历史 KV。
                    extra_config["reuse_config"]["cat_kv"] = reuse_payload["cat_kv"]
                    if args.reuse == "surprisal_chunk":
                        # surprisal_chunk 需要额外的 token 粒度困难度分数与 chunk 边界。
                        surprisal_info = reuse_payload["surprisal_info"]
                        # 基本字段校验，避免下游 KeyError 或 silent bug。
                        if (
                            "scores" not in surprisal_info
                            or "chunk_ranges" not in surprisal_info
                        ):
                            raise ValueError(
                                f"invalid surprisal file format for item_{i}"
                            )

                        # 校验 surprisal 文件记录的序列长度是否与当前 prompt 一致。
                        seq_len = int(surprisal_info.get("seq_len", -1))
                        if seq_len != len(prompt_ids):
                            raise ValueError(
                                f"surprisal seq_len mismatch for item_{i}: {seq_len} vs {len(prompt_ids)}"
                            )

                        scores = surprisal_info["scores"]
                        # 分数长度必须与 token 长度一一对应。
                        if scores.numel() != len(prompt_ids):
                            raise ValueError(
                                f"surprisal score length mismatch for item_{i}: {scores.numel()} vs {len(prompt_ids)}"
                            )
                        # 防止出现 NaN/Inf 导致后续打分或排序异常。
                        if not torch.isfinite(scores).all():
                            raise ValueError(f"non-finite surprisal scores in item_{i}")

                        # 以当前样本真实文档切分结果为准，核对加载的 chunk 边界是否一致。
                        expected_ranges = build_doc_chunk_ranges(doc_ids, params)
                        loaded_ranges = [
                            tuple(r) for r in surprisal_info["chunk_ranges"]
                        ]
                        if loaded_ranges != expected_ranges:
                            raise ValueError(
                                f"chunk_ranges mismatch for item_{i}: loaded={loaded_ranges}, expected={expected_ranges}"
                            )

                        # 把校验后的 surprisal 信息写入配置，供 decode 阶段使用。
                        extra_config["reuse_config"]["surprisal_scores"] = scores
                        extra_config["reuse_config"]["chunk_ranges"] = expected_ranges

                continuation, ttft, tpot = decode(
                    args,
                    model,
                    tokenizer,
                    input,
                    stop_list,
                    max_new_tokens,
                    extra_config,
                )

            # 记录时延统计。
            TTFT.append(ttft)
            TPOT.append(tpot)

            # 组装当前样本输出。
            data["prediction"] = continuation
            data["answers"] = answers
            data["all_classes"] = classes
            data["ttft"] = float(ttft)
            data["tpot"] = float(tpot)

            # 添加重算token信息，占据垂直篇幅过长
            # data["recomputed_tokens"] = build_recomputed_tokens(
            #     args=args,
            #     extra_config=extra_config,
            #     prompt_ids=prompt_ids,
            #     tokenizer=tokenizer,
            # )

            # 生成长度按 tokenizer 编码后的 token 数统计。
            pred_len = len(tokenizer.encode(continuation))
            LEN.append(pred_len)
            data["pred_len"] = int(pred_len)
            Saved.append(data)
            # recomputed_chunks 用于复盘“哪些 chunk 被重算”，单独写文本文件便于审计。
            recomputed_chunks = build_recomputed_chunks(
                args=args,
                extra_config=extra_config,
                doc_ids=doc_ids,
                params=params,
                prompt_ids=prompt_ids,
            )
            # 每条样本完成后立刻落盘，降低长任务中断带来的结果丢失风险。
            flush_result_json(result_json_path, Saved)
            append_recomputed_chunks_txt(recomputed_chunks_path, i, recomputed_chunks)
            # import pdb; pdb.set_trace()
    finally:
        # 无论成功或异常都回收后台线程，避免进程退出时悬挂。
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=False)

    predictions, answers = [], []
    for item in Saved:
        predictions.append(item["prediction"])
        answers.append(item["answers"])
        # LongBench 的 scorer_e 需要 all_classes（分类任务标签空间）。
        all_classes = item["all_classes"]

    # 统一计算任务指标分数（如 F1/EM/Accuracy 等，具体由数据集决定）。
    score = scorer_e(args.dataset, predictions, answers, all_classes)

    # 汇总并写入最终可读报告。
    now = datetime.now()
    formatted_datetime = now.strftime("%Y年%m月%d日 %H时%M分%S秒")
    mean_ttft_ms = np.mean(TTFT) * 1000 if len(TTFT) > 0 else float("nan")
    mean_tpot_ms = np.mean(TPOT) * 1000 if len(TPOT) > 0 else float("nan")
    mean_len = np.mean(LEN) if len(LEN) > 0 else float("nan")
    with open(f"{args.output_path}/result.txt", "w") as f:
        f.write(formatted_datetime)
        f.write("\n")
        f.write(f"|------------- {args.dataset:^10s} {args.reuse:^7s} ------------|\n")
        f.write(
            f"|TTFT: {mean_ttft_ms:8.1f}| TPOT: {mean_tpot_ms:8.1f}| {METRIC_NAME[args.dataset]}: {score:8.2f}|\n"
        )
        f.write(f"Average len: {mean_len:.2f}\n")

    with open(f"{args.output_path}/result.txt", "r") as file:
        content = file.read()
        print(content)
