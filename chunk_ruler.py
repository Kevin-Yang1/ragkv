import os
import json
import random
import argparse
import numpy as np
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
from typing import List

from model_utils import get_model_basename

from utils import *

context_length_list = [4096]
# context_length_list = [4096, 8192, 16384]

datasets = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
            "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt"]

dataset2maxlen = {
    "niah_single_1": 64,
    "niah_single_2": 64,
    "niah_single_3": 64,
    "niah_multikey_1": 64,
    "niah_multikey_2": 64,
    "niah_multikey_3": 64,
    "niah_multiquery": 64,
    "niah_multivalue": 64,
    "cwe": 64,
    "fwe": 64,
    "vt": 64
}


# model2maxlen = {
#     "llama2": 3950,
#     "llama-2": 3950,
#     "llama3": 7950,
#     "llama-3": 7950,
#     "mistral": 7950,
# }

model2maxlen = {
    "llama2": 3950,
    "llama-2": 3950,
    "llama3": 3950,
    "llama-3": 3950,
    "mistral": 3950,
    'qwen': 3950,
}


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prefix_prompt = json.load(open('config/ruler/prefix_prompt.json', 'r'))
    prefix = prefix_prompt[args.dataset]
    chunk_size = 512
    output_dir = os.path.join(args.output_root, get_model_basename(args.model_path))
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    test_data = []
    prompt_list = []
    input_list = []
    outputs_list: List[List[str]] = [] # List of List
    length_list = []
    index_list = []

    model_path = args.model_path.lower()
    for key in model2maxlen:
        if key in model_path:
            model_max_len = model2maxlen[key]

    output_max_len = dataset2maxlen[args.dataset]

    data = []
    i = 0
    with open(args.data_file) as fp:
        for line in fp:
            example = json.loads(line)
            prompt = example["input"]
            outputs = example["outputs"]

            question = prompt.split('\n')[-1]
            document = prompt[len(prefix):-len(question)]
            input_ids = tokenizer.encode(document)[1:]
            chunks = [tokenizer.decode(input_ids[i:i+512]) for i in range(0, len(input_ids), 512)]

            entry = {
                'chunks': chunks,
                'question': question,
                'answers': outputs
            }

            data.append(entry)
            i += 1
            
            if i == args.max_num_examples:
                break


    # with open(f'./inputs/Mistral-7B-Instruct-v0.2/{args.dataset}.json', 'w') as f:
    #     json.dump(data, f, indent=4)

    # with open(f'./inputs/Meta-Llama-3-8B-Instruct/{args.dataset}.json', 'w') as f:
    #     json.dump(data, f, indent=4)

    with open(os.path.join(output_dir, f'{args.dataset}.json'), 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--output_root", type=str, default="./inputs", help="root directory for model-specific chunk files.")
    parser.add_argument("--context_length", type=int, default=None, help="single context length to process. defaults to the built-in list.")
    parser.add_argument("--max_num_examples", type=int, default=300, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")

    args = parser.parse_args()

    seed_everything(42)

    if args.model_path is None:
        raise ValueError("--model_path is required")

    selected_context_lengths = [args.context_length] if args.context_length is not None else context_length_list
    selected_datasets = [args.dataset] if args.dataset else datasets

    for context_length in selected_context_lengths:
        for idx, dataset in enumerate(selected_datasets):

            print(f"Working on context length {context_length}, dataset: {dataset} - {idx}/{len(selected_datasets)}")
            args.context_length = context_length
            args.dataset = dataset
            args.data_file = f"data/RULER/{context_length}/{args.dataset}.jsonl"

            main(args)
