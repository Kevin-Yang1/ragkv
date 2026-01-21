import os
import json
import random
import argparse
import numpy as np
import torch

from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List
from models.loader import load_model
from torch.utils.data import DataLoader, Dataset
import time

from data.RULER.loader import Ruler
from utils import *


def get_stop_tokens(args, tokenizer):
    lst = [tokenizer.eos_token_id]

    lst.append(128009)
    lst.append(128006)
    
    return lst

def main(args, model, tokenizer, dataloader):

    idx = 0
    fout = open(os.path.join(args.output_path, "result.json"), "w")
    for batch in tqdm(dataloader):
        data = {}
        doc_ids, prompt_ids, answers, params = batch[0]['doc_ids'], batch[0]['prompt_ids'], batch[0]['answer'], batch[0]['params']
        # import pdb; pdb.set_trace()
        input_ids = torch.tensor([prompt_ids]).to('cuda')
        past_key_values = None
        position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0).to('cuda')
        input = {'input_ids': input_ids, 'past_key_values': past_key_values, 'position_ids': position_ids} # pos 这里只支持bs 1
        stop_list = get_stop_tokens(args, tokenizer)

        # generate
        max_new_tokens = 64
        if args.reuse == 'no' and args.drop == 'False': # 全部重算+全kv
            response, ttft, tpot = vanilla(args, model, tokenizer, input, stop_list, max_new_tokens, {})
        
        else:
            extra_config = initialize_config(args)
            extra_config['other_config']['data_params'] = params

            if args.reuse != 'no':
                extra_config['reuse_config']['cat_kv'] = load_kv(args, model, tokenizer, doc_ids, params, idx)

            response, ttft, tpot = decode(args, model, tokenizer, input, stop_list, max_new_tokens, extra_config)

        data["answers"] = answers
        data["pred"] = response

        fout.write(json.dumps(data) + "\n")

        idx += 1



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--model_name", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")

    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--reuse', type=str, default='fp16')
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--kv_path', type=str, default=None)
    parser.add_argument('--rate', type=float, default=0.15)
    parser.add_argument('--drop', type=str, default=False)
    parser.add_argument('--drop_config', type=str, default=None)
    args = parser.parse_args()

    seed_everything(42)

    # load model
    print(f'loading {args.dataset}...')
    def custom_collate_fn(batch):
        return batch
    
    dataset = Ruler(args)
    dataloader = DataLoader(dataset, collate_fn=custom_collate_fn)

    # load_model
    print(f'loading {args.model}')
    model, tokenizer = load_model(args)

    main(args, model, tokenizer, dataloader)
