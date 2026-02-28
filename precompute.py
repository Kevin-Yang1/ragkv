import os
from torch.utils.data import Dataset, DataLoader, Subset
import time
import torch
import json
import numpy as np
from tqdm import tqdm
import random
import argparse

# my pack
from data.longbench.loader import LongBench
from data.PaulGrahamEssays.loader import Needle
from data.RULER.loader import Ruler
from models.loader import load_model, load_model_precompute
from utils import *

def parse_args():
    parse = argparse.ArgumentParser(description='')
    parse.add_argument('--model', type=str, default=None)
    parse.add_argument('--kv_path', type=str, default=None)
    parse.add_argument('--dataset', type=str, default=None)
    parse.add_argument('--reuse', type=str, default='no')
    parse.add_argument('--save_surprisal', action='store_true')

    args = parse.parse_args()
    return args


def item_is_precomputed(kv_path, item_id, require_surprisal=False):
    item_dir = os.path.join(kv_path, f"item_{item_id}")
    kv_file = os.path.join(item_dir, "kvs.pt")
    if not os.path.exists(kv_file):
        return False
    if require_surprisal:
        return os.path.exists(os.path.join(item_dir, "surprisal.pt"))
    return True


def find_resume_index(kv_path, dataset_len, require_surprisal=False):
    resume_idx = 0
    while resume_idx < dataset_len:
        if not item_is_precomputed(
            kv_path, resume_idx, require_surprisal=require_surprisal
        ):
            break
        resume_idx += 1
    return resume_idx


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    # load datasets
    print(f'precomputing...')
    print(f'loading {args.dataset}')
    def custom_collate_fn(batch):
        return batch
    
    if args.dataset in ["qasper","multifieldqa_en","hotpotqa","2wikimqa","gov_report","multi_news","trec","triviaqa","samsum","passage_count","lcc"]:
        dataset = LongBench(args)
        flag = 'longbench'

    elif args.dataset == 'needle':
        dataset = Needle(args)
        flag = 'needle'

    elif args.dataset in ["niah_single_1","niah_single_2","niah_single_3","niah_multikey_1","niah_multikey_2","niah_multikey_3","niah_multiquery","niah_multivalue","cwe","fwe","vt"]:
        dataset = Ruler(args)
        flag = 'ruler'

    else:
        raise NotImplementedError

    resume_start = 0
    if flag in ["longbench", "ruler"]:
        require_surprisal = (flag == "longbench") and args.save_surprisal
        dataset_len = len(dataset)
        resume_start = find_resume_index(
            args.kv_path,
            dataset_len,
            require_surprisal=require_surprisal,
        )
        if resume_start >= dataset_len:
            print(f"all {dataset_len} samples are already precomputed, exit.")
            raise SystemExit(0)
        if resume_start > 0:
            print(f"resume from item_{resume_start} (skip 0..{resume_start - 1})")
            dataset = Subset(dataset, range(resume_start, dataset_len))

    dataloader = DataLoader(dataset, collate_fn=custom_collate_fn)

    # load_model
    print(f'loading {args.model}')
    model, _ = load_model_precompute(args)

    # precompute
    idx = resume_start
    for batch in tqdm(dataloader):
        if flag == 'longbench':
            if item_is_precomputed(
                args.kv_path, idx, require_surprisal=args.save_surprisal
            ):
                idx += 1
                continue
            doc_ids, prompt_ids, answers, params, classes = batch[0]['doc_ids'], batch[0]['prompt_ids'], batch[0]['answer'], batch[0]['params'], batch[0]['all_classes']
            save_kvs(
                args,
                model,
                doc_ids,
                idx,
                params,
                prompt_ids=prompt_ids,
                save_surprisal=args.save_surprisal,
            )
            # save_attns(args, model, prompt_ids, idx, params)

        if flag == 'needle':
            doc_ids, prompt_ids, params, depth, context_length= batch[0]['doc_ids'], batch[0]['prompt_ids'], batch[0]['params'], batch[0]['depth'], batch[0]['context_length']
            item_id = f'{str(context_length)}_{str(depth)}'
            if item_is_precomputed(args.kv_path, item_id):
                idx += 1
                continue
            # import pdb; pdb.set_trace()
            save_kvs(args, model, doc_ids, item_id, params)

        if flag == 'ruler':
            if item_is_precomputed(args.kv_path, idx):
                idx += 1
                continue
            doc_ids, prompt_ids, answers, params = batch[0]['doc_ids'], batch[0]['prompt_ids'], batch[0]['answer'], batch[0]['params']
            save_kvs(args, model, doc_ids, idx, params)

        idx += 1
