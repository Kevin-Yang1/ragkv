import os
from torch.utils.data import Dataset, DataLoader
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

    args = parse.parse_args()
    return args

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
    dataloader = DataLoader(dataset, collate_fn=custom_collate_fn)

    # load_model
    print(f'loading {args.model}')
    model, _ = load_model_precompute(args)

    # precompute
    idx = 0
    for batch in tqdm(dataloader):
        if flag == 'longbench':
            doc_ids, prompt_ids, answers, params, classes = batch[0]['doc_ids'], batch[0]['prompt_ids'], batch[0]['answer'], batch[0]['params'], batch[0]['all_classes']
            save_kvs(args, model, doc_ids, idx, params)
            save_surprisal_chunkwise(args, model, doc_ids, prompt_ids, idx, params)
            # save_attns(args, model, prompt_ids, idx, params)

        if flag == 'needle':
            doc_ids, prompt_ids, params, depth, context_length= batch[0]['doc_ids'], batch[0]['prompt_ids'], batch[0]['params'], batch[0]['depth'], batch[0]['context_length']
            # import pdb; pdb.set_trace()
            save_kvs(args, model, doc_ids, f'{str(context_length)}_{str(depth)}', params)

        if flag == 'ruler':
            doc_ids, prompt_ids, answers, params = batch[0]['doc_ids'], batch[0]['prompt_ids'], batch[0]['answer'], batch[0]['params']
            save_kvs(args, model, doc_ids, idx, params)

        idx += 1
