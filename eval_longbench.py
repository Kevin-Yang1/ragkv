import argparse
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

METRIC_NAME = {"qasper": 'F1',
    "multifieldqa_en": 'F1',
    "hotpotqa": 'F1',
    "2wikimqa": 'F1',
    "gov_report": 'RL',
    "multi_news": 'RL',
    "trec": 'Acc',
    "triviaqa": 'F1',
    "samsum": 'RL',
    "passage_retrieval_en": 'Retrieve',
    "passage_count": 'Count',
    "lcc": 'Sim',
    "repobench-p": 'Sim',}

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
    "repobench-p": 64
}


def scorer_e(dataset, predictions, answers, all_classes):
    scores = []
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ['trec', 'triviaqa', 'samsum']:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for gound_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, gound_truth, all_classes=all_classes))

        scores.append(score)
    
    scores = round(100 * np.mean(scores), 2)
    return scores

def get_stop_tokens(args, tokenizer):
    lst = [tokenizer.bos_token_id]
    if 'llama-3' in args.model.lower():
        lst.append(128009)
        lst.append(128006)

        if args.dataset == 'samsum':
            lst.append(tokenizer.encode('Dialogue', add_special_tokens=False)[-1])
        if args.dataset == 'triviaqa':
            lst.append(tokenizer.encode('Passage', add_special_tokens=False)[-1])

    if 'mistral' in args.model.lower():
        if args.dataset in ['qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa']:
            lst.append(tokenizer.encode('\n', add_special_tokens=False)[-1])
        if args.dataset == 'samsum':
            lst.append(tokenizer.encode('Dialogue', add_special_tokens=False)[-1])
        if args.dataset == 'triviaqa':
            lst.append(tokenizer.encode('Passage', add_special_tokens=False)[-1])

    if 'qwen' in args.model.lower():
        if args.dataset in ['qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa']:
            lst.append(tokenizer.encode('\n', add_special_tokens=False)[-1])
        if args.dataset == 'samsum':
            lst.append(tokenizer.encode('Dialogue', add_special_tokens=False)[-1])
        if args.dataset == 'triviaqa':
            lst.append(tokenizer.encode('Passage', add_special_tokens=False)[-1])

    return lst

def parse_args():
    parse = argparse.ArgumentParser(description='')
    parse.add_argument('--model', type=str, default=None)
    parse.add_argument('--reuse', type=str, default='fp16')
    parse.add_argument('--output_path', type=str, default=None)
    parse.add_argument('--dataset', type=str, default=None)
    parse.add_argument('--kv_path', type=str, default=None)
    parse.add_argument('--drop', type=str, default=False)
    parse.add_argument('--drop_config', type=str, default=None)
    parse.add_argument('--rate', type=float, default=0.15)

    args = parse.parse_args()
    return args


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    # load_datasets
    print(f'loading {args.dataset}...')
    def custom_collate_fn(batch):
        return batch
    
    dataset = LongBench(args)
    dataloader = DataLoader(dataset, collate_fn=custom_collate_fn)
    max_new_tokens = data2maxlen[args.dataset]

    # load_model
    print(f'loading {args.model}')
    model, tokenizer = load_model(args)

    # main
    Saved, TTFT, TPOT, LEN = [], [], [], []
    stop_list = get_stop_tokens(args, tokenizer)
    
    i = 0
    for batch in tqdm(dataloader):
        data = {}
        doc_ids, prompt_ids, answers, params, classes = batch[0]['doc_ids'], batch[0]['prompt_ids'], batch[0]['answer'], batch[0]['params'], batch[0]['all_classes']
        
        input_ids = torch.tensor([prompt_ids]).to('cuda')

        past_key_values = None
        position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0).to('cuda')
        input = {'input_ids': input_ids, 'past_key_values': past_key_values, 'position_ids': position_ids} # pos 这里只支持bs 1

        # generate
        if args.reuse == 'no' and args.drop == 'False': # 全部重算+全kv
            continuation, ttft, tpot = vanilla(args, model, tokenizer, input, stop_list, max_new_tokens, {})
        
        else:
            extra_config = initialize_config(args)
            extra_config['other_config']['data_params'] = params

            if args.reuse != 'no':
                extra_config['reuse_config']['cat_kv'] = load_kv(args, model, tokenizer, doc_ids, params, i)

            continuation, ttft, tpot = decode(args, model, tokenizer, input, stop_list, max_new_tokens, extra_config)

        TTFT.append(ttft)
        TPOT.append(tpot)

        data['prediction'] = continuation
        data['answers'] = answers
        data['all_classes'] = classes
        LEN.append(len(tokenizer.encode(continuation)))
        Saved.append(data)
        i += 1
        # import pdb; pdb.set_trace()

    # print
    with open(f'{args.output_path}/result.json', 'w') as f:
        json.dump(Saved, f, indent=4)  

    data = json.load(open(f'{args.output_path}/result.json', "r"))
    predictions, answers = [], []
    for item in data:
        predictions.append(item["prediction"])
        answers.append(item["answers"])
        all_classes = item["all_classes"]

    score = scorer_e(args.dataset, predictions, answers, all_classes)

    # record
    now = datetime.now()
    formatted_datetime = now.strftime("%Y年%m月%d日 %H时%M分%S秒")
    with open(f'{args.output_path}/result.txt', 'w') as f:
        f.write(formatted_datetime)
        f.write('\n')
        f.write(f"|------------- {args.dataset:^10s} {args.reuse:^7s} ------------|\n")
        f.write(f"|TTFT: {np.mean(TTFT)*1000:8.1f}| TPOT: {np.mean(TPOT)*1000:8.1f}| {METRIC_NAME[args.dataset]}: {score:8.2f}|\n")
        f.write(f'Average len: {np.mean(LEN):.2f}\n')

    with open(f'{args.output_path}/result.txt', 'r') as file:
        content = file.read()
        print(content)