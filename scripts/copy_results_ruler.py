import os
import json
import argparse
import numpy as np
import csv

def string_match_all(preds, refs):
    """
    evaluation metric for RULER
    preds: List[str]
    refs: List[List[str]]
    """
    score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

if __name__ == '__main__':
    dataset_list = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt"]


    # results_dir = './outputs/Llama-3-8B-Instruct/debug-causal-snapkv'
    # results_dir = './outputs/Mistral-7B-Instruct/debug-causal-snapkv'
    results_dir = './outputs/Qwen2.5-7B-Instruct/debug-causal-snapkv'
    results_list = []
    for dataset in dataset_list:
        eval_file = os.path.join(results_dir,dataset,'result.json')

        scores = dict()
        predictions, answers, lengths = [], [], []
        # dataset = filename.split('.')[0]
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    predictions.append(data["pred"])
                    answers.append(data["answers"])

            score = string_match_all(predictions, answers)
            scores[dataset] = score
            results_list.append(str(score))

        except:
            results_list.append('N/A')

    with open('./.csv', 'w') as fp:
        writer = csv.writer(fp)
        # import pdb; pdb.set_trace()
        writer.writerow(results_list)
