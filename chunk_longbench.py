import json
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import os

# MAXLENGTH = 3500
MAXLENGTH = 7000
CHUNKSIZE = 512

datasets = [
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

path = '../Models/LLMs/Qwen/Qwen2.5-7B-Instruct'
# path = '../Models/LLMs/llama3/Meta-Llama-3-8B-Instruct'
# path = '../Models/LLMs/Mistral-7B-Instruct-v0.2'

questions_format = json.load(open('./config/longbench/question_format.json', 'r'))
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

chunk_path = 'inputs/Qwen2.5-7B-Instruct'
# chunk_path = 'inputs/Meta-Llama-3-8B-Instruct'
# chunk_path = 'inputs/Mistral-7B-Instruct-v0.2'

os.makedirs(chunk_path, exist_ok=True)

for dataset in tqdm(datasets):
    question_format = questions_format[dataset]
    chunk_data = []

    with open(f'./data/longbench/{dataset}_e.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)

            chunk_list = []
            question = question_format.format(**item)
            context = item['context']
            answers = item['answers']
            classes = item['all_classes']

            tokenized_context = tokenizer(context, truncation=False, return_tensors='pt').input_ids[0]

            if len(tokenized_context) > MAXLENGTH:
                half = int(MAXLENGTH / 2)
                tokenized_context = torch.cat([tokenized_context[:half], tokenized_context[-half:]], dim=0)

            for i in range(0, tokenized_context.shape[0], CHUNKSIZE):
                chunk_list.append(tokenizer.decode(tokenized_context[i:i+CHUNKSIZE], skip_special_tokens=True))

            chunk_data.append({'chunks': chunk_list, 'question': question, 'answers': answers, 'all_classes': classes})

    with open(os.path.join(chunk_path, f'{dataset}.json'), 'w', encoding='utf-8') as f:
        json.dump(chunk_data, f, indent=4, ensure_ascii=False)