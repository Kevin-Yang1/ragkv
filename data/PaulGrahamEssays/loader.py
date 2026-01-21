from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import json
import torch
from transformers import AutoTokenizer

CONFIG = {
    'mistral': {'s_start': [1, 733, 16289, 28793], 's_end': [733, 28748, 16289, 28793]},
    'llama-3': {'s_start': [128000, 128006, 882, 128007, 271], 's_end': [128009, 128006, 78191, 128007, 271]},
    'qwen': {'s_start': [151644, 8948, 198, 2610, 525,   1207,  16948,     11,   3465, 553,  54364,  14817,     13,   1446,    525,    264,  10950,  17847, 13, 151645,    198, 151644,    872,    198], 's_end': [151645,    198, 151644,  77091,    198]},
}

class Needle(Dataset):
    def __init__(self, args):
        self.dataset_path = f'./inputs/{os.path.basename(args.model)}/needle.json'
        self.ori_data = json.load(open(self.dataset_path, 'r'))
        self.prefix_prompt = 'This is a very long story book: <book> '
        self.q_prompt = '</book>.\n Based on the content of the book, Question: The best thing to do in San Francisco is: \nAnswer:'

        if 'mistral' in args.model.lower():
            self.config = CONFIG['mistral']
        elif 'llama-3' in args.model.lower():
            self.config = CONFIG['llama-3']
        elif 'qwen' in args.model.lower():
            self.config = CONFIG['qwen']

        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        if 'qwen' in args.model.lower():
            self._construct_for_qwen(args)
        else:
            self._construct(args)

    def __len__(self):
        return len(self.prompt_ids)
    
    def __getitem__(self, idx):
        return {'doc_ids': self.doc_ids[idx],
                'prompt_ids': self.prompt_ids[idx],
                'depth':self.depth[idx],
                'context_length':self.context_length[idx],
                'params': self.params[idx],
                }
    
    def _construct(self, args):
        self.doc_ids, self.prompt_ids, self.params, self.context_length, self.depth = [], [], [], [], []
        # prefix_prompt = [self.tokenizer.bos_token_id] + self.tokenizer.encode(self.prefix_prompt[args.dataset])[1:]
        prefix_prompt = [self.tokenizer.bos_token_id] + self.config['s_start'] + self.tokenizer.encode(self.prefix_prompt)[1:]
        q_ids = self.tokenizer.encode(self.q_prompt)[1:]+self.config['s_end']
        # q_ids = self.tokenizer.encode(q_prompt)[1:]

        doc_start_len = 1
        for item in tqdm(self.ori_data):
            # import pdb; pdb.set_trace()
            context_length = item['context_length']
            depth = item['depth_percent']
            doc_prompts = item['chunks']

            doc_chunk_ids = [self.tokenizer.encode(doc) for doc in doc_prompts]
            doc_chunk_ids = [chunk_ids for chunk_ids in doc_chunk_ids]
            doc_chunk_ids = [prefix_prompt] + doc_chunk_ids
            doc_chunk_ids = doc_chunk_ids + [q_ids]  # doc_chunk_ids: [[<bos> (chat)prefix], [<bos> ids], [<bos> ids], ..., [q <eos>]]
            prefix_len = len(doc_chunk_ids[0])
            last_len = len(q_ids)

            prompt_ids = []
            for i in range(len(doc_chunk_ids)):
                if i == 0 or i == len(doc_chunk_ids)-1 :
                    prompt_ids += doc_chunk_ids[i]
                else:
                    prompt_ids += doc_chunk_ids[i][doc_start_len:]

            sink_indices = [0] * len(prompt_ids)

            self.context_length.append(context_length)
            self.depth.append(depth)
            self.doc_ids.append(doc_chunk_ids)
            self.prompt_ids.append(prompt_ids)
            self.params.append({'doc_start_len': doc_start_len,'prefix_len': prefix_len, 'last_len': last_len, 'sink_pos': sink_indices})

    def _construct_for_qwen(self, args):
        self.doc_ids, self.prompt_ids, self.params, self.context_length, self.depth = [], [], [], [], []
        # prefix_prompt = self.tokenizer.encode(self.prefix_prompt[args.dataset])
        prefix_prompt = self.config['s_start'] + self.tokenizer.encode(self.prefix_prompt)
        q_ids = self.tokenizer.encode(self.q_prompt)+self.config['s_end']
        # q_ids = self.tokenizer.encode(q_prompt)

        doc_start_len = 0
        for item in tqdm(self.ori_data):
            # import pdb; pdb.set_trace()
            context_length = item['context_length']
            depth = item['depth_percent']
            doc_prompts = item['chunks']

            doc_chunk_ids = [self.tokenizer.encode(doc) for doc in doc_prompts]
            doc_chunk_ids = [chunk_ids for chunk_ids in doc_chunk_ids]
            doc_chunk_ids = [prefix_prompt] + doc_chunk_ids
            doc_chunk_ids = doc_chunk_ids + [q_ids]  # doc_chunk_ids: [[(chat)prefix], [ids], [ids], ..., [q <eos>]]
            prefix_len = len(doc_chunk_ids[0])
            last_len = len(q_ids)

            prompt_ids = []
            for i in range(len(doc_chunk_ids)):
                if i == 0 or i == len(doc_chunk_ids)-1 :
                    prompt_ids += doc_chunk_ids[i]
                else:
                    prompt_ids += doc_chunk_ids[i][doc_start_len:]

            sink_indices = [0] * len(prompt_ids)

            self.context_length.append(context_length)
            self.depth.append(depth)
            self.doc_ids.append(doc_chunk_ids)
            self.prompt_ids.append(prompt_ids)
            self.params.append({'doc_start_len': doc_start_len,'prefix_len': prefix_len, 'last_len': last_len, 'sink_pos': sink_indices})