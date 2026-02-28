import torch
import numpy as np
import random
import os
import time
import math
import torch.nn as nn
import json
import torch.nn.functional as F

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def save_kvs(args, model, doc_ids, item_id, params):
    '''
    save kvs for a single test sample
    '''
    def get_chunk_kvs(doc_index, past_key_values, params):
        if doc_index == 0 or doc_index == len(doc_ids)-1:
            # for prefix and query
            temp_k = past_key_values[0].clone() # prefix保留了bos，chunk的去除bos, q保持不动（q没有bos，只是占位需要全部"重计算"）
            temp_v = past_key_values[1].clone()
        else: # for all chunks
            temp_k = past_key_values[0][:, :, params['doc_start_len']:, :].clone()
            temp_v = past_key_values[1][:, :, params['doc_start_len']:, :].clone()   
        
        return temp_k[0].to('cuda:0'), temp_v[0].to('cuda:0') # (num_key_value_heads, n, head_dim)

    save_path = os.path.join(args.kv_path, f'item_{item_id}')
    os.makedirs(save_path, exist_ok=True)
    saved_k, saved_v = [], []
    for i in range(len(doc_ids)):
        # just prefill
        reuse_config = {'cat_kv': [[], []]*32, 'check': None, 'decode': False}
        input_ids = torch.tensor([doc_ids[i]]).to('cuda')
        attention_mask = (input_ids != 0).int().to('cuda')

        with torch.no_grad():
            _ = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=True,
                            return_dict=True,
                            past_key_values=None,
                            reuse_config=reuse_config,)
        
        # cat in layer dim
        layer_list = model.model.layers
        for layer_idx in range(len(layer_list)):
            past_key_values = layer_list[layer_idx].self_attn.hack_kv
            temp_k, temp_v = get_chunk_kvs(i, past_key_values, params)
            
            if i == 0:
                saved_k.append(temp_k)
                saved_v.append(temp_v)
            else:
                # import pdb; pdb.set_trace()
                saved_k[layer_idx] = torch.cat((saved_k[layer_idx], temp_k), dim=1)
                saved_v[layer_idx] = torch.cat((saved_v[layer_idx], temp_v), dim=1)

        layer_list[layer_idx].self_attn.hack_kv = None

        '''
        [(num_key_value_heads, n, head_dim), 
         (num_key_value_heads, n, head_dim),
         ...]
        
        '''

    saved_k, saved_v = torch.stack(saved_k), torch.stack(saved_v)
    saved_kv = torch.stack([saved_k, saved_v]) # (2, 32, 8, n, 128)
    torch.save(saved_kv, f'{save_path}/kvs.pt')
    
    return

def build_doc_chunk_ranges(doc_ids, params):
    """Return doc chunk ranges in flattened prompt space: [(start, end), ...]."""
    ranges = []
    cursor = len(doc_ids[0])  # prefix contributes full length
    for doc_index in range(1, len(doc_ids) - 1):
        chunk_len = len(doc_ids[doc_index]) - params['doc_start_len']
        if chunk_len < 0:
            raise ValueError(f'invalid chunk length at index {doc_index}: {chunk_len}')
        start = cursor
        end = cursor + chunk_len
        ranges.append((start, end))
        cursor = end
    return ranges

def save_surprisal_chunkwise(args, model, doc_ids, prompt_ids, item_id, params):
    """
    Compute token surprisal per chunk independently, then align to flattened prompt_ids.
    """
    def get_chunk_surprisal(input_ids):
        attention_mask = torch.ones_like(input_ids, dtype=torch.int)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                past_key_values=None,
                reuse_config=None,
            )

        logits = outputs.logits
        seq_len = input_ids.shape[-1]
        if seq_len == 0:
            return torch.empty(0, dtype=torch.float32, device=input_ids.device)

        scores = torch.zeros(seq_len, dtype=torch.float32, device=input_ids.device)
        if seq_len > 1:
            shift_logits = logits[:, :-1, :].float()
            shift_targets = input_ids[:, 1:]
            token_log_probs = F.log_softmax(shift_logits, dim=-1)
            token_logp = token_log_probs.gather(-1, shift_targets.unsqueeze(-1)).squeeze(-1).squeeze(0)
            scores[1:] = -token_logp

        return scores

    save_path = os.path.join(args.kv_path, f'item_{item_id}')
    os.makedirs(save_path, exist_ok=True)

    seg_scores = []
    for seg_ids in doc_ids:
        seg_tensor = torch.tensor([seg_ids], device='cuda')
        seg_scores.append(get_chunk_surprisal(seg_tensor))

    aligned_scores = []
    for i, scores in enumerate(seg_scores):
        if i == 0 or i == len(seg_scores) - 1:
            aligned_scores.append(scores)
        else:
            aligned_scores.append(scores[params['doc_start_len']:])

    merged_scores = torch.cat(aligned_scores, dim=0)
    if merged_scores.numel() != len(prompt_ids):
        raise ValueError(
            f'surprisal length mismatch: got {merged_scores.numel()}, expect {len(prompt_ids)}'
        )

    chunk_ranges = build_doc_chunk_ranges(doc_ids, params)
    torch.save(
        {
            'scores': merged_scores.cpu(),
            'chunk_ranges': chunk_ranges,
            'seq_len': int(merged_scores.numel()),
            'version': 1,
            'mode': 'chunk_independent',
        },
        f'{save_path}/surprisal.pt',
    )

    return

def load_kv(args, model, tokenizer, doc_ids, params, item_idx):
    '''
    原本是返回chunk past key values
    [[(1, 8, n, 512), (1, 8, n, 512)], layer0
     [(1, 8, n, 512), (1, 8, n, 512)], layer1
     ...
    ]
    ./kvs/${model}/${dataset}/item_0
    '''
    old_kv_path = f'{args.kv_path}/item_{item_idx}/kvs.pt'
    old_kv = torch.load(old_kv_path)

    return old_kv

def load_surprisal(args, item_idx):
    surprisal_path = f'{args.kv_path}/item_{item_idx}/surprisal.pt'
    if not os.path.exists(surprisal_path):
        raise FileNotFoundError(f'missing surprisal file: {surprisal_path}')
    return torch.load(surprisal_path, map_location='cpu')

def initialize_config(args):
    if args.reuse != 'no':
        reuse_config = {
            'check': None,
            'fake_q': None,
            'mask': None,
            'recomp_ratio': args.rate,
            'causal': True,
        }
        reuse_config['reuse'] = args.reuse
        if args.reuse == 'surprisal_chunk':
            reuse_config['surprisal_scores'] = None
            reuse_config['chunk_ranges'] = None
    else:
        reuse_config = None

    if args.drop_config not in [None, 'None', '']:
        drop_config = json.load(open(args.drop_config, 'r')) 
        drop_config['drop'] = args.drop
    else:
        drop_config = None

    extra_config = {'reuse_config': reuse_config, 'drop_config': drop_config, 'other_config': {'decode': False}}

    return extra_config

def decode_step(args, model, input, config, stop_list, step):
    
    if config == {}:
        outputs = model(input_ids=input['input_ids'],
                        use_cache=True,
                        return_dict=True,
                        past_key_values=input['past_key_values'],
                        position_ids=input['position_ids'],)
    else:
        
        outputs = model(input_ids=input['input_ids'],
                        use_cache=True,
                        return_dict=True,
                        past_key_values=input['past_key_values'],
                        extra_config=config,
                        position_ids=input['position_ids'],)  
        
    logits = outputs.logits[:,-1,:]
    next_token_logits = logits.log_softmax(dim=-1)
    next_token = torch.argmax(next_token_logits, dim=-1)

    if step == 0:
        first_token_time = time.time()
        if args.drop != 'False':
            outputs.past_key_values.update_after_prefill(args.drop)
            outputs.past_key_values = outputs.past_key_values.to_legacy_cache()
    else:
        first_token_time = 0

    input['input_ids'] = next_token.unsqueeze(0)
    input['past_key_values'] = outputs.past_key_values
    input['position_ids'] = torch.tensor([input['position_ids'][0][-1].item()+1]).unsqueeze(0).cuda()
        
    stop = True if next_token.item() in stop_list else False
    return input, next_token, stop, first_token_time

def vanilla(args, model, tokenizer, input, stop_list, max_new_tokens, config):
    new_token_list = []

    with torch.no_grad():
        start_time = time.time()
        for i in range(max_new_tokens):
            input, next_token, stop, first_token_time = decode_step(args=args, model=model, input=input, config=config, stop_list=stop_list, step=i)

            if i == 0:
                ttft = first_token_time - start_time
            if stop:
                break

            new_token_list.append(next_token.item())

        tpot = (time.time() - start_time) / (i+1)
        output_sequence = tokenizer.decode(new_token_list, skip_special_tokens=True)
        print(output_sequence)
        return output_sequence, ttft, tpot

def decode(args, model, tokenizer, input, stop_list, max_new_tokens, config):
    new_token_list = []

    with torch.no_grad():
        start_time = time.time()
        for i in range(max_new_tokens):
            input, next_token, stop, first_token_time = decode_step(args=args, model=model, input=input, config=config, stop_list=stop_list, step=i)
            
            if i == 0:
                ttft = first_token_time - start_time
                if config['reuse_config']:
                    config['reuse_config']['check'] = None
                config['other_config']['decode'] = True
            if stop:
                break
            
            if config['reuse_config']:
                config['reuse_config']['cat_kv'] = None # decode阶段可能还需要根据q做选择
            new_token_list.append(next_token.item())

        tpot = (time.time() - start_time) / (i+1)
        output_sequence = tokenizer.decode(new_token_list, skip_special_tokens=True)
        print(output_sequence)
        return output_sequence, ttft, tpot
