import torch
from transformers.cache_utils import DynamicCache
import torch.nn.functional as F
import torch.nn as nn
import math
import time

# update past kv after the first token is generated
original_init = DynamicCache.__init__

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def new_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.kv_pool = {'key_pool': [], 'value_pool': [], 'indices_pool': []}

def prepare_update_kv(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    indices: torch.Tensor,
):

    self.kv_pool['key_pool'].append(key_states)
    self.kv_pool['value_pool'].append(value_states)
    self.kv_pool['indices_pool'].append(indices)

    return None

def update_after_prefill(self, mode):

    # for layer_idx in range(32): # TODO: layer nums
    for layer_idx in range(28): # TODO: layer nums
        key_states = self.kv_pool['key_pool'][layer_idx]
        value_states = self.kv_pool['value_pool'][layer_idx]
        save_indices = self.kv_pool['indices_pool'][layer_idx]
        
        if mode == 'Streaming': # streaming返回的indices只有一条token序列
            _, _ = self.update(key_states[:,:,save_indices,:], value_states[:,:,save_indices,:], layer_idx)
        elif 'SnapKV' in mode: # snapkv返回每个头的序列mask
            bsz, num_heads, _, head_dim = value_states.shape
            _, _ = self.update(key_states[save_indices].view(bsz, num_heads, -1, head_dim), value_states[save_indices].view(bsz, num_heads, -1, head_dim), layer_idx)

    self.kv_pool = {}
    return

# eviction methods
def compute_attention_sum(query_states, key_states, last_len):
    head_dim=key_states.shape[-1]
    attn_weights = torch.matmul(query_states[...,-last_len:,:], key_states.transpose(2, 3)) / math.sqrt(head_dim)
    mask = torch.full((last_len, last_len), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(attn_weights.device)
    mask = mask[None, None, :, :]
    attn_weights[:, :, -last_len:, -last_len:] += mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights_sum = attn_weights[:,:,:,:-last_len].sum(dim = -2)
    
    return attn_weights_sum.float()

def streaming_indices(prefix_len, key_states, drop_config):
    # <1ms
    max_capacity = 1024
    recent_len = max_capacity - prefix_len
    bsz, num_heads, total_len, head_dim=key_states.shape

    if  total_len <= max_capacity:
        return torch.arange(total_len).cuda()
    else:
        recent_indices = torch.tensor([total_len - recent_len + l for l in range(recent_len)], device=key_states.device)
        prefix_indices = torch.arange(prefix_len).cuda()
        return torch.cat((prefix_indices, recent_indices)).cuda()
    
def snap_indices(query_states, key_states, last_len, drop_config):
    kernel_size = drop_config['kernel_size']
    max_capacity = drop_config['max_capacity']
    pooling = drop_config['pooling']
    block = drop_config['block']

    bsz, num_heads, total_len, head_dim=key_states.shape
    observe_window = last_len if last_len < max_capacity // 2 else max_capacity // 2

    if total_len <= max_capacity:
        past_mask = torch.ones([bsz,num_heads,total_len],dtype=torch.bool,device=key_states.device)

    else: 
        attn_weights_sum = compute_attention_sum(query_states, key_states, observe_window) # TODO: add block;
        if pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = kernel_size, padding=kernel_size//2, stride=1)
        elif pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = kernel_size, padding=kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')

        # < 0.2 ms *32 
        if block == 1:
            indices = attn_cache.topk(max_capacity - observe_window, dim=-1).indices
        else:
            bs, heads, tokens = attn_weights_sum.shape
            num_groups = tokens // block
            bias = tokens - num_groups * block

            attn_weights_sum = attn_weights_sum[:, :, :num_groups * block]  # [bs, heads, tokens_trunc]
            grouped_scores = attn_weights_sum.view(bs, heads, num_groups, block)  # [bs, 32, 655, 8]
            group_scores = grouped_scores.sum(dim=3)  # [bs, heads, num_groups]
            _, group_indices = torch.topk(group_scores, k=(max_capacity - observe_window)//block, dim=2)  # [bs, heads, top_n]

            base_token_indices = torch.arange(num_groups * block, device=attn_weights_sum.device)
            token_index_table = base_token_indices.view(num_groups, block)  # [655, 8]
            indices = token_index_table[group_indices].view(bs, heads, -1)

            if bias > 0:
                tail_tokens = torch.arange(tokens - bias, tokens, device=attn_weights_sum.device)
                tail_tokens = tail_tokens.view(1, 1, -1).expand(bs, heads, -1)
                indices = torch.cat([indices, tail_tokens], dim=2)  # [bs, heads, N]
            d = time.time()
            # import pdb; pdb.set_trace()

        past_mask=torch.zeros([bsz,num_heads,total_len-observe_window],dtype=torch.bool,device=key_states.device)
        past_mask=past_mask.scatter(-1,indices,1)
        past_mask=torch.cat([past_mask,torch.ones([bsz,num_heads,observe_window],dtype=torch.bool,device=key_states.device)],dim=2)
    
    return past_mask

def get_saveindices(reuse_config, drop_config, other_config, kvgroups, query_states, key_states, value_states, past_key_value, cache_kwargs, layer_idx):
    if 'SnapKV' in drop_config['drop']: # 只有snapkv需要先repeat再保存kv
        key_states = repeat_kv(key_states, kvgroups)
        value_states = repeat_kv(value_states, kvgroups)

    if not other_config['decode']:
        if drop_config['drop'] == 'Streaming':
            drop_config['save_indices'] = streaming_indices(other_config['data_params']['prefix_len'], key_states, drop_config)

        elif 'SnapKV' in drop_config['drop']:
            drop_config['save_indices'] = snap_indices(query_states, key_states, other_config['data_params']['last_len'], drop_config)
            

    if past_key_value is not None:
        if not other_config['decode']:
            past_key_value.prepare_update_kv(key_states, value_states, drop_config['save_indices'])
            
        else:
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)

    if 'SnapKV' not in drop_config['drop']:
        key_states = repeat_kv(key_states, kvgroups)
        value_states = repeat_kv(value_states, kvgroups)

    return drop_config['save_indices'], key_states, value_states, past_key_value