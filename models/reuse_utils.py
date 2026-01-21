import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import time
from flashinfer.quantization import packbits

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

def get_layer(reuse_method):
    return 1

def get_topindices(reuse_config, other_config, query_states, key_states, value_states, value_old, kvgroups):
    # Due to permission restrictions from our collaborators, the core code is temporarily unavailable.

    return top_indices

def compute_attention_sum_wo_head(query_states, key_states, last_len):
    # Due to permission restrictions from our collaborators, the core code is temporarily unavailable.

    return attn_weights_sum.float()

def create_flashinfer_mask(query_state, key_state, indices):
    # Due to permission restrictions from our collaborators, the core code is temporarily unavailable.

    return custom_mask, packed_mask
