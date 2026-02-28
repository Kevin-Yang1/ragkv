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
    if 'blend' in reuse_method:
        return 1 # compute imp indices at layer 1
    if 'debug' in reuse_method:
        return 1
    else:
        return 0

def allocate_by_largest_remainder(lengths, total_k):
    if total_k <= 0:
        return [0] * len(lengths)
    denom = sum(lengths)
    if denom <= 0:
        return [0] * len(lengths)

    raw = [total_k * l / denom for l in lengths]
    alloc = [int(math.floor(v)) for v in raw]
    remain = total_k - sum(alloc)
    if remain > 0:
        order = sorted(
            range(len(lengths)),
            key=lambda idx: (raw[idx] - alloc[idx], lengths[idx], -idx),
            reverse=True,
        )
        for idx in order[:remain]:
            alloc[idx] += 1
    return alloc

def get_topindices(reuse_config, other_config, query_states, key_states, value_states, value_old, kvgroups):
    """
    根据 `reuse_config['reuse']` 指定的策略，选出需要“重新计算/重点保留”的 token 索引。

    参数说明：
    - reuse_config: 复用策略配置，至少包含 `reuse` 和可能用到的 `recomp_ratio` 等字段。
    - other_config: 运行时数据配置，常用字段在 `other_config['data_params']`：
      - `last_len`: 末尾 query 段长度（通常会被强制保留）
      - `prefix_len`: 前缀长度（debug 策略中用于索引偏移）
      - `sink_pos`: attnlink 策略下直接复用的重要索引
    - query_states/key_states/value_states: 当前层 Q/K/V 状态张量。
    - value_old: 历史 value（用于 blend 策略下比较变化幅度）。
    - kvgroups: KV 分组数（debug 策略中用于扩展 key head 维度）。

    返回：
    - top_indices: 1D Long Tensor，表示被选中的 token 位置索引。
    """
    if 'blend' in reuse_config['reuse']:
        # blend: 在文档区(去掉末尾 last_len)中，按 value 新旧差异的平方和选 top-k；
        # 然后无条件拼接末尾 last_len（保证最近 token 始终保留）。
        last_len = other_config['data_params']['last_len']
        total_len = value_states.shape[2]
        last_indices = [total_len - last_len + l for l in range(last_len)]

        topk_num = int((total_len - last_len) * reuse_config['recomp_ratio'])

        temp_diff = torch.sum((value_states[:, :, :-last_len, :] - value_old[:, :, :-last_len, :]) ** 2, dim=[0, 1, 3])
        top_indices = torch.topk(temp_diff, k=topk_num).indices  # shape: [topk_num]
        top_indices, _ = torch.sort(top_indices)
        top_indices = torch.cat([top_indices, torch.tensor(last_indices, device=top_indices.device)])

    elif 'attnlink' in reuse_config['reuse']:
        # attnlink: 直接使用预先给定的 sink 位置作为重要索引。
        reuse_config['imp_indices'] = other_config['data_params']['sink_pos']
        top_indices = reuse_config['imp_indices']

    elif 'full' in reuse_config['reuse']:
        # full: 仅保留末尾 last_len。
        last_len = other_config['data_params']['last_len']
        total_len = value_states.shape[2]
        last_indices = [total_len - last_len + l for l in range(last_len)]

        top_indices = torch.tensor(last_indices, device=value_states.device)

    elif 'cat' in reuse_config['reuse']:
        # cat: 仅保留最后一个 token。
        last_len = other_config['data_params']['last_len']
        total_len = value_states.shape[2]
        last_indices = [total_len - 1]

        top_indices = torch.tensor(last_indices, device=value_states.device)

    elif 'debug' in reuse_config['reuse']:
        # debug:
        # 1) 先估计末尾 query 对历史 key 的注意力总量；
        # 2) 用 1D 平均池化平滑得分；
        # 3) 在文档区中按得分选 top-k；
        # 4) 再拼接末尾 last_len，保证近期 token 全部包含。
        kernel_size = 5
        last_len = other_config['data_params']['last_len']
        total_len = value_states.shape[2]

        attn_weights_sum = compute_attention_sum_wo_head(
            query_states,
            repeat_kv(key_states[:, :, other_config['data_params']['prefix_len']:, :], kvgroups),
            last_len
        )
        attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        topk_num = int((total_len - last_len) * reuse_config['recomp_ratio'])

        top_indices = attn_cache.topk(topk_num, dim=-1).indices.squeeze(0) + other_config['data_params']['prefix_len']
        total_len = value_states.shape[2]
        last_indices = torch.arange(total_len - last_len, total_len, device=top_indices.device)

        top_indices = torch.cat([top_indices, last_indices])

    elif 'surprisal_chunk' in reuse_config['reuse']:
        # surprisal_chunk:
        # - 将文档区按 chunk 切分；
        # - 先按 chunk 长度比例分配预算（largest remainder）；
        # - 每个 chunk 内按 surprisal 选 top；
        # - 最后拼接 question 区(末尾 last_len)并去重排序。
        last_len = other_config['data_params']['last_len']
        total_len = value_states.shape[2]
        doc_end = total_len - last_len

        surprisal_scores = reuse_config.get('surprisal_scores', None)
        chunk_ranges = reuse_config.get('chunk_ranges', None)
        if surprisal_scores is None or chunk_ranges is None:
            raise ValueError('surprisal_chunk requires surprisal_scores and chunk_ranges')

        surprisal_scores = torch.as_tensor(surprisal_scores, device=value_states.device, dtype=torch.float32)
        if surprisal_scores.numel() != total_len:
            raise ValueError(
                f'surprisal length mismatch: {surprisal_scores.numel()} vs total_len={total_len}'
            )

        # 校验每个 chunk 的范围合法，并统计各 chunk 长度。
        lengths = []
        for start, end in chunk_ranges:
            if not (0 <= start <= end <= doc_end):
                raise ValueError(f'invalid chunk range ({start}, {end}) for doc_end={doc_end}')
            lengths.append(end - start)
        doc_total = sum(lengths)
        if doc_total <= 0:
            return torch.arange(doc_end, total_len, device=value_states.device)

        topk_num = int(doc_total * reuse_config['recomp_ratio'])
        topk_num = max(0, min(topk_num, doc_total))

        # 预算按 chunk 分配后，在 chunk 内各自取 top，避免长 chunk 完全占满预算。
        chunk_budget = allocate_by_largest_remainder(lengths, topk_num)
        selected_doc_indices = []
        for (start, end), budget in zip(chunk_ranges, chunk_budget):
            if budget <= 0:
                continue
            chunk_scores = surprisal_scores[start:end]
            local_top = torch.topk(chunk_scores, k=budget, dim=-1).indices + start
            selected_doc_indices.append(local_top)

        if selected_doc_indices:
            doc_indices = torch.cat(selected_doc_indices, dim=0)
        else:
            doc_indices = torch.empty(0, dtype=torch.long, device=value_states.device)

        question_indices = torch.arange(doc_end, total_len, device=value_states.device)
        top_indices = torch.cat([doc_indices, question_indices], dim=0)
        top_indices = torch.unique(top_indices, sorted=True)

    return top_indices

def compute_attention_sum_wo_head(query_states, key_states, last_len):
    query_states = query_states.permute(0, 2, 1, 3).reshape(1, query_states.shape[-2], -1)
    key_states = key_states.permute(0, 2, 1, 3).reshape(1, key_states.shape[-2], -1)

    dim=key_states.shape[-1]
    attn_weights = torch.matmul(query_states[:,-last_len:,:], key_states.transpose(1,2)) / math.sqrt(dim)
    mask = torch.full((last_len, last_len), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)

    mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(attn_weights.device)
    mask = mask[None, :, :]
    attn_weights[:, -last_len:, -last_len:] += mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights_sum = attn_weights[:,:,:-last_len].sum(dim = -2)

    return attn_weights_sum.float()

def create_flashinfer_mask(query_state, key_state, indices, mode):
    """
    为 flashinfer.single_prefill_with_kv_cache 生成布尔 mask 并进行打包。

    参数：
    - q_len: query 序列长度
    - k_len: key 序列长度
    - indices: list[int]，每个 query token 可关注的 key 长度
    - mode: 'causal' or 'rightbottom'

    返回：
    - custom_mask: torch.BoolTensor, shape [q_len, k_len]
    - packed_mask: torch.ByteTensor, shape [ceil(q_len * k_len / 8)]
    """
    q_len, k_len = query_state.shape[2], key_state.shape[2]

    device = 'cuda'
    custom_mask = torch.zeros((q_len, k_len), dtype=torch.bool, device=device)

    if mode:
        # 每个 token 只能看前 index 个 key
        for i, index in enumerate(indices):
            custom_mask[i, :index+1] = True

    else:
        # 对角右下遮蔽逻辑，例如 tril 结构
        for i in range(q_len):
            shift = i - q_len + 1
            if shift < 0:
                custom_mask[i, :k_len + shift] = True
            else:
                custom_mask[i, :] = True

    # 打包 mask 成 packed_custom_mask（1D uint8）
    packed_mask = packbits(custom_mask.view(-1), bitorder="little")
    return custom_mask, packed_mask
