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
    if reuse_method == 'blend_debug':
        return 1
    if reuse_method == 'tail_ratio':
        return 0
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


def minmax_norm(x, eps=1e-12):
    if x.numel() == 0:
        return x
    x = x.to(torch.float32)
    x_min = torch.min(x)
    x_max = torch.max(x)
    denom = x_max - x_min
    if torch.abs(denom) <= eps:
        return torch.zeros_like(x)
    return (x - x_min) / (denom + eps)


def rank_norm(x):
    if x.numel() == 0:
        return x
    x = x.to(torch.float32)
    n = x.numel()
    if n == 1:
        return torch.ones_like(x)
    order = torch.argsort(x, descending=True)
    ranks = torch.empty(n, device=x.device, dtype=torch.float32)
    ranks[order] = torch.arange(n, device=x.device, dtype=torch.float32)
    return 1.0 - ranks / (n - 1)


def fuse_blend_debug_scores(blend_score, debug_score, method, alpha):
    if method == 'mul':
        return minmax_norm(blend_score) * minmax_norm(debug_score)
    if method == 'sum':
        return alpha * minmax_norm(blend_score) + (1.0 - alpha) * minmax_norm(debug_score)
    if method == 'rank':
        return alpha * rank_norm(blend_score) + (1.0 - alpha) * rank_norm(debug_score)
    raise ValueError(f"unsupported blend_debug_fusion: {method}")

def compute_blend_gap_score(
    gap_source,
    key_states,
    value_states,
    key_old,
    value_old,
    start,
    end,
):
    if gap_source == "v":
        diff = value_states[:, :, start:end, :] - value_old[:, :, start:end, :]
    elif gap_source == "k":
        diff = key_states[:, :, start:end, :] - key_old[:, :, start:end, :]
    else:
        raise ValueError(f"unsupported blend_gap_source: {gap_source}")
    return torch.sum(diff ** 2, dim=[0, 1, 3])


def get_topindices(
    reuse_config,
    other_config,
    query_states,
    key_states,
    value_states,
    key_old,
    value_old,
    kvgroups,
):
    """
    根据 `reuse_config['reuse']` 指定的策略，选出需要“重新计算/重点保留”的 token 索引。

    参数说明：
    - reuse_config: 复用策略配置，至少包含 `reuse` 和可能用到的 `recomp_ratio` 等字段。
    - other_config: 运行时数据配置，常用字段在 `other_config['data_params']`：
      - `last_len`: 末尾 query 段长度（通常会被强制保留）
      - `prefix_len`: 前缀长度（debug 策略中用于索引偏移）
      - `sink_pos`: attnlink 策略下直接复用的重要索引
    - query_states/key_states/value_states: 当前层 Q/K/V 状态张量。
    - key_old/value_old: 历史 K/V（用于 blend 相关策略比较变化幅度）。
    - kvgroups: KV 分组数（debug 策略中用于扩展 key head 维度）。

    返回：
    - top_indices: 1D Long Tensor，表示被选中的 token 位置索引。
    """
    if reuse_config['reuse'] == 'blend_debug':
        last_len = other_config['data_params']['last_len']
        prefix_len = other_config['data_params']['prefix_len']
        total_len = value_states.shape[2]
        doc_start = max(0, min(prefix_len, total_len))
        doc_end = max(doc_start, total_len - last_len)
        doc_len = doc_end - doc_start

        question_indices = torch.arange(doc_end, total_len, device=value_states.device)
        if doc_len <= 0:
            return question_indices

        topk_num = int(doc_len * reuse_config['recomp_ratio'])
        topk_num = max(0, min(topk_num, doc_len))

        gap_source = reuse_config.get("blend_gap_source", "v")
        blend_score_doc = compute_blend_gap_score(
            gap_source,
            key_states,
            value_states,
            key_old,
            value_old,
            doc_start,
            doc_end,
        )

        debug_score_doc = compute_attention_sum_wo_head(
            query_states,
            repeat_kv(key_states[:, :, doc_start:, :], kvgroups),
            last_len,
        ).reshape(-1)
        if debug_score_doc.numel() != doc_len:
            raise ValueError(
                f'blend_debug score length mismatch: debug={debug_score_doc.numel()} vs doc_len={doc_len}'
            )

        fusion_method = reuse_config.get('blend_debug_fusion', 'mul')
        alpha = float(reuse_config.get('blend_debug_alpha', 0.5))
        fused_score_doc = fuse_blend_debug_scores(
            blend_score_doc,
            debug_score_doc,
            fusion_method,
            alpha,
        )

        if topk_num > 0:
            doc_top_local = torch.topk(fused_score_doc, k=topk_num, dim=-1).indices
            doc_top_local, _ = torch.sort(doc_top_local)
            doc_indices = doc_top_local + doc_start
        else:
            doc_indices = torch.empty(0, dtype=torch.long, device=value_states.device)

        top_indices = torch.cat([doc_indices, question_indices], dim=0)
        top_indices = torch.unique(top_indices, sorted=True)

    elif 'blend' in reuse_config['reuse']:
        # blend: 在文档区(去掉末尾 last_len)中，按 value 新旧差异的平方和选 top-k；
        # 然后无条件拼接末尾 last_len（保证最近 token 始终保留）。
        last_len = other_config['data_params']['last_len']
        prefix_len = other_config['data_params']['prefix_len']
        total_len = value_states.shape[2]
        doc_start = max(0, min(prefix_len, total_len))
        doc_end = max(doc_start, total_len - last_len)
        doc_len = doc_end - doc_start
        question_indices = torch.arange(doc_end, total_len, device=value_states.device)

        if doc_len <= 0:
            return question_indices

        topk_num = int(doc_len * reuse_config['recomp_ratio'])
        topk_num = max(0, min(topk_num, doc_len))
        gap_source = reuse_config.get("blend_gap_source", "v")
        temp_diff = compute_blend_gap_score(
            gap_source,
            key_states,
            value_states,
            key_old,
            value_old,
            doc_start,
            doc_end,
        )
        if topk_num > 0:
            doc_top_local = torch.topk(temp_diff, k=topk_num).indices
            doc_top_local, _ = torch.sort(doc_top_local)
            doc_indices = doc_top_local + doc_start
        else:
            doc_indices = torch.empty(0, dtype=torch.long, device=value_states.device)

        top_indices = torch.cat([doc_indices, question_indices], dim=0)
        top_indices = torch.unique(top_indices, sorted=True)

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

    elif reuse_config['reuse'] == 'tail_ratio':
        # tail_ratio:
        # - 在文档区 [prefix_len, total_len-last_len) 中按比例取尾部 token；
        # - 始终附带 query 区 [total_len-last_len, total_len)；
        # - 当 floor(doc_len * ratio)=0 时，退化为仅 query 区。
        last_len = other_config['data_params']['last_len']
        prefix_len = other_config['data_params']['prefix_len']
        total_len = value_states.shape[2]
        doc_start = max(0, min(prefix_len, total_len))
        doc_end = max(doc_start, total_len - last_len)
        doc_len = doc_end - doc_start

        ratio = float(reuse_config['recomp_ratio'])
        tail_k = int(math.floor(doc_len * ratio))
        tail_k = max(0, min(tail_k, doc_len))

        question_indices = torch.arange(doc_end, total_len, device=value_states.device)
        if tail_k > 0:
            doc_tail_indices = torch.arange(
                doc_end - tail_k, doc_end, device=value_states.device
            )
            top_indices = torch.cat([doc_tail_indices, question_indices], dim=0)
        else:
            top_indices = question_indices

        top_indices = torch.unique(top_indices, sorted=True)

    elif 'debug' in reuse_config['reuse']:
        # debug:
        # 1) 先估计末尾 query 对历史 key 的注意力总量；
        # 2) 用 1D 平均池化平滑得分；
        # 3) 在文档区中按得分选 top-k；
        # 4) 再拼接末尾 last_len，保证近期 token 全部包含。
        kernel_size = 5
        last_len = other_config['data_params']['last_len']
        prefix_len = other_config['data_params']['prefix_len']
        total_len = value_states.shape[2]
        doc_start = max(0, min(prefix_len, total_len))
        doc_end = max(doc_start, total_len - last_len)
        doc_len = doc_end - doc_start
        question_indices = torch.arange(doc_end, total_len, device=value_states.device)

        if doc_len <= 0:
            return question_indices

        # 用“末尾 last_len 个 query”去评估历史 token 的被关注程度：
        # 1) key 侧先去掉 prefix 区，避免前缀模板 token 干扰排序；
        # 2) repeat_kv 将 KV 头扩展到与 query 头数对齐，便于统一做注意力打分；
        # 3) 返回的 attn_weights_sum 是按 token 位置聚合后的重要性分数。
        attn_weights_sum = compute_attention_sum_wo_head(
            query_states,
            repeat_kv(key_states[:, :, doc_start:, :], kvgroups),
            last_len
        )
        attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        topk_num = int(doc_len * reuse_config['recomp_ratio'])
        topk_num = max(0, min(topk_num, doc_len))

        if topk_num > 0:
            doc_top_local = attn_cache.topk(topk_num, dim=-1).indices.squeeze(0)
            doc_indices = doc_top_local + doc_start
        else:
            doc_indices = torch.empty(0, dtype=torch.long, device=value_states.device)

        top_indices = torch.cat([doc_indices, question_indices], dim=0)
        top_indices = torch.unique(top_indices, sorted=True)

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
    """
    计算“末尾 query 段”对“历史 key 段（不含末尾 query 自身）”的注意力总量。

    该函数用于为重算策略提供打分信号：
    - 输入的 Q/K 原始形状通常为 [bs, num_heads, seq_len, head_dim]；
    - 先把 head 维与特征维合并，得到“无 head 区分”的表示；
    - 再只取最后 `last_len` 个 query 作为查询端，和全量 key 做点积；
    - 对 query-query 子块施加因果 mask，避免未来信息泄露；
    - softmax 后沿 query 维求和，得到每个历史 token 被末尾 query 关注的总强度。

    返回：
    - attn_weights_sum: shape [1, seq_len - last_len]
      表示每个“历史 token”累计收到的注意力分数（越大通常越重要）。
    """
    # [bs, h, q_len, d] -> [1, q_len, h*d]
    # 将多头展平成单一特征维，便于后续统一计算“无 head”注意力。
    query_states = query_states.permute(0, 2, 1, 3).reshape(1, query_states.shape[-2], -1)
    # [bs, h_kv, k_len, d] -> [1, k_len, h_kv*d]
    # key 侧同样做 head 合并，保持与 query 的最后一维可对齐。
    key_states = key_states.permute(0, 2, 1, 3).reshape(1, key_states.shape[-2], -1)

    # 缩放点积注意力中的尺度项 sqrt(dim)。
    dim=key_states.shape[-1]
    # 仅使用最后 last_len 个 query 与全量 key 计算相似度：
    # 结果形状 [1, last_len, k_len]。
    attn_weights = torch.matmul(query_states[:,-last_len:,:], key_states.transpose(1,2)) / math.sqrt(dim)
    # 为“末尾 query 与末尾 key 的子矩阵”构造下三角因果 mask：
    # 非法位置填充为极小值，softmax 后近似 0。
    mask = torch.full((last_len, last_len), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)

    mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    # 仅保留下三角（含对角线）可见，其余位置保持极小值。
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(attn_weights.device)
    mask = mask[None, :, :]
    # 把 mask 叠加到右下角 query-query 子块（对应最后 last_len tokens）。
    attn_weights[:, -last_len:, -last_len:] += mask
    # softmax 归一化到概率分布；内部用 fp32 更稳定，再转回原 dtype。
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # 仅统计“历史 key 区（去掉最后 last_len）”被关注程度：
    # 在 query 维（-2）求和，得到每个历史位置的累计注意力。
    attn_weights_sum = attn_weights[:,:,:-last_len].sum(dim = -2)

    # 返回 float，供上层 top-k 选择逻辑直接使用。
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
