# ragkv 项目配置系统分析报告

## 一、总体架构

项目的配置系统以一个名为 `extra_config` 的顶层字典为核心载体，它在推理时从**调用入口一路透传到每个 attention 层**，实现了"在不修改模型结构的前提下，通过侧信道字典控制任意层的推理行为"。

```
extra_config (顶层容器)
├── reuse_config   ← KV 复用策略（可为 None）
├── drop_config    ← KV 丢弃/蒸馏策略（可为 None）
└── other_config   ← 运行时公共状态（始终存在）
```

这三个子配置互不替代，各司其职：
| 子配置 | 控制维度 | 为 None 时 |
|---|---|---|
| `reuse_config` | 是否/如何复用预计算 KV | 不做 KV 复用 |
| `drop_config` | 是否/如何裁剪 KV Cache | 不做 KV 裁剪 |
| `other_config` | 当前是否处于 decode 阶段、样本数据参数 | 始终有值 |

---

## 二、各配置详解

### 2.1 `extra_config`（顶层容器）

**初始化位置**：[`utils.py` L189 `initialize_config()`](file:///data/ykw/projects/ragkv/utils.py#L189-L224)

```python
extra_config = {
    "reuse_config": reuse_config,   # 由 args.reuse 决定是否为 None
    "drop_config":  drop_config,    # 由 args.drop_config 决定是否为 None
    "other_config": {"decode": False},
}
```

**何时创建**：
- `eval_longbench.py`：在 `reuse != "no"` 或 `drop != False` 时创建
- `eval_needle.py` / `eval_ruler.py`：每条样本循环开始时创建

**传递链路**：
```
eval_*.py
  └─ decode() / vanilla()
       └─ decode_step()
            └─ model(extra_config=config)            # LlamaForCausalLM
                 └─ self.model(extra_config=...)     # LlamaModel
                      └─ decoder_layer(extra_config=...) # LlamaDecoderLayer
                           └─ self.self_attn(extra_config=...) # LlamaSdpaAttention
```

每层 `decoder_layer` 会返回更新后的 `extra_config`，由 `LlamaModel` 接收并传递给下一层：
```python
layer_outputs, extra_config = decoder_layer(..., extra_config=self.extra_config)
self.extra_config = extra_config
```

---

### 2.2 `reuse_config`（KV 复用配置）

**初始化位置**：[`utils.py` L191–L210](file:///data/ykw/projects/ragkv/utils.py#L191-L210)

当 `args.reuse != "no"` 时创建，包含以下字段：

| 字段 | 类型 | 说明 | 来源 |
|---|---|---|---|
| `reuse` | `str` | 复用策略名，如 `"blend"`, `"debug"`, `"surprisal_chunk"` 等 | `args.reuse` |
| `check` | `str/None` | 当前层阶段标记：`None` / `"checking"` / `"postchecking"` | `LlamaModel` 逐层设置 |
| `cat_kv` | `Tensor(2, L, H, N, D)` | 预计算的全层 KV 缓存（被逐层拆解分发） | 从 `kvs.pt` 文件加载 |
| `recomp_ratio` | `float` | 需重算 token 的比例（如 0.15） | `args.rate` |
| `causal` | `bool` | Attention mask 是否采用 causal 模式 | 固定 `True` |
| `fake_q` | `Tensor/None` | 用于对齐 RoPE 的占位 query | attention 层首次进入 checking 时创建 |
| `mask` | `Tensor/None` | 缓存的 flashinfer packed mask，checking 阶段首次计算后复用 | `create_flashinfer_mask()` 生成 |
| `imp_indices` | `LongTensor` | 被选中需重算的 token 位置（"重要 token 索引"） | `get_topindices()` 计算 |
| `org_pos` | `Tensor` | 原始 position_ids 备份（用于 old kv 做 RoPE） | `LlamaModel` 每步保存 |
| `surprisal_scores` | `Tensor` | 每 token 的 surprisal 得分（仅 `surprisal_chunk` 策略） | 从 `surprisal.pt` 加载 |
| `chunk_ranges` | `list[(int,int)]` | 文档 chunk 边界列表（仅 `surprisal_chunk` 策略） | 评测脚本写入 |
| `blend_gap_source` | `str` | blend 策略的差异计算来源（`"v"` 或 `"k"`） | `args.blend_gap_source` |
| `blend_debug_fusion` | `str` | blend_debug 联合打分方式（`"mul"/"sum"/"rank"`） | `args.blend_debug_fusion` |
| `blend_debug_alpha` | `float` | blend_debug 加权融合系数 | 固定 `0.5` |

#### reuse_config 的生命周期（一次 prefill 过程）

```
Layer 0:
  LlamaModel: cat_kv = full_kv[:, 0, ...]  # 截取第0层
  LlamaModel: check = None
  → attention: reuse 逻辑不触发

Layer check_layer (由 get_layer(reuse_method) 确定):
  LlamaModel: check = "checking"
  → attention:
      1. old_k/v = cat_kv + RoPE 对齐
      2. top_indices = get_topindices(...)   ← 核心选 token 逻辑
      3. query_states = query_states[:, :, top_indices, :]  ← 裁剪 query
      4. imp_indices = top_indices
      check 后: LlamaModel 将 position_ids 改为 imp_indices
      check = 'postchecking'

Layer check_layer+1 ~ last:
  check = "postchecking"
  → attention:
      old_k/v 中 imp_indices 位置替换为新算的 k/v
      其余位置沿用 old_k/v（即旧缓存）
```

#### reuse 策略（`args.reuse` 可选值）

| 策略名 | 选 token 逻辑 | 特征 |
|---|---|---|
| `blend` | value 新旧差异平方和最大的 top-k + 末尾 last_len | 按 V gap 选重算点 |
| `blend_debug` | V gap + 注意力权重联合打分 | 双信号融合 |
| `debug` | query 对历史 key 的加权注意力 top-k + 末尾 | 纯注意力驱动 |
| `tail_ratio` | 按比例保留文档尾部 + 全部 query 区 | 最简单的位置策略 |
| `full` | 仅保留末尾 last_len（全量 old kv） | 零重算 |
| `cat` | 仅保留最后 1 个 token | 极端压缩 |
| `attnlink` | 直接使用预给的 sink_pos | 外部指定 sink |
| `surprisal_chunk` | 按 chunk 分配预算后各 chunk 内取 surprisal 最高 top-k | 利用困难度信号 |

**check_layer 的确定**：由 [`reuse_utils.py` `get_layer()`](file:///data/ykw/projects/ragkv/models/reuse_utils.py#L19-L29) 决定：
- `blend/blend_debug/debug` 类策略 → 第 1 层
- `tail_ratio` → 第 0 层
- 其他 → 第 0 层

---

### 2.3 `drop_config`（KV 丢弃配置）

**初始化位置**：[`utils.py` L212–L216](file:///data/ykw/projects/ragkv/utils.py#L212-L216)

当 `args.drop_config` 不为 None 时从 JSON 文件加载，并追加 `drop` 字段：

```python
drop_config = json.load(open(args.drop_config, "r"))
drop_config["drop"] = args.drop   # 来自命令行参数
```

**JSON 文件示例**（[`config/drop/snap1.json`](file:///data/ykw/projects/ragkv/config/drop/snap1.json)）：
```json
{
    "kernel_size": 5,      # avg/max 池化的卷积核大小
    "max_capacity": 1024,  # KV cache 最大保留 token 数
    "pooling": "avgpool",  # 池化方式: avgpool / maxpool
    "block": 1             # 分组 block 大小（1=不分组）
}
```

**所有字段**：

| 字段 | 来源 | 说明 |
|---|---|---|
| `drop` | `args.drop` | 丢弃策略: `"Streaming"`, `"SnapKV"`, `"SnapKV_block"` 等 |
| `kernel_size` | JSON 文件 | 注意力得分池化核大小 |
| `max_capacity` | JSON 文件 | 最多保留的 KV token 数 |
| `pooling` | JSON 文件 | 池化方式 (`avgpool`/`maxpool`) |
| `block` | JSON 文件 | 按块分组大小（>1 时为 block-wise eviction） |
| `save_indices` | 运行时写入 | 当前层计算出的保留 token 索引/mask，供下次复用 |

#### drop 策略：

- **`Streaming`**：保留前缀 `prefix_len` 个 token + 末尾 `max_capacity - prefix_len` 个 token，类似 StreamingLLM
- **`SnapKV`**：用末尾 query 对全序列 key 做注意力打分（可 avg/max pool 平滑），选 top-`max_capacity`，每个头独立选（返回 bool mask）

**执行位置**：[`evict_utils.py` `get_saveindices()`](file:///data/ykw/projects/ragkv/models/evict_utils.py#L135-L159)

prefill 阶段每层将 k/v 和待保留 indices 注册进 `past_key_value.kv_pool`；第一个 decode token 生成后执行 `update_after_prefill()` 统一截断，之后 decode 阶段直接 `past_key_value.update()` 常规更新。

`drop_config` 与 `reuse_config` 可**同时激活**，`get_saveindices()` 在 reuse 的选 token 之后执行：
```
reuse 选出 imp_indices
  → drop 在此基础上进一步控制 cache 截断
```

---

### 2.4 `other_config`（运行时公共状态）

**初始化位置**：[`utils.py` L221](file:///data/ykw/projects/ragkv/utils.py#L221)

```python
"other_config": {"decode": False}
```

**字段说明**：

| 字段 | 类型 | 说明 | 来源 |
|---|---|---|---|
| `decode` | `bool` | `False`=prefill 阶段，`True`=decode 阶段 | `decode()` 在生成第一个 token 后设为 `True` |
| `data_params` | `dict` | 当前样本的分段参数 | 评测脚本从数据集 `params` 注入 |

`data_params` 的内部字段（由数据 loader 生成）：

| 字段 | 说明 | 使用方 |
|---|---|---|
| `prefix_len` | 前缀（系统提示）长度 | `Streaming` eviction、`debug` reuse 策略 |
| `last_len` | query（问题）的 token 数量 | 所有 reuse 策略用于"强制保留末尾段" |
| `doc_start_len` | 每个 chunk 头部 overlap 长度 | chunk KV 切分 |
| `sink_pos` | attnlink 策略下的 sink token 位置 | `attnlink` reuse 策略 |

**`decode` 标志的关键作用**（[`llama.py` L193–L198](file:///data/ykw/projects/ragkv/models/llama/llama.py#L193-L198)）：
```python
# decode 阶段不再分发 cat_kv（old cache 不再有意义）
self.extra_config['reuse_config']['cat_kv'] = (
    cat_kv[:, i, :, :, :] if not self.extra_config['other_config']['decode'] else None
)
# decode 阶段不再执行 checking/postchecking 逻辑
if not self.extra_config['other_config']['decode'] and i == check_layer:
    ...
```

---

## 三、配置流转总图

```
命令行参数 (args)
│
├─ args.reuse ─────────────────────────────────→ reuse_config['reuse']
│   └─ args.rate ────────────────────────────→ reuse_config['recomp_ratio']
│   └─ args.blend_gap_source ───────────────→ reuse_config['blend_gap_source']
│   └─ args.blend_debug_fusion ─────────────→ reuse_config['blend_debug_fusion']
│
├─ args.drop_config (JSON文件路径)
│   └─ JSON 内容 ────────────────────────────→ drop_config (kernel_size/max_capacity/pooling/block)
│   └─ args.drop ───────────────────────────→ drop_config['drop']
│
└─ initialize_config(args) ─────────────────→ extra_config = {reuse, drop, other}
                                                      │
                      eval_*.py 注入样本参数 ──────────→ other_config['data_params'] = params
                      eval_*.py 加载预算KV ────────────→ reuse_config['cat_kv'] = load_kv(...)
                      eval_*.py 加载surprisal ──────────→ reuse_config['surprisal_scores/chunk_ranges']
                                                      │
                             decode(args, model, ..., extra_config)
                                                      │
                                           LlamaForCausalLM.forward
                                                      │
                                              LlamaModel.forward
                                           ┌──────────────────────┐
                                           │ 每层循环:             │
                                           │  1. 按层拆分 cat_kv   │
                                           │  2. 设置 check 阶段   │
                                           │  3. 调用 decoder_layer│
                                           │  4. 更新 imp_indices  │
                                           │  5. 修改 position_ids │
                                           └──────────────────────┘
                                                      │
                                          LlamaDecoderLayer.forward
                                          (将 extra_config 传给 attention)
                                                      │
                                         LlamaSdpaAttention.forward
                                         ┌────────────────────────┐
                                         │ reuse_config:          │
                                         │  checking → get_top    │
                                         │  postchecking → merge  │
                                         │ drop_config:           │
                                         │  → get_saveindices     │
                                         └────────────────────────┘
                                                      │
                                         第一个 decode token 生成后:
                                           other_config['decode'] = True
                                           reuse_config['check'] = None
                                           reuse_config['cat_kv'] = None
```

---

## 四、各配置作用位置汇总

| 配置字段 | 作用文件 | 具体位置 | 作用 |
|---|---|---|---|
| `reuse_config['cat_kv']` | `llama.py` | `LlamaModel.forward` L187, L193 | 按层拆分分发历史 KV |
| `reuse_config['check']` | `llama.py` | `LlamaModel.forward` L194–198 | 控制 checking/postchecking 阶段切换 |
| `reuse_config['imp_indices']` | `llama.py` | `LlamaModel.forward` L232 | 更新 position_ids 以便后续层只关注重要位置 |
| `reuse_config.check == 'checking'` | `llama.py` | `LlamaSdpaAttention.forward` L402 | 调用 `get_topindices()` 选重要 token，裁剪 query |
| `reuse_config.check == 'postchecking'` | `llama.py` | `LlamaSdpaAttention.forward` L424 | old kv 重要位置替换为新算的 k/v |
| `reuse_config['mask']` | `llama.py` | `LlamaSdpaAttention.forward` L478 | 用打包的 flashinfer mask 做自定义 attention |
| `drop_config` | `llama.py` | `LlamaSdpaAttention.forward` L437 | 调用 `get_saveindices()` 执行 eviction |
| `drop_config['save_indices']` | `evict_utils.py` | `update_after_prefill()` | prefill 结束后裁断 KV cache |
| `other_config['decode']` | `llama.py` | `LlamaModel.forward` L193 | prefill/decode 两阶段行为切换的总开关 |
| `other_config['data_params']` | `reuse_utils.py` / `evict_utils.py` | `get_topindices()`, `get_saveindices()` | 提供 prefix_len、last_len 等段落长度参数 |

---

## 五、关键设计模式

1. **字典作为侧信道**：`extra_config` 以返回值形式在层间传递，无需修改模型架构，任意层可读写其中状态，实现灵活的层间协作。

2. **inplace 修改 + 下一层继续读取**：`imp_indices`、`mask`、`cat_kv` 等字段在 attention 层内被更新，写回 `extra_config` 后，下一层（或下一个 decode step）直接读取，实现跨层状态传递。

3. **prefill/decode 两阶段模式**：`other_config['decode']` 是总开关；prefill 阶段执行 checking/postchecking 逻辑并建立 save_indices；第一个 decode token 生成后统一清除临时状态（`check=None`, `cat_kv=None`，`decode=True`），进入纯 decode 路径。

4. **延迟 KV 裁断**：`drop_config` 不在 prefill 时立即截断 cache，而是先用 `prepare_update_kv()` 暂存全量 KV 和 save_indices，待第一个 decode token 生成后由 `update_after_prefill()` 一次性截断，避免多次内存拷贝。
