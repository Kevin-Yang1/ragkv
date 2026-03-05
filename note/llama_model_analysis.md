# Llama 模型模块深度分析

> 文件路径：`models/llama/llama_precompute.py` & `models/llama/llama.py`
> 分析日期：2026-03-04

---

## 概览

本项目在 HuggingFace 标准 LLaMA 实现的基础上进行了深度定制，引入了两套核心机制：

1. **KV Cache 丢弃/压缩（Drop/Evict）**：通过 `drop_config` 控制，在 prefill 阶段选择性地保存重要 KV，降低显存占用。
2. **KV 复用（Reuse）**：通过 `reuse_config` 控制，在当前 token 解码时复用预先计算好的历史 KV，避免重复计算。

两个文件分别承担不同角色：
- `llama_precompute.py`：完整保留 HuggingFace 标准类定义（`LlamaModel`, `LlamaForCausalLM_Precompute` 等），新增了 `reuse_config` 参数透传，为 **预计算（Precompute）流程** 提供模型基础。
- `llama.py`：以**函数替换（Monkey Patch）**的方式覆盖标准类的 `forward` 方法，集成了完整的 Drop 和 Reuse 逻辑，用于 **推理/生成流程**。

---

## 文件一：`llama_precompute.py`

### 1. 文件结构总览

```
llama_precompute.py (1529 行)
├── 辅助函数
│   ├── create_mask()            # 构建自定义注意力掩码
│   ├── rotate_half()            # RoPE 辅助
│   ├── apply_rotary_pos_emb()   # 应用旋转位置编码
│   └── repeat_kv()              # KV 头扩展 (GQA 支持)
├── 归一化层
│   └── LlamaRMSNorm             # RMS 归一化
├── 位置编码
│   ├── LlamaRotaryEmbedding     # 标准/动态 RoPE
│   ├── LlamaLinearScalingRotaryEmbedding  (已废弃)
│   └── LlamaDynamicNTKScalingRotaryEmbedding (已废弃)
├── 前馈网络
│   └── LlamaMLP                 # SwiGLU 门控 MLP
├── 注意力模块
│   ├── LlamaAttention           # Eager 实现
│   ├── LlamaFlashAttention2     # Flash Attention 实现
│   └── LlamaSdpaAttention       # SDPA 实现 (含 hack_kv)
├── 解码层
│   └── LlamaDecoderLayer        # 单层 Transformer 块
├── 基础模型
│   ├── LlamaPreTrainedModel     # 基类
│   └── LlamaModel               # 核心 Transformer (含 reuse_config)
└── 任务头
    ├── LlamaForCausalLM_Precompute  # 因果语言模型（预计算专用）
    ├── LlamaForSequenceClassification
    ├── LlamaForQuestionAnswering
    └── LlamaForTokenClassification
```

---

### 2. 关键组件详解

#### 2.1 `create_mask()` — 自定义注意力掩码

```python
def create_mask(query_state, key_state, indices, mode):
```

| 参数 | 说明 |
|------|------|
| `indices` | 每个 query token 可关注的 key 的边界位置列表 |
| `mode='causal'` | 因果掩码：每个 token 只能看到 `indices[i]` 之前的 key |
| `mode='rightbottom'` | 右下角因果掩码：标准下三角掩码 |

**设计意图**：当 `reuse_config` 中重用部分历史 KV（来自其他时间步）时，注意力范围不再是简单的"前缀全可见"，需要按 token 自定义可见范围。`causal` 模式正是为此服务。

---

#### 2.2 `LlamaRotaryEmbedding` — 旋转位置编码

标准 RoPE 实现，支持：
- `"default"`：标准固定频率
- `"linear"`：线性缩放（已废弃子类）
- `"dynamic"`：动态 NTK 缩放（已废弃子类），超过缓存长度时自动重算 `inv_freq`

**关键细节**：
- 使用 `torch.autocast` 的 `float32` 保证精度
- `attention_scaling` 用于 YaRN 等高级 RoPE 类型的后处理缩放

---

#### 2.3 `LlamaMLP` — 门控前馈网络

实现 SwiGLU 激活：

```
output = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
```

支持 `pretraining_tp > 1` 的张量并行切分。

---

#### 2.4 注意力模块比较

| 类名 | 后端实现 | 特殊机制 |
|------|----------|---------|
| `LlamaAttention` | 手动矩阵乘法 | 标准实现，支持 `output_attentions` |
| `LlamaFlashAttention2` | Flash Attention | 不支持 `output_attentions`，不兼容 `StaticCache` |
| `LlamaSdpaAttention` | `torch.nn.functional.scaled_dot_product_attention` | 新增 `self.hack_kv` 缓存原始 KV（RoPE 前），支持 `reuse_config` |

**`LlamaSdpaAttention` 的核心改动**（第 588 行）：

```python
self.hack_kv = [key_states.clone(), value_states.clone()]
```

在应用 RoPE **之前** 保存 key/value，这使得外部代码可以访问未经位置编码的原始 KV，是预计算流程的关键 hook。

---

#### 2.5 `LlamaDecoderLayer` — 解码层

标准 Pre-Norm Transformer 块：

```
x → RMSNorm → Self-Attention → + residual → RMSNorm → MLP → + residual
```

相较于原版，增加了 `reuse_config` 参数透传至 `self_attn.forward()`。

---

#### 2.6 `LlamaModel.forward()` — 模型前向（预计算版本）

关键差异：新增 `reuse_config: dict = None` 参数，并在每层调用时传入：

```python
layer_outputs = decoder_layer(
    ...,
    reuse_config=self.reuse_config, # 透传给每层 attention
)
```

这使得预计算流程能在正向传播过程中，将 reuse 配置传递给每个注意力层，让其决定如何利用缓存的 KV。

---

#### 2.7 `LlamaForCausalLM_Precompute` — 预计算专用因果语言模型

```python
class LlamaForCausalLM_Precompute(LlamaPreTrainedModel, GenerationMixin):
```

与标准 `LlamaForCausalLM` 的唯一区别：`forward()` 接受并传递 `reuse_config` 参数。这个类是 `precompute.py` 加载模型时使用的专用入口。

---

## 文件二：`llama.py`

### 1. 文件结构总览

```
llama.py (501 行)
├── 导入与 Monkey Patch
│   ├── DynamicCache.__init__ = new_init
│   ├── DynamicCache.prepare_update_kv = prepare_update_kv
│   └── DynamicCache.update_after_prefill = update_after_prefill
└── 替换函数（覆盖原类方法）
    ├── LlamaForCausalLM_Forward()     # 覆盖 LlamaForCausalLM.forward
    ├── LlamaModel_Forward()           # 覆盖 LlamaModel.forward
    ├── LlamaDecoderLayer_Forward()    # 覆盖 LlamaDecoderLayer.forward
    └── LlamaSdpaAttention_Forward()   # 覆盖 LlamaSdpaAttention.forward (核心)
```

> **注意**：`llama.py` 中的函数是独立定义的，需在外部通过 `model.forward = types.MethodType(LlamaModel_Forward, model)` 等方式绑定到模型实例上。

---

### 2. 核心配置：`extra_config` 字典

整个推理流程通过 `extra_config` 字典在各层之间传递状态，其结构如下：

```python
extra_config = {
    'drop_config':  {...} or None,   # KV 丢弃配置
    'reuse_config': {...} or None,   # KV 复用配置
    'other_config': {
        'decode': bool,              # 是否处于 decode 阶段
        ...
    }
}
```

`extra_config` 作为函数返回值在层间传递（不同于原版只返回 `layer_outputs`）：

```python
layer_outputs, extra_config = decoder_layer(...)
```

---

### 3. `LlamaModel_Forward()` — 核心调度逻辑

#### 3.1 reuse_config 的层级调度

```python
if self.extra_config['reuse_config']:
    cat_kv = self.extra_config['reuse_config']['cat_kv']  # 预缓存的全层 KV
    check_layer = get_layer(self.extra_config['reuse_config']['reuse'])

for i in range(len(self.layers)):
    # 按层切分 cat_kv，传给当前层
    self.extra_config['reuse_config']['cat_kv'] = cat_kv[:, i, :, :, :]
    
    # 标记当前层的检查状态
    if i == check_layer:
        self.extra_config['reuse_config']['check'] = 'checking'
    if i > check_layer:
        self.extra_config['reuse_config']['check'] = 'postchecking'
```

`cat_kv` 是一个 `[batch, num_layers, 2, seq_len, head_dim]` 形状的张量，存储了预计算好的历史 KV。

#### 3.2 三阶段 check 状态机

| `check` 值 | 含义 | 行为 |
|-----------|------|------|
| `None` / `False` | 正常层（`check_layer` 之前） | 正常运行，不启用 reuse |
| `'checking'` | **关键检查层** | 计算 top-k 重要 token，选择 `imp_indices` |
| `'postchecking'` | 检查层之后 | 将重要位置的 key/value 替换为新计算值 |

#### 3.3 position_ids 动态更新

```python
if not self.extra_config['other_config']['decode'] and i == check_layer:
    position_ids = self.extra_config['reuse_config']['imp_indices'].unsqueeze(0)
```

在 `check_layer` 之后，`position_ids` 被替换为重要 token 的实际位置索引，确保后续层的 RoPE 编码正确。

---

### 4. `LlamaSdpaAttention_Forward()` — 注意力核心（含 Drop & Reuse）

这是整个定制化系统最核心的函数（第 334-500 行）。

#### 4.1 Reuse 模块（第 382-422 行）

```
if reuse_config:
    ┌─────────────────────────────────────────────────────┐
    │ 1. 对齐 cat_kv 到当前设备和 dtype                    │
    │ 2. 如果 check 状态激活：                             │
    │    - 使用 fake_q + org_pos 对历史 key 应用 RoPE      │
    │    - 取出 key_old, value_old（历史 KV）             │
    │ 3. checking 阶段：                                   │
    │    - get_topindices() 选出重要 query token           │
    │    - 截断 query_states 为重要 token 子集             │
    │    - 记录 imp_indices                                │
    │ 4. postchecking 阶段：                               │
    │    - 将新计算的 key/value 写入历史 KV 的重要位置      │
    │    - 用更新后的历史 KV 替换 key_states/value_states  │
    └─────────────────────────────────────────────────────┘
```

**关键细节**：

- `fake_q`：用随机张量作为占位 query，配合 `org_pos` 对历史 key 重新应用正确的 RoPE（因为历史 key 是以绝对位置存储的原始 key，需要重新编码到正确位置）。
- `get_topindices()`（来自 `reuse_utils`）：根据 query 与历史 key/value 的相关性，选出最重要的 token 位置。

#### 4.2 Drop 模块（第 425-437 行）

```python
if drop_config:
    save_indices, key_states, value_states, past_key_value = get_saveindices(
        reuse_config, drop_config, other_config,
        self.num_key_value_groups,
        query_states, key_states, value_states,
        past_key_value, cache_kwargs, self.layer_idx
    )
    drop_config['save_indices'] = save_indices
```

`get_saveindices()`（来自 `evict_utils`）负责：
1. 计算每个 token 的重要性分数
2. 选择 top-k 重要 KV 保留到 `past_key_value`
3. 返回选中的 key/value（压缩后）

#### 4.3 注意力计算：使用 FlashInfer

```python
attn_output = flashinfer.single_prefill_with_kv_cache(
    query_states.squeeze(0).transpose(0, 1),
    key_states.squeeze(0).transpose(0, 1),
    value_states.squeeze(0).transpose(0, 1),
    causal=is_causal,
    packed_custom_mask=causal_mask,
)
```

使用 [FlashInfer](https://github.com/flashinfer-ai/flashinfer) 替代标准 SDPA，支持自定义掩码（`packed_custom_mask`），适用于 reuse 场景下的非标准 causal 掩码。

#### 4.4 自定义掩码生成

```python
unpacked_mask, reuse_config['mask'] = create_flashinfer_mask(
    query_states, key_states,
    reuse_config['imp_indices'],
    reuse_config.get('causal', True),
)
```

`create_flashinfer_mask()`（来自 `utils`）将自定义的注意力可见范围转换为 FlashInfer 所需的 packed bit-mask 格式。

---

### 5. `DynamicCache` 的 Monkey Patch

```python
DynamicCache.__init__ = new_init
DynamicCache.prepare_update_kv = prepare_update_kv
DynamicCache.update_after_prefill = update_after_prefill
```

对 HuggingFace `DynamicCache` 进行扩展：
- `new_init`：可能增加额外的缓存字段（如保存 save_indices 等元信息）
- `prepare_update_kv`：在 prefill 之后准备更新 KV 缓存的状态
- `update_after_prefill`：在 prefill 完成后执行 KV 压缩或选择性写入

---

## 两文件的协作关系

```
┌───────────────────────────────────────────────────────────────┐
│                         ragkv 推理流程                         │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  [预计算阶段] precompute.py                                    │
│      │                                                        │
│      ▼                                                        │
│  LlamaForCausalLM_Precompute (llama_precompute.py)            │
│      │  使用 LlamaSdpaAttention 的 hack_kv 收集原始 KV        │
│      │  reuse_config 透传，收集各层重要 KV                     │
│      ▼                                                        │
│  保存 cat_kv (shape: [batch, layers, 2, seq, head_dim])       │
│                                                               │
│  [生成阶段] eval_*.py                                          │
│      │                                                        │
│      ▼                                                        │
│  LlamaModel_Forward (llama.py) ─── Monkey Patched             │
│      │  从存储加载 cat_kv                                      │
│      │  按层传入 reuse_config['cat_kv']                       │
│      ▼                                                        │
│  LlamaSdpaAttention_Forward (llama.py)                        │
│      ├── [Reuse 模块] 检查层选 top-k token，后续层复用历史 KV  │
│      ├── [Drop 模块]  prefill 时压缩 KV，只保留重要 token     │
│      └── [FlashInfer] 支持自定义稀疏掩码的注意力计算           │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## 关键设计模式总结

### 1. Monkey Patch 式扩展
`llama.py` 不修改原始类，而是用独立函数替换方法，保持与 HuggingFace 生态的兼容性，同时可以按需切换原版/定制版。

### 2. `extra_config` 字典作为侧信道
通过在 `forward()` 返回值中携带 `extra_config`，实现了跨层状态传递，无需修改模型架构即可在任意层插入业务逻辑。

### 3. 状态机驱动的 Reuse 流程
`check` 字段的三个状态（`None` → `'checking'` → `'postchecking'`）构成一个简单的有限状态机，清晰地将复用流程划分为"检测""主动"三个阶段。

### 4. `hack_kv`：侵入式 KV 获取
在标准前向中无处获取 RoPE 前的原始 KV，`hack_kv` 通过在 `LlamaSdpaAttention.forward()` 中的中间状态保存，为预计算阶段提供了必要的原始数据。

---

## 潜在问题与注意点

| 问题 | 位置 | 说明 |
|------|------|------|
| `attn_output.unsqueeze(0)` 未赋值 | `llama.py` L484 | `attn_output = flashinfer.single_prefill_with_kv_cache(...)` 后紧接 `attn_output.unsqueeze(0)` 但返回值未被接收，是否存在 bug 待确认 |
| `LlamaSdpaAttention` 中注释掉的 `torch.nn.functional.scaled_dot_product_attention` | `llama.py` L486-493 | 原 SDPA 被 FlashInfer 替代，但维护了注释版本供回退参考 |
| `create_mask` 未在 `llama.py` 中使用 | `llama_precompute.py` L59 | 原来用于 SDPA 掩码的 `create_mask` 已被 `create_flashinfer_mask` 替代，对应代码已注释（L456-462） |
| `reuse_config['mask']` 跨 token 复用 | `llama.py` L466 | 掩码在 checking 阶段首次创建后被缓存复用，若 `imp_indices` 每次都变化则需注意缓存失效 |
