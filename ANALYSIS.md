# 项目分析文档（ragkv）

## 1. 项目定位
本项目是论文 **A^3: Attention-Aware Accurate KV Cache Fusion** 的工程实现，目标是在长上下文推理中，通过预计算与选择性重计算融合 KV Cache，在尽量保持准确率的前提下降低 TTFT。

从仓库代码看，核心能力集中在三件事：
- 长文档切块（chunk）
- 离线 KV 预计算（precompute）
- 在线复用/融合 + 评测（eval）

支持三类模型族：`Llama`、`Mistral`、`Qwen`。

## 2. 当前仓库快照（基于当前工作区）
- 输入数据目录：`inputs/`，当前主要有 `inputs/Meta-Llama-3-8B-Instruct/2wikimqa.json`
- KV 目录：`kvs/`，当前约 **108GB**，`kvs/Llama-3-8B-Instruct/2wikimqa` 下有 **164** 个 `kvs.pt`
- 一个样例 KV 张量：`(2, 32, 8, 4202, 128)`，`bfloat16`
- `data/longbench` 是软链接，指向工作区外部目录：`/data/ykw/datasets/LLM_dataset/LongBench/data`
- 工作树已有本地变更（非本次新增），项目处于“实验中”状态而非干净基线

## 3. 目录结构与职责
- `chunk_*.py`
  - `chunk_longbench.py`：LongBench 切块
  - `chunk_needle.py`：Needle-in-a-Haystack 数据构建
  - `chunk_ruler.py`：RULER 数据切块
- `precompute.py`
  - 逐样本预计算并落盘 KV
- `eval_*.py`
  - `eval_longbench.py` / `eval_needle.py` / `eval_ruler.py`
- `models/`
  - `loader.py`：模型装载与 monkeypatch 入口
  - `monkeypatch.py`：替换 HF forward
  - `llama|mistral|qwen/*.py`：在线复用逻辑
  - `*_precompute.py`：预计算模型逻辑（抓取每层 `hack_kv`）
  - `reuse_utils.py`：重要 token 选择
  - `evict_utils.py`：KV 驱逐（Streaming / SnapKV）
- `data/*/loader.py`
  - 把 chunk 后数据组装为 `doc_ids/prompt_ids/params`
- `config/`
  - `longbench/`、`ruler/` 提示词模板
  - `drop/snap*.json` 驱逐超参
- `scripts/*.sh`
  - 预计算与评测批处理脚本

## 4. 端到端流程
### 4.1 数据切块
- LongBench 原始 jsonl -> `inputs/{model_basename}/{dataset}.json`
- 每条样本一般为：
  - `chunks`: 文档分块文本列表
  - `question`
  - `answers`
  - `all_classes`

### 4.2 KV 预计算
`precompute.py` 按数据集加载对应 loader：
- LongBench -> `data.longbench.loader.LongBench`
- Needle -> `data.PaulGrahamEssays.loader.Needle`
- RULER -> `data.RULER.loader.Ruler`

随后 `load_model_precompute()` 装载定制 `*_precompute` 模型，逐 chunk 前向，并在每层 attention 中读取 `hack_kv`，拼接后存盘到：
- `kvs/{model}/{dataset}/item_{id}/kvs.pt`

### 4.3 在线推理与评测
`eval_*.py` 在推理阶段分两条路径：
- `reuse=no` 且 `drop=False`：全量重算（`vanilla`）
- 否则：
  - 从 `kvs.pt` 加载预计算 KV
  - 根据 `reuse` 选择重要 token 重算
  - 与旧 KV 融合
  - 可叠加 `drop` 策略控制缓存容量

最终输出 `result.json` 和指标汇总（如 LongBench 的 `result.txt`）。

## 5. A^3 核心实现解读
### 5.1 模型改写方式
通过 `models/monkeypatch.py` 替换 HF 模型的：
- `ForCausalLM.forward`
- `Model.forward`
- `DecoderLayer.forward`
- `SdpaAttention.forward`

重点发生在 attention forward：
- 读取预计算 KV（`cat_kv`）
- 在特定层做“重要位置”选择（checking）
- 后续层按索引把重算结果写回旧 KV（postchecking）

### 5.2 token 选择策略（`reuse_utils.py`）
`reuse` 支持：
- `blend`：按新旧 value 差异选 top-k
- `debug`：按问题对文档注意力分数选 top-k（当前主推）
- `attnlink`：使用 `sink_pos`
- `full`：保留最后 query 区域
- `cat`：极限压缩，仅保留最后 1 个 token

`get_layer()` 规定在第 0 或 1 层执行关键筛选。

#### 5.2.1 调用时机与上下文
- 选择发生在 attention 的 `checking` 阶段（`models/*/*.py`）：`top_indices = get_topindices(...)`。
- `checking` 完成后，当前层只对 `top_indices` 对应位置做重算；后续层 `postchecking` 把重算结果写回旧 KV。
- 分层触发点由 `get_layer(reuse)` 控制：
  - `blend` / `debug` 在第 1 层做选择
  - 其他策略在第 0 层做选择
- 预算由 `recomp_ratio` 控制（`--rate`），核心公式：
  - `topk_num = int((total_len - last_len) * recomp_ratio)`
  - 其中 `last_len` 是 query 段长度（`params['last_len']`）

#### 5.2.2 各策略细化
1. `blend`
- 候选区间：除 `last_len` 之外的历史 token。
- 打分：新算 `value_states` 与旧缓存 `value_old` 的 L2 差：
  - `score[t] = sum_{head,dim}(V_new - V_old)^2`
- 选择：取 `topk_num` 个最高差异位置，再强制拼接全部 query 位置 `last_indices`。
- 直觉：优先重算“变化最大”的位置，降低旧 KV 误差传播。

2. `debug`
- 候选区间：`prefix` 之后到 query 之前（代码里通过 `prefix_len` 跳过前缀）。
- 打分：用 query 段（最后 `last_len` tokens）对文档 keys 的注意力，跨 head 求和后再做 `avg_pool1d(kernel=5)` 平滑。
- 选择：对平滑后注意力取 top-k，再拼接全部 query 位置。
- 直觉：把重算预算分给“对问题最相关”的上下文 token，这是 A^3 的主思路。

3. `attnlink`
- 直接使用 `params['sink_pos']` 作为 `top_indices`。
- 依赖外部提供的 sink 位置，`reuse_utils.py` 不做计算。

4. `full`
- 仅保留 query 段全部位置（`[total_len-last_len, ..., total_len-1]`）。
- 本质是“只重算问题段，不重算文档段”的基线。

5. `cat`
- 仅保留最后 1 个 token（`total_len-1`）。
- 是极限压缩设定，通常用于速度/退化边界测试。

#### 5.2.3 与注意力 mask 的关系
- `get_topindices` 输出的 `imp_indices` 会驱动：
  - query 子序列抽取（只算被选位置）
  - 自定义 attention mask 构造（`create_flashinfer_mask`）
  - 后续层重算结果回填到旧 KV（按 `imp_indices` 写回）
- 因此 `imp_indices` 的质量直接决定“速度-精度”权衡上限。

#### 5.2.4 复杂度与预算
- `blend` 主要成本：一次 `value` 差分与归约，复杂度近似 `O(H * T * D)`。
- `debug` 主要成本：query-vs-key 注意力打分，复杂度近似 `O(last_len * T * (H*D))`，通常比 `blend` 更重，但更语义相关。
- `full/cat/attnlink` 成本最低（几乎无额外打分）。
- 真实收益取决于：
  - `recomp_ratio`（默认 0.15）
  - `last_len` 占比
  - 上下文结构（信息集中还是分散）

#### 5.2.5 当前实现中的关键风险（基于现代码）
1. `attnlink` 在当前 loader 中不可直接用  
- `sink_pos` 默认是全 0 列表，若不预处理会导致选择退化（重复索引 0）。

2. `create_flashinfer_mask` 的 `mode` 判定实现与注释不一致  
- 注释写的是 `'causal'/'rightbottom'` 两种模式，但实现使用 `if mode:` 布尔分支，字符串都会进入同一分支。

3. `causal` 字段来源不明确  
- `reuse_config` 初始化未设置 `causal`，但 Llama 分支会读取 `reuse_config['causal']`，有运行时风险。

4. Mistral/Qwen 调用签名与函数定义不一致  
- 两者调用 `create_flashinfer_mask(...)` 少传一个参数（函数定义需要 `mode`），存在潜在 `TypeError` 风险。

5. `topk_num` 边界未保护  
- `int(...)` 可能得到 0；当候选长度或比例极端时，`topk` 可能触发异常或退化行为，建议显式 `clamp`。

#### 5.2.6 已落地：`surprisal_chunk` 策略
- 离线阶段（`precompute.py`）通过 `--save_surprisal` 可选开启 surprisal 提取：
  - 默认仅保存 `kvs.pt`；
  - 开启后在 `save_kvs(...)` 的同一次前向中同步提取 surprisal（不再额外二次前向）；
  - 结果保存到 `kvs/{model}/{dataset}/item_{id}/surprisal.pt`。
- surprisal 计算公式已改为低显存路径：
  - `token_nll = cross_entropy(shift_logits, shift_targets, reduction='none')`；
  - chunk 首 token 分数为 0，其余位置为 `-log p(x_t|x_<t)`；
  - 中间 doc chunk 仍按 `doc_start_len` 对齐后拼接。
- 在线阶段（`eval_longbench.py`）在 `reuse=surprisal_chunk` 时加载并校验：
  - `scores` 长度必须等于 `prompt_ids` 长度；
  - `chunk_ranges` 必须与当前样本重建的区间一致；
  - 缺失或不一致直接报错终止（fail-fast）。
- 选点逻辑（`reuse_utils.py`）：
  - 只在文档 chunks 范围内选点；
  - 总预算 `K=int(doc_total_len * rate)`；
  - 按 chunk 长度比例分配 `k_i`（`floor + 最大余数`）；
  - 每个 chunk 内部按 surprisal top-`k_i` 选 token；
  - question 区全部强制保留。

### 5.3 掩码与注意力计算
- 使用 `flashinfer.packbits` 打包自定义掩码
- 使用 `flashinfer.single_prefill_with_kv_cache` 执行 prefill 注意力
- 对 RoPE 对齐做了旧 KV 旋转处理，确保 query/key 位置一致

### 5.4 KV 驱逐策略（`evict_utils.py`）
- `Streaming`：保留 prefix + recent 窗口
- `SnapKV`：基于注意力池化选择历史 token（支持 block）

并通过 monkeypatch 过的 `DynamicCache` 在 first-token 后更新缓存。

## 6. 工程实现特点
- 优点
  - 主流程闭环完整（chunk -> precompute -> eval）
  - 三模型族实现结构统一，便于横向对比
  - 复用策略与驱逐策略可组合实验
- 局限
  - 强依赖本地路径、GPU 编号、软链接数据目录
  - 脚本化实验较重，可移植性与自动化较弱
  - 缺少回归测试与 CI

## 7. 已确认问题与风险
### P0（建议优先修复）
1. `visualize.py` 存在语法错误，无法运行（`reuse_method = 'debug':`）。
2. `scripts/eval_longbench.sh` 的 python 命令续行写法错误（反斜杠后接空格+注释），实际执行会把下一行当成新命令。
3. `models/evict_utils.py` 中 `update_after_prefill()` 固定循环 `range(28)`，对 32 层模型（如 Llama-3-8B/Mistral-7B）有错配风险。

### P1（中高优先级）
1. 路径和设备高度硬编码（模型路径、`CUDA_VISIBLE_DEVICES`、`cuda:0`）。
2. `data/longbench` 依赖工作区外软链接，迁移环境容易失效。
3. `readme.md` 的命令存在不一致（如 `python .scripts/copy_results_ruler.py` 路径错误）。
4. `chunk_longbench.py` 当前只启用了 `2wikimqa`，默认不是全量 LongBench。

### P2（工程质量）
1. 缺少自动化测试（仓库中无 test 文件）。
2. 依赖列表非常重，且混入大量推理服务相关包，复现门槛高。
3. 部分函数有可维护性风险（全局变量耦合、调试残留、日志与注释风格不统一）。

## 8. 复现实验建议（最小可用路径）
1. 先固定一个模型和一个数据集（例如 Llama-3-8B + 2wikimqa）。
2. 顺序执行：
   - `python chunk_longbench.py`
   - `bash scripts/precompute.sh`
   - `python eval_longbench.py ...`（暂不通过 `eval_longbench.sh`，先规避脚本续行问题）
3. 在 `reuse=no`、`reuse=debug` 两组下比较：
   - 任务指标（F1/ROUGE 等）
   - `TTFT`、`TPOT`
4. 再叠加 `drop=Streaming/SnapKV` 做容量-效果权衡。

## 9. 建议的改造路线
- 阶段 1（稳定运行）
  - 修复语法错误与 shell 续行问题
  - 把路径/GPU 改为 CLI 参数或环境变量
  - `update_after_prefill` 按 `len(self.kv_pool['key_pool'])` 动态层数
- 阶段 2（可复现）
  - 引入最小 smoke tests（数据 loader、单条 precompute、单条 decode）
  - 增加 `Makefile`/统一入口脚本
  - 增加环境检查脚本（模型路径、数据路径、flashinfer 可用性）
- 阶段 3（可维护）
  - 把 LongBench/Needle/RULER 公共逻辑抽象
  - 统一配置体系（YAML/JSON）替代多处硬编码
  - 明确实验产物规范（目录命名、元信息、版本标识）

## 10. 总结
这是一个“研究原型向工程实现过渡”的项目：算法主线完整、实验功能可跑，但脚本健壮性与可移植性仍是主要短板。若目标是论文复现实验，当前代码已具备基础条件；若目标是团队协作或长期维护，建议先完成 P0/P1 修复再继续扩展。
