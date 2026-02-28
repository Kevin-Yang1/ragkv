"""
LongBench 数据集加载器
用于加载和处理 LongBench 长上下文基准测试数据集
"""

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import json
import torch
from transformers import AutoTokenizer

# =============================================================================
# 不同模型的特殊 Token 配置
# =============================================================================
# 这些是各个模型的对话格式模板 token IDs
CONFIG = {
    # Mistral 模型的对话格式
    # s_start: <s>[INST] (开始用户输入)
    # s_end: [/INST] (结束用户输入，开始助手回复)
    'mistral': {
        's_start': [1, 733, 16289, 28793],      # <s>[INST]
        's_end': [733, 28748, 16289, 28793]     # [/INST]
    },
    
    # Llama-3 模型的对话格式
    # s_start: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
    # s_end: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
    'llama-3': {
        's_start': [128000, 128006, 882, 128007, 271],           # user 消息开始
        's_end': [128009, 128006, 78191, 128007, 271]            # assistant 消息开始
    },
    
    # Qwen 模型的对话格式
    # s_start: <|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n
    # s_end: <|im_end|>\n<|im_start|>assistant\n
    'qwen': {
        's_start': [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198],
        's_end': [151645, 198, 151644, 77091, 198]
    },
}


class LongBench(Dataset):
    """
    LongBench 数据集类
    
    功能：
    1. 加载分块后的 LongBench 数据（来自 chunk_longbench.py）
    2. 将文本转换为 token IDs
    3. 组织成适合模型处理的格式
    4. 添加对话格式模板（根据模型类型）
    
    数据结构：
    - 每个样本包含多个 chunks（文档片段）
    - 每个样本有一个问题和答案
    - 数据会被转换为 token IDs 并添加特殊格式
    
    使用方式：
        dataset = LongBench(args)
        sample = dataset[0]
        # sample 包含: doc_ids, prompt_ids, answer, params, all_classes
    """
    
    def __init__(self, args):
        """
        初始化 LongBench 数据集
        
        Args:
            args: 命令行参数对象，包含：
                - args.model: 模型路径 (如 '/data/ykw/model/Meta-Llama-3-8B-Instruct')
                - args.dataset: 数据集名称 (如 '2wikimqa', 'qasper', 'hotpotqa')
        
        加载内容：
        1. 分块后的数据文件 (inputs/{model_name}/{dataset}.json)
        2. 前缀提示模板 (config/longbench/prefix_prompt.json)
        3. 模型对应的 tokenizer
        4. 对话格式配置（根据模型类型）
        """
        # -------------------------------------------------------------------------
        # 1. 加载数据文件
        # -------------------------------------------------------------------------
        # 路径格式: ./inputs/Meta-Llama-3-8B-Instruct/2wikimqa.json
        self.dataset_path = f'./inputs/{os.path.basename(args.model)}/{args.dataset}.json'
        
        # ori_data 结构：
        # [
        #   {
        #     'chunks': ['chunk1 text', 'chunk2 text', ...],  # 分块后的文档
        #     'question': 'formatted question text',           # 格式化的问题
        #     'answers': ['answer1', 'answer2'],              # 标准答案列表
        #     'all_classes': ['class1', 'class2']             # 分类任务的类别
        #   },
        #   ...
        # ]
        self.ori_data = json.load(open(self.dataset_path, 'r'))
        
        # -------------------------------------------------------------------------
        # 2. 加载前缀提示模板
        # -------------------------------------------------------------------------
        # prefix_prompt 例如：
        # {
        #   "2wikimqa": "Answer the question based on the given passages...",
        #   "qasper": "Answer the following question based on the paper...",
        #   ...
        # }
        self.prefix_prompt = json.load(open('config/longbench/prefix_prompt.json', 'r'))
        
        # -------------------------------------------------------------------------
        # 3. 检测模型类型并加载对应配置
        # -------------------------------------------------------------------------
        # 根据模型路径判断模型类型
        if 'mistral' in args.model.lower():
            self.config = CONFIG['mistral']
        elif 'llama-3' in args.model.lower():
            self.config = CONFIG['llama-3']
        elif 'qwen' in args.model.lower():
            self.config = CONFIG['qwen']
        
        # -------------------------------------------------------------------------
        # 4. 加载 tokenizer
        # -------------------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        # -------------------------------------------------------------------------
        # 5. 构建数据集
        # -------------------------------------------------------------------------
        # Qwen 模型有特殊处理（没有 BOS token）
        if 'qwen' in args.model.lower():
            self._construct_for_qwen(args)
        else:
            self._construct(args)
    
    def __len__(self):
        """返回数据集中样本的数量"""
        return len(self.prompt_ids)
    
    def __getitem__(self, idx):
        """
        获取第 idx 个样本
        
        Returns:
            dict: 包含以下键值的字典
                - doc_ids: List[List[int]] - 分块的 token IDs
                  格式: [[prefix_tokens], [chunk1_tokens], [chunk2_tokens], ..., [question_tokens]]
                  
                - prompt_ids: List[int] - 完整拼接后的 token IDs
                  格式: prefix + chunk1 + chunk2 + ... + question
                  
                - answer: List[str] - 标准答案列表
                  例如: ['Barack Obama', 'Obama']
                  
                - params: dict - 重要的参数信息
                  {
                    'doc_start_len': int,   # 每个 doc chunk 开始的 token 数（通常是 BOS）
                    'prefix_len': int,      # prefix prompt 的长度
                    'last_len': int,        # question 部分的长度
                    'sink_pos': List[int]   # Sink token 位置（用于某些策略）
                  }
                  
                - all_classes: List[str] - 分类任务的所有类别
                  例如: ['ABBR', 'ENTY', 'DESC', ...]
        
        示例：
            sample = dataset[0]
            doc_ids = sample['doc_ids']       # [[128000, 128006, ...], [1234, 5678, ...], ...]
            prompt_ids = sample['prompt_ids'] # [128000, 128006, ..., 1234, 5678, ...]
            answer = sample['answer']         # ['Barack Obama']
            params = sample['params']         # {'prefix_len': 50, 'last_len': 20, ...}
        """
        return {
            'doc_ids': self.doc_ids[idx],
            'prompt_ids': self.prompt_ids[idx],
            'answer': self.answers[idx],
            'params': self.params[idx],
            'all_classes': self.all_classes[idx]
        }
    
    def _construct(self, args):
        """
        构建数据集（Llama、Mistral 等标准模型）
        
        处理流程：
        1. 为每个样本添加 prefix prompt（任务指令）
        2. 将每个 chunk 转换为 token IDs
        3. 添加对话格式模板（s_start, s_end）
        4. 拼接所有部分：prefix + chunks + question
        5. 记录关键参数（prefix_len, last_len 等）
        
        最终格式：
        <bos> [对话开始] prefix_prompt [chunks] question [对话结束]
        """
        # 初始化存储列表
        self.doc_ids = []         # 分块的 token IDs
        self.prompt_ids = []      # 完整的 token IDs
        self.answers = []         # 答案
        self.params = []          # 参数
        self.all_classes = []     # 类别
        
        # -------------------------------------------------------------------------
        # 构建 prefix prompt（任务指令部分）
        # -------------------------------------------------------------------------
        # 某些数据集不需要对话格式（直接作为完形填空任务）
        if args.dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            # 格式: <bos> + prefix_prompt
            prefix_prompt = [self.tokenizer.bos_token_id] + self.tokenizer.encode(self.prefix_prompt[args.dataset])[1:]
        else:
            # 格式: <bos> + [INST] + prefix_prompt
            prefix_prompt = [self.tokenizer.bos_token_id] + self.config['s_start'] + self.tokenizer.encode(self.prefix_prompt[args.dataset])[1:]
        
        # doc_start_len: 每个 chunk 开头的 <bos> token 数量
        doc_start_len = 1
        
        # -------------------------------------------------------------------------
        # 处理每个样本
        # -------------------------------------------------------------------------
        for item in tqdm(self.ori_data):
            # 提取数据
            doc_prompts = item['chunks']      # ['chunk1 text', 'chunk2 text', ...]
            q_prompt = item['question']        # 'Question: ...'
            
            # 将每个 chunk 转换为 token IDs
            doc_chunk_ids = [self.tokenizer.encode(doc) for doc in doc_prompts]
            
            # 处理问题部分
            if args.dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                # 不需要对话格式
                q_ids = self.tokenizer.encode(q_prompt)[1:]
            else:
                # 添加对话结束标记
                # 格式: question + [/INST] 或 <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                q_ids = self.tokenizer.encode(q_prompt)[1:] + self.config['s_end']
            
            # -------------------------------------------------------------------------
            # 组装完整的 doc_chunk_ids 列表
            # -------------------------------------------------------------------------
            # 格式: [prefix, chunk1, chunk2, ..., question]
            doc_chunk_ids = [chunk_ids for chunk_ids in doc_chunk_ids]
            doc_chunk_ids = [prefix_prompt] + doc_chunk_ids
            doc_chunk_ids = doc_chunk_ids + [q_ids]
            
            # 示例：
            # doc_chunk_ids[0] = [<bos>, <inst_start>, prefix_tokens, ...]  (prefix)
            # doc_chunk_ids[1] = [<bos>, chunk1_tokens, ...]                (chunk 1)
            # doc_chunk_ids[2] = [<bos>, chunk2_tokens, ...]                (chunk 2)
            # doc_chunk_ids[-1] = [question_tokens, <inst_end>, ...]        (question)
            
            # 记录关键长度
            prefix_len = len(doc_chunk_ids[0])  # prefix prompt 的 token 数
            last_len = len(q_ids)               # question 的 token 数
            
            # -------------------------------------------------------------------------
            # 拼接所有 chunks 成完整的 prompt
            # -------------------------------------------------------------------------
            prompt_ids = []
            for i in range(len(doc_chunk_ids)):
                if i == 0 or i == len(doc_chunk_ids) - 1:
                    # prefix 和 question：保留所有 tokens
                    prompt_ids += doc_chunk_ids[i]
                else:
                    # 其他 chunks：去掉开头的 <bos> token（避免重复）
                    prompt_ids += doc_chunk_ids[i][doc_start_len:]
            
            # 最终 prompt_ids 格式：
            # [<bos>, <inst_start>, prefix, chunk1[1:], chunk2[1:], ..., question, <inst_end>]
            
            # -------------------------------------------------------------------------
            # Sink token 位置（用于 attnlink 策略）
            # -------------------------------------------------------------------------
            # 这里初始化为全 0，表示所有位置都可能是 sink
            # 实际使用时会在特定策略中更新
            sink_indices = [0] * len(prompt_ids)
            
            # -------------------------------------------------------------------------
            # 保存到列表
            # -------------------------------------------------------------------------
            self.doc_ids.append(doc_chunk_ids)
            self.prompt_ids.append(prompt_ids)
            self.params.append({
                'doc_start_len': doc_start_len,  # 1 (每个 chunk 开头的 BOS)
                'prefix_len': prefix_len,        # prefix prompt 长度
                'last_len': last_len,            # question 长度
                'sink_pos': sink_indices         # sink token 位置
            })
            self.answers.append(item['answers'])
            self.all_classes.append(item['all_classes'])
    
    def _construct_for_qwen(self, args):
        """
        为 Qwen 模型构建数据集（特殊处理）
        
        Qwen 的特殊之处：
        1. 没有 BOS token
        2. doc_start_len = 0（不需要跳过 BOS）
        3. 对话格式不同
        
        其他逻辑与 _construct 相同
        """
        # 初始化存储列表
        self.doc_ids = []
        self.prompt_ids = []
        self.answers = []
        self.params = []
        self.all_classes = []
        
        # 构建 prefix prompt（无 BOS token）
        if args.dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prefix_prompt = self.tokenizer.encode(self.prefix_prompt[args.dataset])
        else:
            # Qwen 格式: <|im_start|>system\n{prefix}<|im_end|>\n<|im_start|>user\n
            prefix_prompt = self.config['s_start'] + self.tokenizer.encode(self.prefix_prompt[args.dataset])
        
        # Qwen 没有 BOS，所以不需要跳过
        doc_start_len = 0
        
        # 处理每个样本（逻辑与 _construct 类似）
        for item in tqdm(self.ori_data):
            doc_prompts = item['chunks']
            q_prompt = item['question']
            doc_chunk_ids = [self.tokenizer.encode(doc) for doc in doc_prompts]
            
            if args.dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                q_ids = self.tokenizer.encode(q_prompt)
            else:
                # Qwen 格式: question + <|im_end|>\n<|im_start|>assistant\n
                q_ids = self.tokenizer.encode(q_prompt) + self.config['s_end']
            
            doc_chunk_ids = [chunk_ids for chunk_ids in doc_chunk_ids]
            doc_chunk_ids = [prefix_prompt] + doc_chunk_ids
            doc_chunk_ids = doc_chunk_ids + [q_ids]
            
            prefix_len = len(doc_chunk_ids[0])
            last_len = len(q_ids)
            
            prompt_ids = []
            for i in range(len(doc_chunk_ids)):
                if i == 0 or i == len(doc_chunk_ids) - 1:
                    prompt_ids += doc_chunk_ids[i]
                else:
                    # doc_start_len=0，所以不跳过任何 token
                    prompt_ids += doc_chunk_ids[i][doc_start_len:]
            
            sink_indices = [0] * len(prompt_ids)
            
            self.doc_ids.append(doc_chunk_ids)
            self.prompt_ids.append(prompt_ids)
            self.params.append({
                'doc_start_len': doc_start_len,
                'prefix_len': prefix_len,
                'last_len': last_len,
                'sink_pos': sink_indices
            })
            self.answers.append(item['answers'])
            self.all_classes.append(item['all_classes'])


# =============================================================================
# 使用示例和数据结构说明
# =============================================================================
"""
使用示例：
    dataset = LongBench(args)
    print(f"数据集大小: {len(dataset)}")
    
    sample = dataset[0]
    print(f"doc_ids 结构: {len(sample['doc_ids'])} chunks")
    print(f"prompt_ids 长度: {len(sample['prompt_ids'])} tokens")
    print(f"答案: {sample['answer']}")
    print(f"参数: {sample['params']}")

数据流转：
    原始数据 (chunk_longbench.py)
        ↓
    {'chunks': [...], 'question': '...', 'answers': [...]}
        ↓
    LongBench.__init__() 
        ↓
    {'doc_ids': [[...], [...]], 'prompt_ids': [...], 'answer': [...], 'params': {...}}
        ↓
    预计算 KV Cache (precompute.py)

关键字段说明：
    - doc_ids: 保留了分块边界信息，用于预计算时逐块处理
    - prompt_ids: 完整序列，用于生成时的输入
    - params['prefix_len']: A³ 算法需要知道哪些是 prefix（不参与选择）
    - params['last_len']: A³ 算法需要知道哪些是 question（必须保留）
"""