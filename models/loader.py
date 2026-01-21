from transformers import AutoTokenizer, LlamaConfig, MistralConfig, Qwen2Config
import  transformers
import torch

def load_model(args):
    if 'llama' in args.model.lower():
        config = LlamaConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        if args.reuse == 'no' and args.drop == 'False':
            pass
        else:
            from models.monkeypatch import replace_llama
            replace_llama()

        model = transformers.LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model,
            config=config,
            cache_dir='./cache',
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )

    elif 'mistral' in args.model.lower():
        config = MistralConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        if args.reuse == 'no' and args.drop == 'False':
            pass
        else:
            from models.monkeypatch import replace_mistral
            replace_mistral()

        model = transformers.MistralForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model,
            config=config,
            cache_dir='./cache',
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            use_safetensors=False,
            trust_remote_code=True,
        )       

    elif 'qwen' in args.model.lower():
        config = Qwen2Config.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        if args.reuse == 'no' and args.drop == 'False':
            pass
        else:
            from models.monkeypatch import replace_qwen
            replace_qwen()

        model = transformers.Qwen2ForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model,
            config=config,
            cache_dir='./cache',
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )    

    else:
        raise NotImplementedError
    
    model.eval()
    return model, tokenizer

def load_model_precompute(args):
    if 'llama' in args.model.lower():
        config = LlamaConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        from models.llama.llama_precompute import LlamaForCausalLM_Precompute
        transformers.LlamaForCausalLM = LlamaForCausalLM_Precompute

        model = transformers.LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model,
            config=config,
            cache_dir='./cache',
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )

    elif 'mistral' in args.model.lower():
        config = MistralConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        from models.mistral.mistral_precompute import MistralForCausalLM_Precompute
        transformers.MistralForCausalLM = MistralForCausalLM_Precompute

        model = transformers.MistralForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model,
            config=config,
            cache_dir='./cache',
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            use_safetensors=False,
            trust_remote_code=True,
        )       

    elif 'qwen' in args.model.lower():
        config = Qwen2Config.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        from models.qwen.qwen_precompute import Qwen2ForCausalLM_Precompute
        transformers.Qwen2ForCausalLM= Qwen2ForCausalLM_Precompute

        model = transformers.Qwen2ForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model,
            config=config,
            cache_dir='./cache',
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )     

    else:
        raise NotImplementedError
    
    model.eval()
    return model, tokenizer