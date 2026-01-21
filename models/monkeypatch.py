import transformers
from models.llama.llama import *
from models.mistral.mistral import *
from models.qwen.qwen import *

def replace_llama():
    print('relacing llama ...')
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = LlamaForCausalLM_Forward
    transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = LlamaDecoderLayer_Forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = LlamaModel_Forward
    transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = LlamaSdpaAttention_Forward

def replace_mistral():
    print('relacing mistral ...')
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.forward = MistralForCausalLM_Forward
    transformers.models.mistral.modeling_mistral.MistralDecoderLayer.forward = MistralDecoderLayer_Forward
    transformers.models.mistral.modeling_mistral.MistralModel.forward = MistralModel_Forward
    transformers.models.mistral.modeling_mistral.MistralSdpaAttention.forward = MistralSdpaAttention_Forward

def replace_qwen():
    print('relacing qwen ...')
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.forward = Qwen2ForCausalLM_Forward
    transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer.forward = Qwen2DecoderLayer_Forward
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = Qwen2Model_Forward
    transformers.models.qwen2.modeling_qwen2.Qwen2SdpaAttention.forward = Qwen2SdpaAttention_Forward