export HF_ENDPOINT=https://hf-mirror.com
export TORCH_CUDA_ARCH_LIST="8.6"
declare -A path_map
path_map['Mistral-7B-Instruct']='../Models/LLMs/Mistral-7B-Instruct-v0.2'
path_map['Llama-3-8B-Instruct']='../Models/LLMs/llama3/Meta-Llama-3-8B-Instruct'
path_map['Qwen2.5-7B-Instruct']='../Models/LLMs/Qwen2.5-7B-Instruct'

# -------------Edit here-------------
export CUDA_VISIBLE_DEVICES=6

model='Mistral-7B-Instruct'
# model='Llama-3-8B-Instruct'
# model='Qwen2.5-7B-Instruct'

declare -a reuse_list=(
    'debug'
)

drop='False'
# -------------Edit here-------------

# eval in needle
for reuse in "${reuse_list[@]}"; do
    if [ ${drop} != "False" ]; then
        output_dir=./outputs/${model}/${reuse}-${drop}/needle
    else
        drop_config=None
        output_dir=./outputs/${model}/${reuse}/needle
    fi
    mkdir -p ${output_dir}

    python ./eval_needle.py \
        --model ${path_map[$model]} \
        --reuse ${reuse} \
        --output_path ${output_dir} \
        --dataset needle \
        --kv_path ./kvs/${model}/needle \
        --drop ${drop} \
        --drop_config ${drop_config}\

    wait
done