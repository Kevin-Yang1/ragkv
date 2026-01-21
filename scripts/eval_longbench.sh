export HF_ENDPOINT=https://hf-mirror.com
export TORCH_CUDA_ARCH_LIST="8.6"
declare -A path_map
path_map['Mistral-7B-Instruct']='../Models/LLMs/Mistral-7B-Instruct-v0.2'
path_map['Llama-3-8B-Instruct']='../Models/LLMs/llama3/Meta-Llama-3-8B-Instruct'
path_map['Qwen2.5-7B-Instruct']='../Models/LLMs/Qwen2.5-7B-Instruct'

# -------------Edit here-------------
export CUDA_VISIBLE_DEVICES=5

# model='Mistral-7B-Instruct'
model='Llama-3-8B-Instruct'
# model='Qwen2.5-7B-Instruct'

declare -a reuse_list=(
    'debug'
)

# drop='Streaming'
# drop='SnapKV'
# drop_config='./config/drop/snap1.json'

drop='False'
# -------------Edit here-------------

# eval in longbench
declare -a dataset_list=(
    "qasper"
    "multifieldqa_en"
    "hotpotqa"
    "2wikimqa"
    "gov_report"
    "multi_news"
    "trec"
    "triviaqa"
    "samsum"
    "passage_count"
    "lcc"
)

for reuse in "${reuse_list[@]}"; do
    for dataset in "${dataset_list[@]}"; do

        if [ ${drop} != "False" ]; then
            output_dir=./outputs/${model}/${reuse}-${drop}/${dataset}
        else
            drop_config=None
            output_dir=./outputs/${model}/${reuse}/${dataset}
        fi
        mkdir -p ${output_dir}

        python ./eval_longbench.py \
            --model ${path_map[$model]} \
            --reuse ${reuse} \
            --output_path ${output_dir} \
            --dataset ${dataset} \
            --kv_path ./kvs/${model}/${dataset} \
            --drop ${drop} \
            --drop_config ${drop_config}\

        wait
    done
done