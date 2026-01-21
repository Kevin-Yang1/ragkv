export HF_ENDPOINT=https://hf-mirror.com
export TORCH_CUDA_ARCH_LIST="8.6"
declare -A path_map
path_map['Mistral-7B-Instruct']='../Models/LLMs/Mistral-7B-Instruct-v0.2'
path_map['Llama-3-8B-Instruct']='../Models/LLMs/llama3/Meta-Llama-3-8B-Instruct'
path_map['Qwen2.5-7B-Instruct']='../Models/LLMs/Qwen2.5-7B-Instruct'
# -------------Edit here-------------
export CUDA_VISIBLE_DEVICES=4,5

model='Mistral-7B-Instruct'
# model='Llama-3-8B-Instruct'
# model='Qwen2.5-7B-Instruct'
# -------------Edit here-------------

# precompute
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

    "needle"

    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multiquery"
    "niah_multivalue"
    "cwe"
    "fwe"
    "vt"

)

for dataset in "${dataset_list[@]}"; do

    output_dir=./kvs/${model}/${dataset}
    mkdir -p ${output_dir}

    python ./precompute.py \
        --model ${path_map[$model]} \
        --kv_path ${output_dir} \
        --dataset ${dataset} \

    wait
done