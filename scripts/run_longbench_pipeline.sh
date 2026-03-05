#!/usr/bin/env bash
set -euo pipefail

# 解析仓库根目录，保证脚本可在任意工作目录启动。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# precompute.py / eval_longbench.py 当前支持的 LongBench 子集。
LONG_BENCH_DATASETS=(
  "qasper"
  "multifieldqa_en"
  "hotpotqa"
  "2wikimqa"
  # "gov_report"
  # "multi_news"
  "trec"
  "triviaqa"
  "samsum"
  "passage_count"
  "lcc"
)

declare -A MODEL_ALIAS_MAP
MODEL_ALIAS_MAP['Mistral-7B-Instruct']='../Models/LLMs/Mistral-7B-Instruct-v0.2'
MODEL_ALIAS_MAP['Llama-3-8B-Instruct']='/data/ykw/models/Meta-Llama-3-8B-Instruct'
MODEL_ALIAS_MAP['Qwen2.5-7B-Instruct']='../Models/LLMs/Qwen2.5-7B-Instruct'

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_longbench_pipeline.sh \
    --model <model_path_or_alias> \
    --dataset <dataset|a,b,c|all> \
    --reuse <reuse_method> \
    --rate <float> \
    [--blend_gap_source <v|k>] \
    [--blend_debug_fusion <mul|sum|rank>] \
    [--resume <true|false>] \
    [--drop <False|Streaming|SnapKV...>] \
    [--drop_config <path|None>] \
    [--max_length <int>] \
    [--chunk_size <int>] \
    [--cuda_visible_devices <ids>] \
    [--dry_run]

Examples:
  bash scripts/run_longbench_pipeline.sh \
    --model Llama-3-8B-Instruct \
    --dataset all \
    --reuse blend_debug \
    --blend_gap_source k \
    --rate 0.15 \
    --cuda_visible_devices 1

  bash scripts/run_longbench_pipeline.sh \
    --model /data/ykw/models/Meta-Llama-3-8B-Instruct \
    --dataset all \
    --reuse surprisal_chunk \
    --rate 0.15 \
    --dry_run

  bash scripts/run_longbench_pipeline.sh \
    --model Llama-3-8B-Instruct \
    --dataset all \
    --reuse tail_ratio \
    --rate 0.15 \
    --cuda_visible_devices 1 \
    --dry_run
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

run_cmd() {
  # 统一命令执行入口：支持 --dry_run 仅打印不执行。
  local cmd=("$@")
  printf '+ '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  "${cmd[@]}"
}

is_supported_dataset() {
  # 数据集在支持列表中时返回 0。
  local ds="$1"
  local candidate
  for candidate in "${LONG_BENCH_DATASETS[@]}"; do
    if [[ "${candidate}" == "${ds}" ]]; then
      return 0
    fi
  done
  return 1
}

chunk_count_from_json() {
  # 统计 inputs/<model>/<dataset>.json 中的样本数。
  local chunk_json="$1"
  python - "${chunk_json}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
if not isinstance(data, list):
    raise ValueError(f'{path} is not a JSON list')
print(len(data))
PY
}

needs_rechunk_dataset() {
  # 以下任一情况触发重切：
  # 1) chunk 文件不存在
  # 2) chunk JSON 无法解析
  # 3) chunk 样本数与原始 *_e.jsonl 行数不一致
  local dataset="$1"
  local raw_file="./data/longbench/${dataset}_e.jsonl"
  local chunk_file="${INPUT_ROOT}/${MODEL_BASENAME}/${dataset}.json"

  if [[ ! -f "${raw_file}" ]]; then
    die "missing raw file: ${raw_file}"
  fi

  if [[ ! -f "${chunk_file}" ]]; then
    return 0
  fi

  local raw_count
  raw_count=$(wc -l < "${raw_file}")

  local chunk_count
  if ! chunk_count=$(chunk_count_from_json "${chunk_file}"); then
    return 0
  fi

  if [[ "${raw_count}" != "${chunk_count}" ]]; then
    return 0
  fi

  return 1
}

MODEL_ARG=""
DATASET_ARG=""
REUSE=""
RATE=""
BLEND_GAP_SOURCE="v"
BLEND_DEBUG_FUSION="mul"
DROP="False"
DROP_CONFIG="None"
RESUME="true"
MAX_LENGTH="7000"
CHUNK_SIZE="512"
CUDA_VISIBLE_DEVICES_ARG=""
DRY_RUN="false"
INPUT_ROOT="./inputs"
KV_ROOT="./kvs"
OUTPUT_ROOT="./outputs"

while [[ $# -gt 0 ]]; do
  # 解析命令行参数。
  case "$1" in
    --model)
      MODEL_ARG="${2:-}"
      shift 2
      ;;
    --dataset)
      DATASET_ARG="${2:-}"
      shift 2
      ;;
    --reuse)
      REUSE="${2:-}"
      shift 2
      ;;
    --rate)
      RATE="${2:-}"
      shift 2
      ;;
    --blend_gap_source)
      BLEND_GAP_SOURCE="${2:-}"
      shift 2
      ;;
    --blend_debug_fusion)
      BLEND_DEBUG_FUSION="${2:-}"
      shift 2
      ;;
    --resume)
      RESUME="${2:-}"
      shift 2
      ;;
    --drop)
      DROP="${2:-}"
      shift 2
      ;;
    --drop_config)
      DROP_CONFIG="${2:-}"
      shift 2
      ;;
    --max_length)
      MAX_LENGTH="${2:-}"
      shift 2
      ;;
    --chunk_size)
      CHUNK_SIZE="${2:-}"
      shift 2
      ;;
    --cuda_visible_devices)
      CUDA_VISIBLE_DEVICES_ARG="${2:-}"
      shift 2
      ;;
    --dry_run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -z "${MODEL_ARG}" ]] && die "--model is required"
[[ -z "${DATASET_ARG}" ]] && die "--dataset is required"
[[ -z "${REUSE}" ]] && die "--reuse is required"
[[ -z "${RATE}" ]] && die "--rate is required"

[[ "${BLEND_GAP_SOURCE}" =~ ^(v|k)$ ]] || die "--blend_gap_source must be one of: v,k"
[[ "${BLEND_DEBUG_FUSION}" =~ ^(mul|sum|rank)$ ]] || die "--blend_debug_fusion must be one of: mul,sum,rank"
[[ "${RESUME}" =~ ^(true|false|True|False|1|0)$ ]] || die "--resume must be one of: true,false,True,False,1,0"
[[ "${MAX_LENGTH}" =~ ^[0-9]+$ ]] || die "--max_length must be a positive integer"
[[ "${CHUNK_SIZE}" =~ ^[0-9]+$ ]] || die "--chunk_size must be a positive integer"
[[ "${MAX_LENGTH}" -gt 0 ]] || die "--max_length must be > 0"
[[ "${CHUNK_SIZE}" -gt 0 ]] || die "--chunk_size must be > 0"
[[ "${RATE}" =~ ^[0-9]*\.?[0-9]+$ ]] || die "--rate must be a float like 0.15"

# 兼容模型别名与模型绝对路径两种输入。
MODEL_PATH="${MODEL_ALIAS_MAP[${MODEL_ARG}]:-${MODEL_ARG}}"
MODEL_BASENAME="$(basename "${MODEL_PATH}")"

# 可选运行环境变量覆盖。
if [[ -n "${CUDA_VISIBLE_DEVICES_ARG}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_ARG}"
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

DATASETS=()
if [[ "${DATASET_ARG}" == "all" ]]; then
  # 展开为预定义 LongBench 数据集列表。
  DATASETS=("${LONG_BENCH_DATASETS[@]}")
else
  # 解析逗号分隔数据集并去重。
  declare -A seen=()
  IFS=',' read -r -a raw_datasets <<< "${DATASET_ARG}"
  for raw_ds in "${raw_datasets[@]}"; do
    ds="${raw_ds//[[:space:]]/}"
    [[ -z "${ds}" ]] && continue
    if ! is_supported_dataset "${ds}"; then
      die "unsupported dataset '${ds}'. supported: ${LONG_BENCH_DATASETS[*]}"
    fi
    if [[ -z "${seen[${ds}]+x}" ]]; then
      DATASETS+=("${ds}")
      seen["${ds}"]=1
    fi
  done
fi

[[ ${#DATASETS[@]} -gt 0 ]] || die "--dataset produced empty dataset set"

RATE_TAG="${RATE}"
OUTPUT_TAG="${REUSE}_rate${RATE_TAG}"
if [[ "${REUSE}" == "blend" || "${REUSE}" == "blend_debug" ]]; then
  OUTPUT_TAG="${OUTPUT_TAG}_gap${BLEND_GAP_SOURCE}"
fi
if [[ "${DROP}" != "False" ]]; then
  # 开启 drop 时追加标签，避免不同配置结果互相覆盖。
  DROP_TAG="${DROP//\//_}"
  DROP_TAG="${DROP_TAG// /_}"
  OUTPUT_TAG="${OUTPUT_TAG}_${DROP_TAG}"
  if [[ "${DROP_CONFIG}" != "None" && "${DROP_CONFIG}" != "" ]]; then
    DROP_CFG_BASE="$(basename "${DROP_CONFIG}")"
    DROP_CFG_BASE="${DROP_CFG_BASE%.*}"
    OUTPUT_TAG="${OUTPUT_TAG}_${DROP_CFG_BASE}"
  fi
fi

log "pipeline start"
log "model_arg=${MODEL_ARG} model_path=${MODEL_PATH} model_basename=${MODEL_BASENAME}"
log "datasets=${DATASETS[*]}"
log "reuse=${REUSE} rate=${RATE} drop=${DROP} drop_config=${DROP_CONFIG}"
log "blend_gap_source=${BLEND_GAP_SOURCE} blend_debug_fusion=${BLEND_DEBUG_FUSION}"
log "resume=${RESUME}"
log "max_length=${MAX_LENGTH} chunk_size=${CHUNK_SIZE} dry_run=${DRY_RUN}"

for dataset in "${DATASETS[@]}"; do
  log "========== dataset=${dataset} =========="

  # 步骤 1：先做完整性校验，不完整才重跑 chunk。
  if needs_rechunk_dataset "${dataset}"; then
    log "chunk missing/incomplete, rerun chunk_longbench.py for ${dataset}"
    run_cmd python ./chunk_longbench.py \
      --model "${MODEL_PATH}" \
      --dataset "${dataset}" \
      --max_length "${MAX_LENGTH}" \
      --chunk_size "${CHUNK_SIZE}" \
      --output_root "${INPUT_ROOT}"
  else
    log "chunk complete, reuse existing chunk for ${dataset}"
  fi

  # 步骤 2：预计算 KV；surprisal_chunk 模式自动补充 surprisal 文件。
  KV_DIR="${KV_ROOT}/${MODEL_BASENAME}/${dataset}"
  run_cmd mkdir -p "${KV_DIR}"

  PRECOMPUTE_CMD=(
    python ./precompute.py
    --model "${MODEL_PATH}"
    --kv_path "${KV_DIR}"
    --dataset "${dataset}"
  )
  if [[ "${REUSE}" == "surprisal_chunk" ]]; then
    PRECOMPUTE_CMD+=(--save_surprisal)
  fi
  log "precompute ${dataset}"
  run_cmd "${PRECOMPUTE_CMD[@]}"

  # 步骤 3：执行评测，输出目录按 reuse+rate（及 drop）隔离。
  OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_BASENAME}/${OUTPUT_TAG}/${dataset}"
  run_cmd mkdir -p "${OUTPUT_DIR}"

  EVAL_CMD=(
    python ./eval_longbench.py
    --model "${MODEL_PATH}"
    --reuse "${REUSE}"
    --blend_gap_source "${BLEND_GAP_SOURCE}"
    --blend_debug_fusion "${BLEND_DEBUG_FUSION}"
    --resume "${RESUME}"
    --output_path "${OUTPUT_DIR}"
    --dataset "${dataset}"
    --kv_path "${KV_DIR}"
    --drop "${DROP}"
    --drop_config "${DROP_CONFIG}"
    --rate "${RATE}"
  )
  log "eval ${dataset}"
  run_cmd "${EVAL_CMD[@]}"
done

log "pipeline done"
