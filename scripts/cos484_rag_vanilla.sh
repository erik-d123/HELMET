#!/bin/bash -l
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-output/Llama-3.1-8B-Instruct}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
LENGTHS="${LENGTHS:-8k 32k}"
ENV_DIR="${ENV_DIR:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"

if [[ -n "$ENV_DIR" && -f "$ENV_DIR/bin/activate" ]]; then
  source "$ENV_DIR/bin/activate"
elif [[ -f env/bin/activate ]]; then
  source env/bin/activate
fi
mkdir -p "$OUTPUT_DIR"

BASE_ARGS=(
  --model_name_or_path "$MODEL"
  --output_dir "$OUTPUT_DIR"
  --max_test_samples "$MAX_SAMPLES"
  --num_workers "$NUM_WORKERS"
  --seed 42
  --use_chat_template False
)
if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
  BASE_ARGS+=(--attn_implementation "$ATTN_IMPLEMENTATION")
fi
if [[ "$NO_TORCH_COMPILE" == "1" ]]; then
  BASE_ARGS+=(--no_torch_compile)
fi

for LENGTH in $LENGTHS; do
  python eval.py \
    --config "configs/cos484_rag_${LENGTH}.yaml" \
    "${BASE_ARGS[@]}" \
    --tag "cos484_rag_vanilla_${LENGTH}" \
    --cd_mode off
done
