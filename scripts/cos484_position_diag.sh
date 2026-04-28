#!/bin/bash -l
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-output/Llama-3.1-8B-Instruct}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
ENV_DIR="${ENV_DIR:-}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"

if [[ -n "$ENV_DIR" && -f "$ENV_DIR/bin/activate" ]]; then
  source "$ENV_DIR/bin/activate"
elif [[ -f env/bin/activate ]]; then
  source env/bin/activate
fi
mkdir -p "$OUTPUT_DIR" analysis/cos484

EXTRA_ARGS=()
if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
  EXTRA_ARGS+=(--attn_implementation "$ATTN_IMPLEMENTATION")
fi
if [[ "$NO_TORCH_COMPILE" == "1" ]]; then
  EXTRA_ARGS+=(--no_torch_compile)
fi

python eval.py \
  --config configs/cos484_rag_32k.yaml \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --tag cos484_diag_vanilla_32k \
  --max_test_samples "$MAX_SAMPLES" \
  --num_workers "$NUM_WORKERS" \
  --seed 42 \
  --use_chat_template False \
  --cd_mode off \
  "${EXTRA_ARGS[@]}"

python scripts/cos484_analyze.py \
  --input_dir "$OUTPUT_DIR" \
  --output_dir analysis/cos484 \
  --tag_substring cos484_diag_vanilla_32k \
  --position_length 32768
