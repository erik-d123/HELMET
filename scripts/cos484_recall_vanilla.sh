#!/bin/bash -l
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-output/Llama-3.1-8B-Instruct}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"

[[ -f env/bin/activate ]] && source env/bin/activate
[[ -f scripts/neuronic_stage.sh ]] && source scripts/neuronic_stage.sh
mkdir -p "$OUTPUT_DIR"

for LENGTH in 8k 32k 64k; do
  python eval.py \
    --config "configs/cos484_recall_${LENGTH}.yaml" \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --tag "cos484_recall_vanilla_${LENGTH}" \
    --max_test_samples "$MAX_SAMPLES" \
    --seed 42 \
    --use_chat_template False \
    --cd_mode off
done
