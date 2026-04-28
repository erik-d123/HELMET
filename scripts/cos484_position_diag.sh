#!/bin/bash -l
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-output/Llama-3.1-8B-Instruct}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"

[[ -f env/bin/activate ]] && source env/bin/activate
[[ -f scripts/neuronic_stage.sh ]] && source scripts/neuronic_stage.sh
mkdir -p "$OUTPUT_DIR" analysis/cos484

python eval.py \
  --config configs/cos484_rag_32k.yaml \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --tag cos484_diag_vanilla_32k \
  --max_test_samples "$MAX_SAMPLES" \
  --seed 42 \
  --use_chat_template False \
  --cd_mode off

python scripts/cos484_analyze.py \
  --input_dir "$OUTPUT_DIR" \
  --output_dir analysis/cos484 \
  --tag_substring cos484_diag_vanilla_32k \
  --position_length 32768
