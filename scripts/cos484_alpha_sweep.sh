#!/bin/bash -l
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-output/Llama-3.1-8B-Instruct}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
ALPHAS=(${ALPHAS:-0.5 1.0 1.5})

[[ -f env/bin/activate ]] && source env/bin/activate
[[ -f scripts/neuronic_stage.sh ]] && source scripts/neuronic_stage.sh
mkdir -p "$OUTPUT_DIR"

for ALPHA in "${ALPHAS[@]}"; do
  python eval.py \
    --config configs/cos484_rag_32k.yaml \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --tag "cos484_alpha_cad_a${ALPHA}_32k" \
    --max_test_samples "$MAX_SAMPLES" \
    --seed 42 \
    --use_chat_template False \
    --cd_mode cad \
    --cd_alpha "$ALPHA"

  python eval.py \
    --config configs/cos484_rag_32k.yaml \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --tag "cos484_alpha_lw2k_a${ALPHA}_32k" \
    --max_test_samples "$MAX_SAMPLES" \
    --seed 42 \
    --use_chat_template False \
    --cd_mode local_window \
    --cd_alpha "$ALPHA" \
    --cd_window_tokens 2000

  python eval.py \
    --config configs/cos484_rag_32k.yaml \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --tag "cos484_alpha_lw8k_a${ALPHA}_32k" \
    --max_test_samples "$MAX_SAMPLES" \
    --seed 42 \
    --use_chat_template False \
    --cd_mode local_window \
    --cd_alpha "$ALPHA" \
    --cd_window_tokens 8000
done
