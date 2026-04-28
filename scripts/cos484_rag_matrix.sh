#!/bin/bash -l
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-output/Llama-3.1-8B-Instruct}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
CAD_ALPHA="${CAD_ALPHA:-1.0}"
LW2K_ALPHA="${LW2K_ALPHA:-1.0}"
LW8K_ALPHA="${LW8K_ALPHA:-1.0}"

[[ -f env/bin/activate ]] && source env/bin/activate
[[ -f scripts/neuronic_stage.sh ]] && source scripts/neuronic_stage.sh
mkdir -p "$OUTPUT_DIR"

for LENGTH in 8k 32k 64k; do
  CONFIG="configs/cos484_rag_${LENGTH}.yaml"

  python eval.py --config "$CONFIG" --model_name_or_path "$MODEL" --output_dir "$OUTPUT_DIR" --tag "cos484_vanilla_${LENGTH}" --max_test_samples "$MAX_SAMPLES" --seed 42 --use_chat_template False --cd_mode off
  python eval.py --config "$CONFIG" --model_name_or_path "$MODEL" --output_dir "$OUTPUT_DIR" --tag "cos484_cad_${LENGTH}" --max_test_samples "$MAX_SAMPLES" --seed 42 --use_chat_template False --cd_mode cad --cd_alpha "$CAD_ALPHA"
  python eval.py --config "$CONFIG" --model_name_or_path "$MODEL" --output_dir "$OUTPUT_DIR" --tag "cos484_lw2k_${LENGTH}" --max_test_samples "$MAX_SAMPLES" --seed 42 --use_chat_template False --cd_mode local_window --cd_alpha "$LW2K_ALPHA" --cd_window_tokens 2000
  python eval.py --config "$CONFIG" --model_name_or_path "$MODEL" --output_dir "$OUTPUT_DIR" --tag "cos484_lw8k_${LENGTH}" --max_test_samples "$MAX_SAMPLES" --seed 42 --use_chat_template False --cd_mode local_window --cd_alpha "$LW8K_ALPHA" --cd_window_tokens 8000
  python eval.py --config "$CONFIG" --model_name_or_path "$MODEL" --output_dir "$OUTPUT_DIR" --tag "cos484_trunc2k_${LENGTH}" --max_test_samples "$MAX_SAMPLES" --seed 42 --use_chat_template False --context_mode truncate_last --context_window_tokens 2000
  python eval.py --config "$CONFIG" --model_name_or_path "$MODEL" --output_dir "$OUTPUT_DIR" --tag "cos484_trunc8k_${LENGTH}" --max_test_samples "$MAX_SAMPLES" --seed 42 --use_chat_template False --context_mode truncate_last --context_window_tokens 8000
  python eval.py --config "$CONFIG" --model_name_or_path "$MODEL" --output_dir "$OUTPUT_DIR" --tag "cos484_oracle_${LENGTH}" --max_test_samples "$MAX_SAMPLES" --seed 42 --use_chat_template False --context_mode oracle_passages_only

  if [[ "$LENGTH" == "32k" ]]; then
    python eval.py --config "$CONFIG" --model_name_or_path "$MODEL" --output_dir "$OUTPUT_DIR" --tag "cos484_shuffled_${LENGTH}" --max_test_samples "$MAX_SAMPLES" --seed 42 --use_chat_template False --cd_mode shuffled --cd_alpha "$CAD_ALPHA"
    python eval.py --config "$CONFIG" --model_name_or_path "$MODEL" --output_dir "$OUTPUT_DIR" --tag "cos484_reversed_${LENGTH}" --max_test_samples "$MAX_SAMPLES" --seed 42 --use_chat_template False --cd_mode reversed --cd_alpha "$CAD_ALPHA"
  fi
done
