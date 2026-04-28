#!/bin/bash -l
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-output/Llama-3.1-8B-Instruct}"
CONFIG="${CONFIG:-configs/cos484_rag_32k.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"

[[ -f env/bin/activate ]] && source env/bin/activate
[[ -f scripts/neuronic_stage.sh ]] && source scripts/neuronic_stage.sh
mkdir -p "$OUTPUT_DIR" joblog

COMMON_ARGS=(
  --config "$CONFIG"
  --model_name_or_path "$MODEL"
  --output_dir "$OUTPUT_DIR"
  --max_test_samples "$MAX_SAMPLES"
  --seed 42
  --use_chat_template False
)

python eval.py "${COMMON_ARGS[@]}" --tag cos484_smoke_vanilla --cd_mode off
python eval.py "${COMMON_ARGS[@]}" --tag cos484_smoke_cad --cd_mode cad --cd_alpha 1.0
python eval.py "${COMMON_ARGS[@]}" --tag cos484_smoke_lw2k --cd_mode local_window --cd_alpha 1.0 --cd_window_tokens 2000
python eval.py "${COMMON_ARGS[@]}" --tag cos484_smoke_trunc2k --context_mode truncate_last --context_window_tokens 2000
python eval.py "${COMMON_ARGS[@]}" --tag cos484_smoke_oracle --context_mode oracle_passages_only
