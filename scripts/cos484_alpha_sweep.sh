#!/bin/bash -l
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-output/Llama-3.1-8B-Instruct}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
ALPHAS=(${ALPHAS:-0.5 1.0 1.5})
METHODS=(${METHODS:-cad lw2k lw8k})
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

for METHOD in "${METHODS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    case "$METHOD" in
      cad)
        python eval.py \
          --config configs/cos484_rag_32k.yaml \
          "${BASE_ARGS[@]}" \
          --tag "cos484_alpha_cad_a${ALPHA}_32k" \
          --cd_mode cad \
          --cd_alpha "$ALPHA"
        ;;
      lw2k)
        python eval.py \
          --config configs/cos484_rag_32k.yaml \
          "${BASE_ARGS[@]}" \
          --tag "cos484_alpha_lw2k_a${ALPHA}_32k" \
          --cd_mode local_window \
          --cd_alpha "$ALPHA" \
          --cd_window_tokens 2000
        ;;
      lw8k)
        python eval.py \
          --config configs/cos484_rag_32k.yaml \
          "${BASE_ARGS[@]}" \
          --tag "cos484_alpha_lw8k_a${ALPHA}_32k" \
          --cd_mode local_window \
          --cd_alpha "$ALPHA" \
          --cd_window_tokens 8000
        ;;
      *)
        echo "Unknown method '$METHOD'. Expected one of: cad lw2k lw8k" >&2
        exit 1
        ;;
    esac
  done
done
