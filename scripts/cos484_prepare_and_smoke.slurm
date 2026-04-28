#!/bin/bash -l
#SBATCH --job-name=cos484-prep-smoke
#SBATCH --output=joblog/%x-%j.out
#SBATCH --error=joblog/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=lowprio
#SBATCH --account=seas

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/helmet-cd-cos484-clean}"
ENV_DIR="${ENV_DIR:-/scratch/$USER/helmet-cd-job-env}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/helmet-cd-data}"
DATA_ARCHIVE="${DATA_ARCHIVE:-/scratch/$USER/helmet_data.tar.gz}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-output/Llama-3.1-8B-Instruct}"
CONFIG="${CONFIG:-configs/cos484_rag_8k.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}"

export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export HF_HOME="${HF_HOME:-/scratch/$USER/helmet-cd-cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/helmet-cd-cache/pip}"
mkdir -p "$TMPDIR" "$HF_HOME" "$DATA_ROOT/data/kilt"
if [[ -f "$HOME/.cache/huggingface/token" && ! -f "$HF_HOME/token" ]]; then
  cp "$HOME/.cache/huggingface/token" "$HF_HOME/token"
  chmod 600 "$HF_HOME/token"
fi
if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
  echo "Creating node-local Python environment at $ENV_DIR"
  python -m venv "$ENV_DIR"
  source "$ENV_DIR/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install -r "$REPO_DIR/requirements.txt"
else
  source "$ENV_DIR/bin/activate"
fi
python - <<'PY'
import torch
import transformers
print("torch", torch.__version__)
print("transformers", transformers.__version__)
PY

cd "$DATA_ROOT"
if [[ ! -f data/kilt/nq-dev-multikilt_1000_k50_dep6.jsonl ]]; then
  echo "Extracting required KILT files from $DATA_ARCHIVE"
  tar -xzf "$DATA_ARCHIVE" \
    data/kilt/nq-dev-multikilt_1000_k50_dep6.jsonl \
    data/kilt/triviaqa-dev-multikilt_1000_k50_dep6.jsonl \
    data/kilt/hotpotqa-dev-multikilt_1000_k50_dep3.jsonl \
    data/kilt/popqa_test_1000_k50_dep6.jsonl \
    data/kilt/nq-train-multikilt_1000_k3_dep6.jsonl \
    data/kilt/triviaqa-train-multikilt_1000_k3_dep6.jsonl \
    data/kilt/hotpotqa-train-multikilt_1000_k3_dep3.jsonl \
    data/kilt/popqa_test_1000_k3_dep6.jsonl
fi

cd "$REPO_DIR"
mkdir -p data
rm -rf data/kilt
ln -s "$DATA_ROOT/data/kilt" data/kilt
mkdir -p joblog "$OUTPUT_DIR"

ENV_DIR="$ENV_DIR" \
MODEL="$MODEL" \
OUTPUT_DIR="$OUTPUT_DIR" \
CONFIG="$CONFIG" \
MAX_SAMPLES="$MAX_SAMPLES" \
ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
NO_TORCH_COMPILE="$NO_TORCH_COMPILE" \
bash scripts/cos484_smoke.sh
