#!/bin/bash -l
#SBATCH --job-name=cos484-smoke
#SBATCH --output=joblog/%x-%j.out
#SBATCH --error=joblog/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=lowprio
#SBATCH --account=seas

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/helmet-cd-cos484-clean}"
ENV_DIR="${ENV_DIR:-/scratch/$USER/helmet-cd-env}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-output/Llama-3.1-8B-Instruct}"
CONFIG="${CONFIG:-configs/cos484_rag_8k.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}"

export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export HF_HOME="${HF_HOME:-/scratch/$USER/helmet-cd-cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
mkdir -p "$TMPDIR" "$HF_HOME"

cd "$REPO_DIR"
mkdir -p joblog "$OUTPUT_DIR"

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Repo              = $REPO_DIR"
echo "Config            = $CONFIG"
echo "Model             = $MODEL"
echo "Output            = $OUTPUT_DIR"
echo "Attention         = $ATTN_IMPLEMENTATION"
echo "Torch compile off = $NO_TORCH_COMPILE"

ENV_DIR="$ENV_DIR" \
MODEL="$MODEL" \
OUTPUT_DIR="$OUTPUT_DIR" \
CONFIG="$CONFIG" \
MAX_SAMPLES="$MAX_SAMPLES" \
ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
NO_TORCH_COMPILE="$NO_TORCH_COMPILE" \
bash scripts/cos484_smoke.sh
