#!/bin/bash -l
# Phase A — reproduce vanilla Llama-3.1-8B-Instruct RAG baseline on n=50.
# HF path (no vLLM), greedy deterministic, input_max_length=32K.
# Gate: accuracy within ~2 pts of HELMET's published Llama-3.1-8B-Instruct RAG number.

#SBATCH --job-name=cd-phaseA
#SBATCH --output=./joblog/%x-%j.out
#SBATCH --error=./joblog/%x-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80

set -euo pipefail

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"

source env/bin/activate
export OMP_NUM_THREADS=8

MODEL="meta-llama/Llama-3.1-8B-Instruct"
TAG="phaseA_vanilla_32k_n50"
OUTPUT_DIR="output/Llama-3.1-8B-Instruct"

mkdir -p "$OUTPUT_DIR" joblog

python eval.py \
    --config configs/rag.yaml \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --tag "$TAG" \
    --input_max_length 32768 \
    --max_test_samples 50 \
    --seed 42 \
    --use_chat_template False

echo "Scores:"
ls -1 "$OUTPUT_DIR"/*"$TAG"*.score 2>/dev/null | while read f; do
    echo "--- $f ---"
    cat "$f"
done
