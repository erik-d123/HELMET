#!/bin/bash -l
# Phase B — alpha sweep for CAD on RAG @ 32K, n=100.
# Picks alpha for the main Phase B/C runs.
#
# Usage: sbatch scripts/run_cd_alpha_sweep.sh
# or:    bash scripts/run_cd_alpha_sweep.sh (interactive)

#SBATCH --job-name=cd-alpha-sweep
#SBATCH --output=./joblog/%x-%j.out
#SBATCH --error=./joblog/%x-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80

set -euo pipefail

echo "Date=$(date)  Host=$(hostname -s)  PWD=$(pwd)"
source env/bin/activate
export OMP_NUM_THREADS=8

MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR="output/Llama-3.1-8B-Instruct"
mkdir -p "$OUTPUT_DIR" joblog

for ALPHA in 0.5 1.0 1.5; do
    TAG="phaseB_alpha${ALPHA}_cad_32k_n100"
    echo ">>> alpha=${ALPHA}"
    python eval.py \
        --config configs/cd_rag.yaml \
        --model_name_or_path "$MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --tag "$TAG" \
        --input_max_length 32768 \
        --max_test_samples 100 \
        --cd_mode cad \
        --cd_alpha "$ALPHA" \
        --seed 42 \
        --use_chat_template False
done

echo "Alpha sweep scores:"
for f in "$OUTPUT_DIR"/*phaseB_alpha*.score; do
    [[ -e "$f" ]] || continue
    echo "--- $f ---"
    cat "$f"
done
