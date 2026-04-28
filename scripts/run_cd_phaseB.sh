#!/bin/bash -l
# Phase B — CAD vs vanilla on RAG @ 32K, full test set.
# Pass the chosen alpha from the alpha sweep as env var ALPHA (default 1.0).
#
# Usage: ALPHA=1.0 sbatch scripts/run_cd_phaseB.sh

#SBATCH --job-name=cd-phaseB
#SBATCH --output=./joblog/%x-%j.out
#SBATCH --error=./joblog/%x-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80

set -euo pipefail

echo "Date=$(date)  Host=$(hostname -s)  PWD=$(pwd)"
source env/bin/activate
export OMP_NUM_THREADS=8

MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR="output/Llama-3.1-8B-Instruct"
ALPHA="${ALPHA:-1.0}"
mkdir -p "$OUTPUT_DIR" joblog

# Vanilla baseline on the same sample set (full test set at 32K)
python eval.py \
    --config configs/rag.yaml \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --tag "phaseB_vanilla_32k_full" \
    --input_max_length 32768 \
    --max_test_samples 100 \
    --seed 42 \
    --use_chat_template False

# CAD
python eval.py \
    --config configs/cd_rag.yaml \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --tag "phaseB_cad_a${ALPHA}_32k_full" \
    --input_max_length 32768 \
    --max_test_samples 100 \
    --cd_mode cad \
    --cd_alpha "$ALPHA" \
    --seed 42 \
    --use_chat_template False

echo "Phase B scores:"
for f in "$OUTPUT_DIR"/*phaseB_{vanilla,cad}_*.score; do
    [[ -e "$f" ]] || continue
    echo "--- $f ---"
    cat "$f"
done
