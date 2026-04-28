#!/bin/bash -l
#SBATCH --job-name=cos484-run
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
RUN_SCRIPT="${RUN_SCRIPT:-}"
RUN_ARGS="${RUN_ARGS:-}"
CONFIG_PATHS="${CONFIG_PATHS:-}"

if [[ -z "$RUN_SCRIPT" ]]; then
  echo "RUN_SCRIPT must be set" >&2
  exit 1
fi

export TMPDIR="${TMPDIR:-/scratch/$USER/tmp}"
export HF_HOME="${HF_HOME:-/scratch/$USER/helmet-cd-cache/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/helmet-cd-cache/pip}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}"
mkdir -p "$TMPDIR" "$HF_HOME" "$DATA_ROOT/data"
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

if [[ -n "$CONFIG_PATHS" ]]; then
  cd "$REPO_DIR"
  mapfile -t REQUIRED_FILES < <(
    python - <<'PY'
from pathlib import Path
import os

raw = os.environ.get("CONFIG_PATHS", "")
if "|" in raw:
    configs = [p for p in raw.split("|") if p]
else:
    configs = [p for p in raw.split(",") if p]
keys = {"test_files", "demo_files"}
seen = set()
for config_path in configs:
    text = Path(config_path).read_text()
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in keys:
            continue
        for part in value.strip().split(","):
            part = part.strip().strip("'").strip('"')
            if not part:
                continue
            if part == ".":
                continue
            if part not in seen:
                seen.add(part)
                print(part)
PY
  )

  if ((${#REQUIRED_FILES[@]} > 0)); then
    cd "$DATA_ROOT"
    MISSING_FILES=()
    for relpath in "${REQUIRED_FILES[@]}"; do
      if [[ ! -e "$relpath" ]]; then
        MISSING_FILES+=("$relpath")
      fi
    done
    if ((${#MISSING_FILES[@]} > 0)); then
      echo "Extracting ${#MISSING_FILES[@]} required files from $DATA_ARCHIVE"
      tar -xzf "$DATA_ARCHIVE" "${MISSING_FILES[@]}"
    fi
  fi
fi

cd "$REPO_DIR"
mkdir -p data joblog
for subdir in kilt json_kv ruler; do
  if [[ -d "$DATA_ROOT/data/$subdir" ]]; then
    rm -rf "data/$subdir"
    ln -s "$DATA_ROOT/data/$subdir" "data/$subdir"
  fi
done

FULL_RUN_SCRIPT="$REPO_DIR/$RUN_SCRIPT"
if [[ ! -f "$FULL_RUN_SCRIPT" ]]; then
  echo "Run script not found: $FULL_RUN_SCRIPT" >&2
  exit 1
fi

ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" \
NO_TORCH_COMPILE="$NO_TORCH_COMPILE" \
ENV_DIR="$ENV_DIR" \
bash "$FULL_RUN_SCRIPT" $RUN_ARGS
